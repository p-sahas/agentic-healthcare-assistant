"""
Patient identity endpoints — phone-based "login", no auth.

Four operations:

  POST /patients/lookup        body {phone}     → 200 PatientResponse | 404
  POST /patients/register      body {full_name, phone, dob, gender}
                                                 → 201 PatientResponse | 409
  GET  /patients/{patient_id}                    → 200 PatientResponse | 404
  PUT  /patients/{patient_id}  body {email?, notes?}
                                                 → 200 PatientResponse | 404

Uniqueness is enforced at the database level (``patients.phone`` has a
``UNIQUE`` constraint added via ``scripts/check_patient_uniques.py``).
The 409 response on register exists so the client can show a sensible
"this phone is already registered" message instead of a 500.

Phone normalization:
    Whatever the user types is collapsed to digits-only via
    ``api.utils.normalize_phone`` and stored in
    ``patients.external_user_id`` for indexed lookup. The display form
    (``+94…``) goes into ``patients.phone``.
"""

import asyncio
import time
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from loguru import logger
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from infrastructure.db import get_sql_engine
from infrastructure.db.crm_models import Patient

from api.schemas import (
    PatientLookupRequest,
    PatientRegisterRequest,
    PatientResponse,
    PatientUpdateRequest,
)
from api.utils import display_phone, normalize_phone


router = APIRouter(prefix="/patients", tags=["Patients"])


# ── Helpers ──────────────────────────────────────────────────────────

def _session():
    """One short-lived session per call. Engine is a process-wide singleton."""
    engine = get_sql_engine()
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)()


def _to_response(p: Patient) -> PatientResponse:
    return PatientResponse(
        patient_id=p.patient_id,
        full_name=p.full_name,
        phone=p.phone or "",
        dob=p.dob,
        gender=p.gender,                # type: ignore[arg-type]  Pydantic Literal validates
        email=p.email,
        notes=p.notes,
        active=int(p.active or 0),
        created_at=int(p.created_at or 0),
        updated_at=int(p.updated_at or 0),
    )


def _find_by_phone(canonical: str) -> Optional[Patient]:
    """Sync DB read — call via ``asyncio.to_thread``."""
    s = _session()
    try:
        return s.query(Patient).filter(Patient.external_user_id == canonical).first()
    finally:
        s.close()


def _find_by_id(patient_id: str) -> Optional[Patient]:
    s = _session()
    try:
        return s.query(Patient).filter(Patient.patient_id == patient_id).first()
    finally:
        s.close()


# ── Endpoints ────────────────────────────────────────────────────────

@router.post("/lookup", response_model=PatientResponse)
async def lookup(req: PatientLookupRequest) -> PatientResponse:
    """Find a patient by phone (any common format, normalized server-side)."""
    canonical = normalize_phone(req.phone)
    if not canonical:
        raise HTTPException(status_code=422, detail="Invalid phone number")

    patient = await asyncio.to_thread(_find_by_phone, canonical)
    if patient is None:
        raise HTTPException(status_code=404, detail="No patient with that phone")
    return _to_response(patient)


@router.post("/register", response_model=PatientResponse, status_code=201)
async def register(req: PatientRegisterRequest) -> PatientResponse:
    """Create a new patient row. Returns 409 if the phone is already taken."""
    canonical = normalize_phone(req.phone)
    if not canonical:
        raise HTTPException(status_code=422, detail="Invalid phone number")

    def _insert() -> Patient:
        s = _session()
        try:
            now = int(time.time())
            p = Patient(
                patient_id=str(uuid.uuid4()),
                external_user_id=canonical,
                full_name=req.full_name.strip(),
                dob=req.dob,
                gender=req.gender,
                phone=display_phone(canonical),
                email=None,
                notes=None,
                active=1,
                created_at=now,
                updated_at=now,
            )
            s.add(p)
            s.commit()
            s.refresh(p)
            return p
        finally:
            s.close()

    try:
        patient = await asyncio.to_thread(_insert)
    except IntegrityError as exc:
        # patients.phone has a UNIQUE constraint — this is the dup case.
        logger.info("Patient register conflict for {}: {}", canonical, exc.orig)
        raise HTTPException(status_code=409, detail="That phone is already registered")
    except Exception as exc:
        logger.exception("Patient register failed: {}", exc)
        raise HTTPException(status_code=500, detail=f"Could not register: {exc}")

    return _to_response(patient)


@router.get("/{patient_id}", response_model=PatientResponse)
async def get_patient(patient_id: str) -> PatientResponse:
    """Fetch a full patient profile by id."""
    patient = await asyncio.to_thread(_find_by_id, patient_id)
    if patient is None:
        raise HTTPException(status_code=404, detail="Patient not found")
    return _to_response(patient)


@router.put("/{patient_id}", response_model=PatientResponse)
async def update_patient(
    patient_id: str,
    req: PatientUpdateRequest,
    request: Request,
) -> PatientResponse:
    """Update editable profile fields (``email``, ``notes``)."""

    def _update() -> Optional[Patient]:
        s = _session()
        try:
            p = s.query(Patient).filter(Patient.patient_id == patient_id).first()
            if p is None:
                return None
            if req.email is not None:
                p.email = req.email.strip() or None
            if req.notes is not None:
                p.notes = req.notes.strip() or None
            p.updated_at = int(time.time())
            s.commit()
            s.refresh(p)
            return p
        finally:
            s.close()

    patient = await asyncio.to_thread(_update)
    if patient is None:
        raise HTTPException(status_code=404, detail="Patient not found")

    # Invalidate any warm session cache entries for this patient — the
    # next chat request will re-fetch the fresh profile.
    cache = getattr(request.app.state, "session_cache", None)
    if cache is not None:
        for key in list(cache.keys()):
            if key[0] == patient_id:
                cache.pop(key, None)

    return _to_response(patient)
