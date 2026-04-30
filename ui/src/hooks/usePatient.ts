import { useCallback, useEffect, useState } from "react";
import { ApiError, patientApi } from "@/api/client";
import type { Patient, PatientRegisterPayload, PatientUpdatePayload } from "@/types";

const LS_KEY = "nawaloka.patient";

/**
 * Phone-based identity (no auth). The chosen patient is persisted in
 * ``localStorage`` so refreshes don't kick the user back to the gate.
 *
 * On mount, if a patient is cached, we re-fetch by id to pick up profile
 * edits made elsewhere — non-blocking, falls back to the cached copy on
 * network failure.
 */
export function usePatient() {
  const [patient, setPatient] = useState<Patient | null>(null);
  const [loaded, setLoaded] = useState(false);

  // Hydrate from localStorage on first render. We do NOT background-
  // refetch from the API — every mount in dev StrictMode would fire two
  // requests, and the profile rarely changes anyway. Updates flow back
  // via ``update()`` (which writes through). On full logout/login the
  // fresh row is fetched by the patient gate.
  useEffect(() => {
    try {
      const raw = localStorage.getItem(LS_KEY);
      if (raw) setPatient(JSON.parse(raw) as Patient);
    } catch { /* ignore corrupt cache */ }
    setLoaded(true);
  }, []);

  // Persist on every change.
  useEffect(() => {
    if (!loaded) return;
    try {
      if (patient) localStorage.setItem(LS_KEY, JSON.stringify(patient));
      else localStorage.removeItem(LS_KEY);
    } catch { /* ignore quota errors */ }
  }, [patient, loaded]);

  const login = useCallback(async (phone: string): Promise<{ ok: true } | { ok: false; status: number; message: string }> => {
    try {
      const p = await patientApi.lookup(phone);
      setPatient(p);
      return { ok: true };
    } catch (e) {
      const err = e as ApiError;
      return { ok: false, status: err.status ?? 0, message: err.message };
    }
  }, []);

  const register = useCallback(async (payload: PatientRegisterPayload): Promise<{ ok: true } | { ok: false; status: number; message: string }> => {
    try {
      const p = await patientApi.register(payload);
      setPatient(p);
      return { ok: true };
    } catch (e) {
      const err = e as ApiError;
      return { ok: false, status: err.status ?? 0, message: err.message };
    }
  }, []);

  const update = useCallback(async (payload: PatientUpdatePayload) => {
    if (!patient) return;
    const updated = await patientApi.update(patient.patient_id, payload);
    setPatient(updated);
    return updated;
  }, [patient]);

  const logout = useCallback(() => {
    setPatient(null);
  }, []);

  return { patient, loaded, login, register, update, logout };
}
