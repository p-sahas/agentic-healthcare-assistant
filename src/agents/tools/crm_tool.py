"""
CRM Tool - Patient lookup, doctor search, booking CRUD.

Exposes 5 actions for the routing engine:
  1. lookup_patient   - find patient by phone/name/ID
  2. search_doctors   - search doctors by specialty/location/availability
  3. create_booking   - book a new appointment
  4. cancel_booking   - cancel an existing booking
  5. reschedule_booking - change date/time of an existing booking

All actions return plain-text summaries for the synthesiser LLM.
"""

from loguru import logger
import uuid
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from sqlalchemy import text, and_
from sqlalchemy.orm import sessionmaker

from infrastructure.db import get_sql_engine
from infrastructure.db.crm_models import (
    Booking,
    Doctor,
    Location,
    Patient,
    Specialty,
)
from infrastructure.observability import observe, update_current_observation


class CRMTool:
    """
    CRM tool for the routing-engine agent.

    Each public method corresponds to one routable action.
    All methods return a human-readable string (never raw dicts).
    """

    def __init__(self) -> None:
        self.engine = get_sql_engine()

    # helpers

    def _session(self):
        """Create a new SQLAlchemy session."""
        factory = sessionmaker(bind=self.engine)
        return factory()

    @staticmethod
    def _epoch_to_str(epoch: int) -> str:
        """Convert epoch seconds to a readable date-time string."""
        return datetime.fromtimestamp(epoch).strftime("%Y-%m-%d %H:%M")

    # 1. lookup_patient

    def lookup_patient(
        self,
        phone: Optional[str] = None,
        name: Optional[str] = None,
        patient_id: Optional[str] = None,
        external_user_id: Optional[str] = None,
    ) -> str:
        """
        Find a patient record by phone, name, patient_id, or external_user_id.

        Returns a formatted string with patient details + upcoming bookings.
        """
        session = self._session()
        try:
            query = session.query(Patient)

            if patient_id:
                query = query.filter(Patient.patient_id == patient_id)
            elif external_user_id:
                query = query.filter(
                    Patient.external_user_id == external_user_id)
            elif phone:
                query = query.filter(Patient.phone == phone)
            elif name:
                query = query.filter(Patient.full_name.ilike(f"%{name}%"))
            else:
                return "No search criteria provided. Please supply phone, name, or patient ID."

            patients = query.limit(5).all()

            if not patients:
                return "No patient found matching the given criteria."

            lines: List[str] = []
            for pat in patients:
                lines.append(
                    f"• {pat.full_name}  |  Phone: {pat.phone or 'N/A'}  "
                    f"|  DOB: {pat.dob or 'N/A'}  |  ID: {pat.patient_id}"
                )

                # Fetch upcoming bookings
                now_epoch = int(time.time())
                bookings = (
                    session.query(Booking)
                    .filter(
                        and_(
                            Booking.patient_id == pat.patient_id,
                            Booking.start_at >= now_epoch,
                            Booking.status.in_(
                                ["PENDING", "CONFIRMED", "RESCHEDULED"]),
                        )
                    )
                    .order_by(Booking.start_at)
                    .limit(5)
                    .all()
                )

                if bookings:
                    lines.append("  Upcoming bookings:")
                    for bk in bookings:
                        doctor = session.query(Doctor).get(bk.doctor_id)
                        loc = session.query(Location).get(bk.location_id)
                        lines.append(
                            f"    - {self._epoch_to_str(bk.start_at)} → "
                            f"{self._epoch_to_str(bk.end_at)}  "
                            f"| Dr. {doctor.full_name if doctor else 'N/A'} "
                            f"| {loc.name if loc else 'N/A'} "
                            f"| Status: {bk.status}  | Booking ID: {bk.booking_id}"
                        )
                else:
                    lines.append("  No upcoming bookings.")

            return "\n".join(lines)

        except Exception as exc:
            logger.error("lookup_patient failed: {}", exc)
            return f"Error looking up patient: {exc}"
        finally:
            session.close()

    # 2. search_doctors

    def search_doctors(
        self,
        specialty: Optional[str] = None,
        location: Optional[str] = None,
        name: Optional[str] = None,
    ) -> str:
        """
        Search doctors by specialty, location, or name.

        Returns a formatted list of matching doctors.
        """
        session = self._session()
        try:
            query = session.query(Doctor).filter(Doctor.active == 1)

            if specialty:
                query = query.join(Specialty).filter(
                    Specialty.name.ilike(f"%{specialty}%")
                )
            if name:
                query = query.filter(Doctor.full_name.ilike(f"%{name}%"))
            if location:
                # Find doctors who have bookings at this location
                loc_sub = (
                    session.query(Location.location_id)
                    .filter(Location.name.ilike(f"%{location}%"))
                    .subquery()
                )
                booking_docs = (
                    session.query(Booking.doctor_id)
                    .filter(Booking.location_id.in_(loc_sub))
                    .distinct()
                    .subquery()
                )
                query = query.filter(Doctor.doctor_id.in_(booking_docs))

            doctors = query.limit(10).all()

            if not doctors:
                return "No doctors found matching the criteria."

            lines: List[str] = []
            for doc in doctors:
                spec_name = doc.specialty.name if doc.specialty else "General"
                lines.append(
                    f"• Dr. {doc.full_name}  |  Specialty: {spec_name}  "
                    f"|  Phone: {doc.phone or 'N/A'}  |  ID: {doc.doctor_id}"
                )

            return "\n".join(lines)

        except Exception as exc:
            logger.error("search_doctors failed: {}", exc)
            return f"Error searching doctors: {exc}"
        finally:
            session.close()

    # 3. create_booking

    def create_booking(
        self,
        patient_id: str,
        doctor_id: str,
        location_id: str,
        start_time: str,
        duration_minutes: int = 30,
        title: str = "Consultation",
        reason: Optional[str] = None,
    ) -> str:
        """
        Create a new appointment booking.

        Args:
            patient_id: Patient identifier
            doctor_id: Doctor identifier
            location_id: Location identifier
            start_time: ISO-format datetime string (e.g. '2026-02-20 09:00')
            duration_minutes: Duration in minutes (default 30)
            title: Booking title
            reason: Optional reason for the visit

        Returns:
            Confirmation or error message.
        """
        session = self._session()
        try:
            # Validate entities
            patient = session.query(Patient).get(patient_id)
            if not patient:
                return f"Patient {patient_id} not found."
            doctor = session.query(Doctor).get(doctor_id)
            if not doctor:
                return f"Doctor {doctor_id} not found."
            loc = session.query(Location).get(location_id)
            if not loc:
                return f"Location {location_id} not found."

            # Parse time
            try:
                dt = datetime.strptime(start_time, "%Y-%m-%d %H:%M")
            except ValueError:
                return f"Invalid date format '{start_time}'. Use YYYY-MM-DD HH:MM."

            start_epoch = int(dt.timestamp())
            end_epoch = start_epoch + duration_minutes * 60

            # Conflict check
            conflict = (
                session.query(Booking)
                .filter(
                    and_(
                        Booking.doctor_id == doctor_id,
                        Booking.status.in_(["PENDING", "CONFIRMED"]),
                        Booking.start_at < end_epoch,
                        Booking.end_at > start_epoch,
                    )
                )
                .first()
            )
            if conflict:
                return (
                    f"Conflict: Dr. {doctor.full_name} already has a booking at "
                    f"{self._epoch_to_str(conflict.start_at)}–{self._epoch_to_str(conflict.end_at)}. "
                    "Please choose another time."
                )

            booking_id = str(uuid.uuid4())
            now_epoch = int(time.time())

            new_booking = Booking(
                booking_id=booking_id,
                patient_id=patient_id,
                doctor_id=doctor_id,
                location_id=location_id,
                title=title,
                reason=reason,
                start_at=start_epoch,
                end_at=end_epoch,
                status="CONFIRMED",
                source="MEMORY",
                created_at=now_epoch,
                updated_at=now_epoch,
            )
            session.add(new_booking)
            session.commit()

            return (
                f"✅ Booking confirmed!\n"
                f"  Booking ID: {booking_id}\n"
                f"  Patient: {patient.full_name}\n"
                f"  Doctor: Dr. {doctor.full_name} ({doctor.specialty.name if doctor.specialty else 'General'})\n"
                f"  Location: {loc.name}\n"
                f"  Time: {self._epoch_to_str(start_epoch)} → {self._epoch_to_str(end_epoch)}\n"
                f"  Reason: {reason or 'N/A'}"
            )

        except Exception as exc:
            session.rollback()
            logger.error("create_booking failed: {}", exc)
            return f"Error creating booking: {exc}"
        finally:
            session.close()

    #  4. cancel_booking

    def cancel_booking(self, booking_id: str) -> str:
        """Cancel an existing booking by ID."""
        session = self._session()
        try:
            booking = session.query(Booking).get(booking_id)
            if not booking:
                return f"Booking {booking_id} not found."

            if booking.status == "CANCELLED":
                return f"Booking {booking_id} is already cancelled."

            old_status = booking.status
            booking.status = "CANCELLED"
            booking.updated_at = int(time.time())
            session.commit()

            doctor = session.query(Doctor).get(booking.doctor_id)
            return (
                f" Booking cancelled.\n"
                f"  Booking ID: {booking_id}\n"
                f"  Was: {self._epoch_to_str(booking.start_at)} with "
                f"Dr. {doctor.full_name if doctor else 'N/A'}\n"
                f"  Previous status: {old_status}"
            )

        except Exception as exc:
            session.rollback()
            logger.error("cancel_booking failed: {}", exc)
            return f"Error cancelling booking: {exc}"
        finally:
            session.close()

    # 5. reschedule_booking

    def reschedule_booking(
        self,
        booking_id: str,
        new_start_time: str,
        duration_minutes: int = 30,
    ) -> str:
        """
        Reschedule an existing booking to a new time.

        Args:
            booking_id: Existing booking ID
            new_start_time: ISO-format datetime string
            duration_minutes: Duration in minutes (default 30)
        """
        session = self._session()
        try:
            booking = session.query(Booking).get(booking_id)
            if not booking:
                return f"Booking {booking_id} not found."

            if booking.status in ("CANCELLED", "COMPLETED"):
                return f"Cannot reschedule a {booking.status} booking."

            try:
                dt = datetime.strptime(new_start_time, "%Y-%m-%d %H:%M")
            except ValueError:
                return f"Invalid date format '{new_start_time}'. Use YYYY-MM-DD HH:MM."

            new_start_epoch = int(dt.timestamp())
            new_end_epoch = new_start_epoch + duration_minutes * 60

            # Conflict check
            conflict = (
                session.query(Booking)
                .filter(
                    and_(
                        Booking.doctor_id == booking.doctor_id,
                        Booking.booking_id != booking_id,
                        Booking.status.in_(
                            ["PENDING", "CONFIRMED", "RESCHEDULED"]),
                        Booking.start_at < new_end_epoch,
                        Booking.end_at > new_start_epoch,
                    )
                )
                .first()
            )
            if conflict:
                return (
                    f"Conflict: The doctor already has a booking at "
                    f"{self._epoch_to_str(conflict.start_at)}–{self._epoch_to_str(conflict.end_at)}. "
                    "Please choose another time."
                )

            old_time = self._epoch_to_str(booking.start_at)
            booking.start_at = new_start_epoch
            booking.end_at = new_end_epoch
            booking.status = "RESCHEDULED"
            booking.updated_at = int(time.time())
            session.commit()

            doctor = session.query(Doctor).get(booking.doctor_id)
            return (
                f" Booking rescheduled!\n"
                f"  Booking ID: {booking_id}\n"
                f"  Doctor: Dr. {doctor.full_name if doctor else 'N/A'}\n"
                f"  Old time: {old_time}\n"
                f"  New time: {self._epoch_to_str(new_start_epoch)} → "
                f"{self._epoch_to_str(new_end_epoch)}"
            )

        except Exception as exc:
            session.rollback()
            logger.error("reschedule_booking failed: {}", exc)
            return f"Error rescheduling booking: {exc}"
        finally:
            session.close()

    #  dispatch

    @observe(name="crm_dispatch")
    def dispatch(self, action: str, params: Dict[str, Any]) -> str:
        """
        Dispatch a CRM action by name.

        Traced via LangFuse so each CRM call is visible with its
        action type, parameters, and latency.
        """
        handler_map = {
            "lookup_patient": self.lookup_patient,
            "search_doctors": self.search_doctors,
            "create_booking": self.create_booking,
            "cancel_booking": self.cancel_booking,
            "reschedule_booking": self.reschedule_booking,
        }
        handler = handler_map.get(action)
        if not handler:
            return f"Unknown CRM action: {action}. Available: {list(handler_map.keys())}"

        update_current_observation(
            input=f"action={action} params={params}",
        )

        start = time.time()
        result = handler(**params)
        latency_ms = int((time.time() - start) * 1000)

        update_current_observation(
            output=result[:500],
            metadata={"action": action, "latency_ms": latency_ms},
        )

        return result
