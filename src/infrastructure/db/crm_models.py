"""
CRM database models (SQLAlchemy ORM).

Matches the schema defined in sql/crm_schema.sql.
"""

import time
from sqlalchemy import Column, String, Integer, Float, CheckConstraint, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class Location(Base):
    """Healthcare location (hospital, OPD, lab, clinic)."""
    
    __tablename__ = "locations"
    
    location_id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    type = Column(String, nullable=False)
    address = Column(String)
    tz = Column(String, nullable=False)  # IANA timezone
    lat = Column(Float)
    lng = Column(Float)
    active = Column(Integer, nullable=False, default=1)
    created_at = Column(Integer, default=lambda: int(time.time()))
    updated_at = Column(Integer, default=lambda: int(time.time()), onupdate=lambda: int(time.time()))
    
    __table_args__ = (
        CheckConstraint("type IN ('HOSPITAL','OPD','LAB','CLINIC')", name="check_location_type"),
    )
    
    # Relationships
    bookings = relationship("Booking", back_populates="location")
    
    def to_dict(self):
        """Convert to dictionary."""
        return {
            "location_id": self.location_id,
            "name": self.name,
            "type": self.type,
            "address": self.address,
            "tz": self.tz,
            "lat": self.lat,
            "lng": self.lng,
            "active": self.active,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class Specialty(Base):
    """Medical specialty (cardiology, neurology, etc)."""
    
    __tablename__ = "specialties"
    
    specialty_id = Column(String, primary_key=True)
    name = Column(String, nullable=False, unique=True)
    
    # Relationships
    doctors = relationship("Doctor", back_populates="specialty")
    
    def to_dict(self):
        """Convert to dictionary."""
        return {
            "specialty_id": self.specialty_id,
            "name": self.name,
        }


class Doctor(Base):
    """Healthcare provider."""
    
    __tablename__ = "doctors"
    
    doctor_id = Column(String, primary_key=True)
    full_name = Column(String, nullable=False)
    specialty_id = Column(String, ForeignKey("specialties.specialty_id"))
    license_no = Column(String, unique=True)
    phone = Column(String)
    email = Column(String)
    active = Column(Integer, nullable=False, default=1)
    created_at = Column(Integer, default=lambda: int(time.time()))
    updated_at = Column(Integer, default=lambda: int(time.time()), onupdate=lambda: int(time.time()))
    
    # Relationships
    specialty = relationship("Specialty", back_populates="doctors")
    bookings = relationship("Booking", back_populates="doctor")
    
    def to_dict(self):
        """Convert to dictionary."""
        return {
            "doctor_id": self.doctor_id,
            "full_name": self.full_name,
            "specialty_id": self.specialty_id,
            "specialty_name": self.specialty.name if self.specialty else None,
            "license_no": self.license_no,
            "phone": self.phone,
            "email": self.email,
            "active": self.active,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class Patient(Base):
    """Patient record."""
    
    __tablename__ = "patients"
    
    patient_id = Column(String, primary_key=True)
    external_user_id = Column(String, unique=True)  # Link to app's user_id
    full_name = Column(String, nullable=False)
    dob = Column(String)  # ISO format YYYY-MM-DD
    gender = Column(String)
    phone = Column(String)
    email = Column(String)
    notes = Column(Text)
    active = Column(Integer, nullable=False, default=1)
    created_at = Column(Integer, default=lambda: int(time.time()))
    updated_at = Column(Integer, default=lambda: int(time.time()), onupdate=lambda: int(time.time()))
    
    __table_args__ = (
        CheckConstraint("gender IN ('M','F','X')", name="check_patient_gender"),
    )
    
    # Relationships
    bookings = relationship("Booking", back_populates="patient")
    
    def to_dict(self):
        """Convert to dictionary."""
        return {
            "patient_id": self.patient_id,
            "external_user_id": self.external_user_id,
            "full_name": self.full_name,
            "dob": self.dob,
            "gender": self.gender,
            "phone": self.phone,
            "email": self.email,
            "notes": self.notes,
            "active": self.active,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class Booking(Base):
    """Healthcare appointment booking."""
    
    __tablename__ = "bookings"
    
    booking_id = Column(String, primary_key=True)
    patient_id = Column(String, ForeignKey("patients.patient_id"), nullable=False)
    doctor_id = Column(String, ForeignKey("doctors.doctor_id"), nullable=False)
    location_id = Column(String, ForeignKey("locations.location_id"), nullable=False)
    title = Column(String, nullable=False)
    reason = Column(String)
    start_at = Column(Integer, nullable=False)  # Epoch seconds UTC
    end_at = Column(Integer, nullable=False)
    status = Column(String, nullable=False)
    source = Column(String, nullable=False)
    created_at = Column(Integer, default=lambda: int(time.time()))
    updated_at = Column(Integer, default=lambda: int(time.time()), onupdate=lambda: int(time.time()))
    
    __table_args__ = (
        CheckConstraint(
            "status IN ('PENDING','CONFIRMED','RESCHEDULED','CANCELLED','NO_SHOW','COMPLETED')",
            name="check_booking_status"
        ),
        CheckConstraint(
            "source IN ('CRM','MEMORY','MIGRATED','SEED')",
            name="check_booking_source"
        ),
    )
    
    # Relationships
    patient = relationship("Patient", back_populates="bookings")
    doctor = relationship("Doctor", back_populates="bookings")
    location = relationship("Location", back_populates="bookings")
    
    def to_dict(self, include_relations=False):
        """Convert to dictionary."""
        result = {
            "booking_id": self.booking_id,
            "patient_id": self.patient_id,
            "doctor_id": self.doctor_id,
            "location_id": self.location_id,
            "title": self.title,
            "reason": self.reason,
            "start_at": self.start_at,
            "end_at": self.end_at,
            "status": self.status,
            "source": self.source,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
        
        if include_relations:
            result["patient_name"] = self.patient.full_name if self.patient else None
            result["doctor_name"] = self.doctor.full_name if self.doctor else None
            result["location_name"] = self.location.name if self.location else None
            result["specialty"] = self.doctor.specialty.name if self.doctor and self.doctor.specialty else None
        
        return result
