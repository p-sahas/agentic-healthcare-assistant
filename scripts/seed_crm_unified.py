"""
Unified CRM Data Seeder - Single script with all capabilities.

Features:
- **SQL-first seeding**: If pre-exported SQL files exist in ``sql/`` they are
  loaded directly — every student gets identical, deterministic data.
- Fallback to LLM / Template generation when SQL files are absent.
- Switch between storage backends (Database vs JSONL).
- Configurable scale parameters and batch generation with progress tracking.
"""

import argparse
import random
import uuid
import time
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional
from enum import Enum
from dataclasses import dataclass
import pytz
import sys
import os

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sqlalchemy.orm import sessionmaker
from sqlalchemy import text

# LangFuse tracing for CRM seeding
from loguru import logger
from infrastructure.log import setup_logging
from infrastructure.observability import observe, update_current_observation, flush


# ============================================================================
# CONFIGURATION ENUMS
# ============================================================================

class DataGenerationMode(Enum):
    """Data generation mode."""
    LLM = "llm"          # Use LLM to generate realistic names/notes
    TEMPLATE = "template"  # Use predefined templates (fast, free)


class StorageBackend(Enum):
    """Storage backend for CRM data."""
    DATABASE = "database"  # SQLite database (relational, indexed)
    JSONL = "jsonl"       # JSON Lines file (simple, portable)


# ============================================================================
# CONFIGURATION DATACLASS
# ============================================================================

@dataclass
class CRMSeederConfig:
    """Configuration for CRM data seeder."""
    
    # Generation mode
    generation_mode: DataGenerationMode = DataGenerationMode.LLM
    storage_backend: StorageBackend = StorageBackend.DATABASE
    
    # Scale parameters
    n_doctors: int = 10
    n_patients: int = 20
    n_specialties: int = 7
    n_locations: int = 4
    n_slots_per_doctor: int = 15
    
    # Scheduling parameters
    start_date: str = datetime.now().strftime("%Y-%m-%d")
    timezone: str = "Asia/Colombo"
    no_overlap: bool = True
    work_hours: tuple = (9, 17)  # 9am to 5pm
    
    # Other parameters
    rand_seed: int = 42
    output_file: Optional[Path] = None  # For JSONL mode
    
    def __post_init__(self):
        """Validate configuration."""
        if self.storage_backend == StorageBackend.JSONL and not self.output_file:
            self.output_file = Path("data/crm_bookings.jsonl")


# ============================================================================
# BASE CLASSES
# ============================================================================

class DataGenerator:
    """Base class for data generators."""
    
    def generate_doctors(self, n: int, specialties: List[str]) -> List[Dict]:
        raise NotImplementedError
    
    def generate_patients(self, n: int) -> List[Dict]:
        raise NotImplementedError
    
    def generate_appointment_reasons(self, n: int, specialty: str) -> List[str]:
        raise NotImplementedError
    
    def generate_medical_notes(self, n: int) -> List[str]:
        raise NotImplementedError


class StorageAdapter:
    """Base class for storage adapters."""
    
    def initialize(self):
        raise NotImplementedError
    
    def store_data(self, data: Dict):
        raise NotImplementedError
    
    def finalize(self):
        raise NotImplementedError


# ============================================================================
# LLM DATA GENERATOR
# ============================================================================

class LLMDataGenerator(DataGenerator):
    """Generate data using LLM (OpenAI). LLM calls are traced via LangFuse."""
    
    def __init__(self):
        from infrastructure.llm import get_chat_llm
        from infrastructure.observability import get_langfuse
        get_langfuse()  # eagerly init so traces are captured
        self.llm = get_chat_llm()
        self._cache = {}
        self.logger = logger
    
    @observe(name="seed_generate_doctors", as_type="generation")
    def generate_doctors(self, n: int, specialties: List[str]) -> List[Dict]:
        """Generate doctor data with Sri Lankan names."""
        cache_key = f"doctors_{n}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        self.logger.info(f"🤖 Generating {n} Sri Lankan doctor names via LLM...")
        
        prompt = f"""Generate {n} realistic Sri Lankan doctor names for a hospital system.

Requirements:
- Mix of Sinhala and Tamil names
- Include appropriate titles (Dr.)
- Mix of male and female names
- Professional medical names

Output as JSON array:
[
  {{"full_name": "Dr. Priya Fernando", "gender": "F"}},
  {{"full_name": "Dr. Rohan Silva", "gender": "M"}},
  ...
]

Generate exactly {n} doctors:"""

        try:
            response = self.llm.invoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            
            json_start = content.find('[')
            json_end = content.rfind(']') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = content[json_start:json_end]
                doctors_data = json.loads(json_str)
            else:
                raise ValueError("No JSON found")
            
            doctors = []
            for i, doc_data in enumerate(doctors_data[:n]):
                doctors.append({
                    "full_name": doc_data["full_name"],
                    "gender": doc_data.get("gender", "M"),
                    "specialty": specialties[i % len(specialties)],
                })
            
            self._cache[cache_key] = doctors
            self.logger.info(f"✓ Generated {len(doctors)} doctor names")
            return doctors
            
        except Exception as e:
            self.logger.error(f"LLM generation failed: {e}")
            self.logger.warning("Falling back to template mode...")
            return TemplateDataGenerator().generate_doctors(n, specialties)
    
    @observe(name="seed_generate_patients", as_type="generation")
    def generate_patients(self, n: int) -> List[Dict]:
        """Generate patient data with Sri Lankan names."""
        cache_key = f"patients_{n}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        self.logger.info(f"🤖 Generating {n} Sri Lankan patient names via LLM...")
        
        prompt = f"""Generate {n} realistic Sri Lankan patient names for a hospital database.

Requirements:
- Mix of Sinhala and Tamil names
- Mix of male and female names
- Include children, adults, and elderly
- Culturally appropriate names

Output as JSON array:
[
  {{"full_name": "Anushka Perera", "gender": "F", "dob": "1985-03-15"}},
  ...
]

Generate exactly {n} patients:"""

        try:
            response = self.llm.invoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            
            json_start = content.find('[')
            json_end = content.rfind(']') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = content[json_start:json_end]
                patients = json.loads(json_str)
            else:
                raise ValueError("No JSON found")
            
            self._cache[cache_key] = patients[:n]
            self.logger.info(f"✓ Generated {len(patients[:n])} patient names")
            return patients[:n]
            
        except Exception as e:
            self.logger.error(f"LLM generation failed: {e}")
            return TemplateDataGenerator().generate_patients(n)
    
    @observe(name="seed_generate_reasons", as_type="generation")
    def generate_appointment_reasons(self, n: int, specialty: str) -> List[str]:
        """Generate appointment reasons for a specialty."""
        cache_key = f"reasons_{specialty}_{n}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        prompt = f"""Generate {n} realistic medical appointment reasons for {specialty}.

Requirements:
- Professional medical terminology
- Mix of common and specific conditions
- Brief and clear (5-10 words)

Output as JSON array of strings:
["Chest pain and shortness of breath", ...]

Generate exactly {n} reasons:"""

        try:
            response = self.llm.invoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            
            json_start = content.find('[')
            json_end = content.rfind(']') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = content[json_start:json_end]
                reasons = json.loads(json_str)
            else:
                raise ValueError("No JSON found")
            
            self._cache[cache_key] = reasons[:n]
            return reasons[:n]
            
        except Exception as e:
            self.logger.error(f"LLM generation failed: {e}")
            return TemplateDataGenerator().generate_appointment_reasons(n, specialty)
    
    @observe(name="seed_generate_notes", as_type="generation")
    def generate_medical_notes(self, n: int) -> List[str]:
        """Generate medical notes for patients."""
        cache_key = f"notes_{n}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        prompt = f"""Generate {n} brief medical notes for patient records.

Requirements:
- Professional medical language
- Mix of conditions (diabetes, hypertension, allergies)
- 1-2 sentences each

Output as JSON array of strings.

Generate exactly {n} notes:"""

        try:
            response = self.llm.invoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            
            json_start = content.find('[')
            json_end = content.rfind(']') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = content[json_start:json_end]
                notes = json.loads(json_str)
            else:
                raise ValueError("No JSON found")
            
            self._cache[cache_key] = notes[:n]
            return notes[:n]
            
        except Exception as e:
            self.logger.error(f"LLM generation failed: {e}")
            return TemplateDataGenerator().generate_medical_notes(n)


# ============================================================================
# TEMPLATE DATA GENERATOR
# ============================================================================

class TemplateDataGenerator(DataGenerator):
    """Generate data using predefined templates (fast, free)."""
    
    def __init__(self):
        self.logger = logger
    
    def generate_doctors(self, n: int, specialties: List[str]) -> List[Dict]:
        """Generate doctor data from templates."""
        base_names = [
            ("Priya Fernando", "F"),
            ("Rohan Silva", "M"),
            ("Amara Perera", "F"),
            ("Nimal Jayasuriya", "M"),
            ("Sanduni Wijesinghe", "F"),
            ("Kasun Rajapaksa", "M"),
            ("Chamari De Silva", "F"),
            ("Dinesh Wickramasinghe", "M"),
            ("Tharini Gunawardena", "F"),
            ("Saman Bandara", "M"),
        ]
        
        doctors = []
        for i in range(n):
            name, gender = base_names[i % len(base_names)]
            # Don't append numbers for duplicate names - will cause issues
            # Just cycle through base names
            doctors.append({
                "full_name": name,
                "gender": gender,
                "specialty": specialties[i % len(specialties)],
            })
        
        self.logger.info(f"✓ Generated {len(doctors)} doctors from templates")
        return doctors
    
    def generate_patients(self, n: int) -> List[Dict]:
        """Generate patient data from templates."""
        base_names = [
            ("Anushka Perera", "F", "1985-03-15"),
            ("Kamal Jayasuriya", "M", "1972-08-22"),
            ("Nethmi Wijesinghe", "F", "2010-11-05"),
            ("Sunil Fernando", "M", "1968-12-30"),
            ("Madhavi Silva", "F", "1995-06-18"),
        ]
        
        patients = []
        for i in range(n):
            name, gender, dob = base_names[i % len(base_names)]
            # Cycle through base names without number suffixes
            patients.append({
                "full_name": name,
                "gender": gender,
                "dob": dob,
            })
        
        self.logger.info(f"✓ Generated {len(patients)} patients from templates")
        return patients
    
    def generate_appointment_reasons(self, n: int, specialty: str) -> List[str]:
        """Generate appointment reasons from templates."""
        reasons = [
            f"Routine {specialty.lower()} consultation",
            f"{specialty} follow-up visit",
            f"Annual {specialty.lower()} checkup",
            "Diagnostic consultation",
            "Treatment review",
        ]
        return [reasons[i % len(reasons)] for i in range(n)]
    
    def generate_medical_notes(self, n: int) -> List[str]:
        """Generate medical notes from templates."""
        notes = [
            "No known allergies. Regular checkups recommended.",
            "Hypertension, on medication. Monitor BP regularly.",
            "Diabetes mellitus type 2. Diet-controlled.",
            "Asthma, uses inhaler PRN.",
            "No significant medical history.",
        ]
        return [notes[i % len(notes)] for i in range(n)]


# ============================================================================
# STORAGE ADAPTERS
# ============================================================================

class DatabaseStorageAdapter(StorageAdapter):
    """Store data in SQLite database."""
    
    def __init__(self, config: CRMSeederConfig):
        self.config = config
        self.logger = logger
        
        # Direct imports to avoid triggering entire package initialization
        from infrastructure.db.sql_client import get_sql_engine
        from infrastructure.db.crm_models import (
            Location, Specialty, Doctor, Patient, Booking
        )
        
        self.engine = get_sql_engine()
        self.Session = sessionmaker(bind=self.engine)
        self.models = {
            'Location': Location,
            'Specialty': Specialty,
            'Doctor': Doctor,
            'Patient': Patient,
            'Booking': Booking,
        }
    
    def initialize(self):
        """Initialize database connection and clear existing data."""
        # First, ensure schema exists using the same engine
        from infrastructure.db.crm_init import init_crm_schema
        init_crm_schema()
        self.logger.info("✓ CRM schema ready")
        
        # Now create session
        self.session = self.Session()
        
        # Enable foreign keys (SQLite only)
        if "sqlite" in str(self.engine.url):
            self.session.execute(text("PRAGMA foreign_keys = ON"))
        
        # Clear existing CRM data (cascade delete)
        self._clear_existing_data()
        
        self.logger.info("✓ Database initialized (existing data cleared)")
    
    def _clear_existing_data(self):
        """Delete all existing CRM data before seeding."""
        # Check if tables exist first
        try:
            # Use database-agnostic table name check
            if "postgresql" in str(self.engine.url):
                # PostgreSQL: check pg_tables
                result = self.session.execute(text(
                    "SELECT tablename FROM pg_tables WHERE schemaname='public' AND tablename='locations'"
                ))
            else:
                # SQLite: check sqlite_master
                result = self.session.execute(text(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='locations'"
                ))
            
            if result.fetchone() is None:
                self.logger.info("⏭️  Tables don't exist yet, skipping data clear")
                return
        except Exception:
            self.logger.info("⏭️  First run, skipping data clear")
            return
        
        # Tables exist, clear them
        try:
            # Delete in reverse dependency order
            self.session.execute(text("DELETE FROM bookings"))
            self.session.execute(text("DELETE FROM patients"))
            self.session.execute(text("DELETE FROM doctors"))
            self.session.execute(text("DELETE FROM specialties"))
            self.session.execute(text("DELETE FROM locations"))
            self.session.commit()
            self.logger.info("✓ Cleared existing CRM data")
        except Exception as e:
            self.logger.error(f"Failed to clear existing data: {e}")
            self.session.rollback()
            raise
    
    def store_data(self, data: Dict):
        """Store data in database."""
        model_class = self.models[data['type']]
        instance = model_class(**data['data'])
        self.session.add(instance)
    
    def finalize(self):
        """Commit and close session."""
        self.session.commit()
        self.session.close()
        self.logger.info("✓ Data committed to database")


class JSONLStorageAdapter(StorageAdapter):
    """Store data in JSONL file."""
    
    def __init__(self, config: CRMSeederConfig):
        self.config = config
        self.logger = logger
        self.bookings = []
    
    def initialize(self):
        """Initialize file."""
        self.config.output_file.parent.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"✓ JSONL file: {self.config.output_file}")
    
    def store_data(self, data: Dict):
        """Store booking data."""
        if data['type'] == 'Booking':
            self.bookings.append(data['data'])
    
    def finalize(self):
        """Write to file."""
        with open(self.config.output_file, 'w') as f:
            for booking in self.bookings:
                f.write(json.dumps(booking) + '\n')
        self.logger.info(f"✓ Wrote {len(self.bookings)} bookings to {self.config.output_file}")


# ============================================================================
# UNIFIED CRM SEEDER
# ============================================================================

class UnifiedCRMSeeder:
    """Unified CRM data seeder with all capabilities."""
    
    # ── Pre-exported SQL files (deterministic seeding) ──────────
    SQL_SEED_FILES = [
        "sql/01_specialties.sql",
        "sql/02_locations.sql",
        "sql/03_doctors.sql",
        "sql/04_patients.sql",
        "sql/05_bookings.sql",
    ]
    
    # Master data (fallback when SQL files don't exist)
    LOCATIONS = [
        {"name": "Nawaloka Hospitals", "type": "HOSPITAL", "address": "23 Deshamanya H K Dharmadasa Mawatha, Colombo 00200"},
        {"name": "Nawaloka City OPD", "type": "OPD", "address": "123 Galle Road, Colombo 03"},
        {"name": "Central Lab", "type": "LAB", "address": "45 Baseline Road, Colombo 09"},
        {"name": "Heart Care Clinic", "type": "CLINIC", "address": "78 Ward Place, Colombo 07"},
    ]
    
    SPECIALTIES = [
        "Cardiology", "Neurology", "Orthopedics", "Pediatrics",
        "General Practice", "Radiology", "Pathology", "Dermatology",
        "Gynecology", "Psychiatry",
    ]
    
    APPOINTMENT_TYPES = [
        "Consultation", "Follow-up", "Lab Test", "X-Ray",
        "Checkup", "Specialist Visit", "Diagnostic Test", "Treatment Session",
    ]
    
    def __init__(self, config: CRMSeederConfig):
        self.config = config
        self.logger = logger
        
        # Resolve project root (scripts/ → parent)
        self.project_root = Path(__file__).parent.parent
        
        # Initialize components
        self.generator = self._create_generator()
        self.storage = self._create_storage()
        
        random.seed(config.rand_seed)
    
    def _create_generator(self) -> DataGenerator:
        """Create data generator based on mode."""
        if self.config.generation_mode == DataGenerationMode.LLM:
            self.logger.info("🤖 Using LLM data generator")
            return LLMDataGenerator()
        else:
            self.logger.info("📋 Using template data generator")
            return TemplateDataGenerator()
    
    def _create_storage(self) -> StorageAdapter:
        """Create storage adapter based on backend."""
        if self.config.storage_backend == StorageBackend.DATABASE:
            self.logger.info("🗄️  Using database storage")
            return DatabaseStorageAdapter(self.config)
        else:
            self.logger.info("📄 Using JSONL storage")
            return JSONLStorageAdapter(self.config)

    # ── SQL-first seeding ──────────────────────────────────────
    
    def _sql_files_exist(self) -> bool:
        """Check if all pre-exported SQL seed files are present."""
        return all(
            (self.project_root / f).exists()
            for f in self.SQL_SEED_FILES
        )
    
    def _seed_from_sql(self) -> bool:
        """
        Load CRM data from pre-exported SQL files.
        
        Returns True if successful, False otherwise.
        """
        from infrastructure.db.sql_client import get_sql_engine
        engine = get_sql_engine()
        
        self.logger.info("📂 Found pre-exported SQL seed files — loading deterministic data")
        
        try:
            with engine.connect() as conn:
                for sql_file in self.SQL_SEED_FILES:
                    path = self.project_root / sql_file
                    sql_content = path.read_text(encoding="utf-8")
                    
                    # Strip comment lines BEFORE splitting by semicolon
                    lines = [
                        line for line in sql_content.splitlines()
                        if line.strip() and not line.strip().startswith("--")
                    ]
                    cleaned_sql = "\n".join(lines)
                    statements = [
                        s.strip() for s in cleaned_sql.split(";")
                        if s.strip()
                    ]
                    
                    row_count = 0
                    for stmt in statements:
                        conn.execute(text(stmt))
                        if stmt.upper().startswith("INSERT"):
                            row_count += 1
                    
                    conn.commit()
                    table_name = path.stem.split("_", 1)[1]  # "01_specialties" → "specialties"
                    self.logger.info(f"  ✅ {table_name}: {row_count} rows loaded from {sql_file}")
            
            return True
        except Exception as exc:
            self.logger.error(f"❌ SQL seed failed: {exc}")
            self.logger.info("   Falling back to generated data...")
            return False
    
    def seed(self):
        """Main seeding workflow."""
        self.logger.info("=" * 70)
        self.logger.info("🌱 Starting CRM data seeding")
        self.logger.info("=" * 70)
        
        start_time = time.time()
        
        # ── Try SQL-first (deterministic, identical for all students) ──
        if self.config.storage_backend == StorageBackend.DATABASE and self._sql_files_exist():
            # Init schema only (no clear — SQL files handle TRUNCATE)
            from infrastructure.db.crm_init import init_crm_schema
            init_crm_schema()
            
            if self._seed_from_sql():
                elapsed = time.time() - start_time
                self.logger.info("=" * 70)
                self.logger.info(f"✅ CRM seeded from SQL files in {elapsed:.1f}s")
                self.logger.info("=" * 70)
                return
        
        # ── Fallback: generate data dynamically ────────────────
        self.logger.info("⚙️  No SQL seed files found — generating data dynamically")
        
        # Initialize storage (will init schema and clear existing data if database mode)
        self.storage.initialize()
        
        # Seed master data
        locations = self._seed_locations()
        specialties = self._seed_specialties()
        doctors = self._seed_doctors(specialties)
        patients = self._seed_patients()
        
        # Seed bookings
        bookings_count = self._seed_bookings(doctors, patients, locations, specialties)
        
        # Finalize
        self.storage.finalize()
        
        elapsed = time.time() - start_time
        
        self.logger.info("=" * 70)
        self.logger.info("✅ Seeding complete!")
        self.logger.info(f"   Time: {elapsed:.1f}s")
        self.logger.info(f"   Locations: {len(locations)}")
        self.logger.info(f"   Specialties: {len(specialties)}")
        self.logger.info(f"   Doctors: {len(doctors)}")
        self.logger.info(f"   Patients: {len(patients)}")
        self.logger.info(f"   Bookings: {bookings_count}")
        self.logger.info("=" * 70)
    
    def _seed_locations(self) -> List:
        """Seed locations."""
        locations_to_create = self.LOCATIONS[:self.config.n_locations]
        self.logger.info(f"📍 Seeding {len(locations_to_create)} locations...")
        
        locations = []
        for loc_data in locations_to_create:
            location_id = str(uuid.uuid4())
            data = {
                'type': 'Location',
                'data': {
                    'location_id': location_id,
                    'name': loc_data['name'],
                    'type': loc_data['type'],
                    'address': loc_data['address'],
                    'tz': self.config.timezone,
                    'lat': None,
                    'lng': None,
                    'active': 1,
                    'created_at': int(time.time()),
                    'updated_at': int(time.time()),
                }
            }
            self.storage.store_data(data)
            locations.append({'id': location_id, **loc_data})
        
        return locations
    
    def _seed_specialties(self) -> List:
        """Seed specialties."""
        specialties_to_create = self.SPECIALTIES[:self.config.n_specialties]
        self.logger.info(f"🏥 Seeding {len(specialties_to_create)} specialties...")
        
        specialties = []
        for specialty_name in specialties_to_create:
            specialty_id = str(uuid.uuid4())
            data = {
                'type': 'Specialty',
                'data': {
                    'specialty_id': specialty_id,
                    'name': specialty_name,
                }
            }
            self.storage.store_data(data)
            specialties.append({'id': specialty_id, 'name': specialty_name})
        
        return specialties
    
    def _seed_doctors(self, specialties: List) -> List:
        """Seed doctors."""
        self.logger.info(f"👨‍⚕️ Seeding {self.config.n_doctors} doctors...")
        
        specialty_names = [s['name'] for s in specialties]
        doctors_data = self.generator.generate_doctors(self.config.n_doctors, specialty_names)
        
        specialty_map = {s['name']: s['id'] for s in specialties}
        
        doctors = []
        for i, doc_data in enumerate(doctors_data):
            doctor_id = str(uuid.uuid4())
            
            # Add "Dr." prefix if not present
            full_name = doc_data['full_name']
            if not full_name.startswith("Dr. "):
                full_name = f"Dr. {full_name}"
            
            data = {
                'type': 'Doctor',
                'data': {
                    'doctor_id': doctor_id,
                    'full_name': full_name,
                    'specialty_id': specialty_map.get(doc_data['specialty']),
                    'license_no': f"SL-{doc_data['specialty'][:4].upper()}-{str(i+1).zfill(4)}",
                    'phone': f"+94{random.randint(700000000, 799999999)}",
                    'email': f"{doc_data['full_name'].lower().replace(' ', '.').replace('dr.', '').replace(' ', '.')}@nawaloka.lk",
                    'active': 1,
                    'created_at': int(time.time()),
                    'updated_at': int(time.time()),
                }
            }
            self.storage.store_data(data)
            doc_data['full_name'] = full_name  # Update with prefix for return value
            doctors.append({'id': doctor_id, **doc_data, 'specialty_id': specialty_map.get(doc_data['specialty'])})
        
        return doctors
    
    def _seed_patients(self) -> List:
        """Seed patients."""
        self.logger.info(f"👤 Seeding {self.config.n_patients} patients...")
        
        patients_data = self.generator.generate_patients(self.config.n_patients)
        medical_notes = self.generator.generate_medical_notes(self.config.n_patients)
        
        patients = []
        for i, patient_data in enumerate(patients_data):
            patient_id = str(uuid.uuid4())
            
            # Generate phone number
            phone = f"+94{random.randint(700000000, 799999999)}"
            # Use phone number as external_user_id (without +)
            external_user_id = phone.replace("+", "")
            
            # Generate email from name (firstname.lastname@domain)
            full_name = patient_data['full_name']
            name_parts = full_name.lower().split()
            if len(name_parts) >= 2:
                email = f"{name_parts[0]}.{name_parts[-1]}@gmail.com"
            else:
                email = f"{name_parts[0]}@gmail.com"
            
            data = {
                'type': 'Patient',
                'data': {
                    'patient_id': patient_id,
                    'external_user_id': external_user_id,
                    'full_name': patient_data['full_name'],
                    'dob': patient_data.get('dob', '1990-01-01'),
                    'gender': patient_data.get('gender', random.choice(['M', 'F'])),
                    'phone': phone,
                    'email': email,
                    'notes': medical_notes[i] if i < len(medical_notes) else "No notes",
                    'active': 1,
                    'created_at': int(time.time()),
                    'updated_at': int(time.time()),
                }
            }
            self.storage.store_data(data)
            patients.append({'id': patient_id, **patient_data, 'user_id': external_user_id})
        
        return patients
    
    def _seed_bookings(self, doctors: List, patients: List, locations: List, specialties: List) -> int:
        """Seed bookings."""
        self.logger.info(f"📅 Seeding bookings ({self.config.n_slots_per_doctor} slots per doctor)...")
        
        tz = pytz.timezone(self.config.timezone)
        start_dt = datetime.strptime(self.config.start_date, "%Y-%m-%d")
        start_dt = tz.localize(start_dt)
        
        # Get appointment reasons per specialty
        specialty_reasons = {}
        for specialty in specialties:
            reasons = self.generator.generate_appointment_reasons(20, specialty['name'])
            specialty_reasons[specialty['name']] = reasons
        
        bookings_created = 0
        work_start, work_end = self.config.work_hours
        
        # Track patient-date combinations to avoid multiple appointments same day
        patient_dates = set()
        
        for doctor_idx, doctor in enumerate(doctors):
            doctor_slots = []
            # Stagger start dates for each doctor to spread appointments over time
            # This avoids having all 500 appointments on the first few days
            current_date = start_dt + timedelta(days=doctor_idx // 5)  # Group every 5 doctors per day
            
            # Generate slots for this doctor
            while len(doctor_slots) < self.config.n_slots_per_doctor:
                if current_date.weekday() >= 5:  # Skip weekends
                    current_date += timedelta(days=1)
                    continue
                
                for hour in range(work_start, work_end):
                    if len(doctor_slots) >= self.config.n_slots_per_doctor:
                        break
                    for minute in [0, 30]:
                        if len(doctor_slots) >= self.config.n_slots_per_doctor:
                            break
                        slot_dt = current_date.replace(hour=hour, minute=minute, second=0, microsecond=0)
                        doctor_slots.append(slot_dt)
                
                current_date += timedelta(days=1)
            
            # Create bookings
            for slot_dt in doctor_slots[:self.config.n_slots_per_doctor]:
                # Pick a patient who doesn't have an appointment on this date
                slot_date = slot_dt.date()
                attempts = 0
                max_attempts = len(patients) * 2
                
                while attempts < max_attempts:
                    patient = random.choice(patients)
                    patient_date_key = (patient['id'], slot_date)
                    
                    if patient_date_key not in patient_dates:
                        patient_dates.add(patient_date_key)
                        break
                    
                    attempts += 1
                
                # If couldn't find unique patient, use the last one (edge case for small datasets)
                if attempts >= max_attempts:
                    pass  # Use the last selected patient anyway
                
                location = random.choice(locations)
                duration_minutes = random.choice([30, 60])
                end_dt = slot_dt + timedelta(minutes=duration_minutes)
                
                # Get specialty and reasons
                specialty_name = doctor.get('specialty', 'General Practice')
                reasons = specialty_reasons.get(specialty_name, ["Consultation"])
                reason = random.choice(reasons)
                
                appointment_type = random.choice(self.APPOINTMENT_TYPES)
                
                data = {
                    'type': 'Booking',
                    'data': {
                        'booking_id': str(uuid.uuid4()),
                        'patient_id': patient['id'],
                        'doctor_id': doctor['id'],
                        'location_id': location['id'],
                        'title': f"{appointment_type} with {doctor['full_name']}",
                        'reason': reason,
                        'start_at': int(slot_dt.timestamp()),
                        'end_at': int(end_dt.timestamp()),
                        'status': random.choice(['CONFIRMED', 'CONFIRMED', 'CONFIRMED', 'PENDING']),
                        'source': 'SEED',
                        'created_at': int(time.time()),
                        'updated_at': int(time.time()),
                    }
                }
                self.storage.store_data(data)
                bookings_created += 1
        
        return bookings_created


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Unified CRM Data Seeder - All capabilities in one script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # LLM + Database (default)
  python seed_crm_unified.py --n-doctors 10 --n-patients 20

  # Template + Database (fast, free)
  python seed_crm_unified.py --mode template --n-doctors 25

  # LLM + JSONL file
  python seed_crm_unified.py --storage jsonl --output data/crm.jsonl

  # Large scale
  python seed_crm_unified.py --n-doctors 100 --n-patients 500
        """
    )
    
    # Generation mode
    parser.add_argument(
        '--mode',
        choices=['llm', 'template'],
        default='llm',
        help='Data generation mode (default: llm)'
    )
    
    # Storage backend
    parser.add_argument(
        '--storage',
        choices=['database', 'jsonl'],
        default='database',
        help='Storage backend (default: database)'
    )
    
    # Scale parameters
    parser.add_argument('--n-doctors', type=int, default=10, help='Number of doctors')
    parser.add_argument('--n-patients', type=int, default=20, help='Number of patients')
    parser.add_argument('--n-specialties', type=int, default=7, help='Number of specialties')
    parser.add_argument('--n-locations', type=int, default=4, help='Number of locations')
    parser.add_argument('--n-slots-per-doctor', type=int, default=15, help='Slots per doctor')
    
    # Scheduling
    parser.add_argument('--start', default=datetime.now().strftime("%Y-%m-%d"), help='Start date')
    parser.add_argument('--tz', default='Asia/Colombo', help='Timezone')
    parser.add_argument('--no-overlap', action='store_true', help='Prevent overlaps')
    
    # Other
    parser.add_argument('--output', type=Path, help='Output file (for JSONL mode)')
    parser.add_argument('--rand-seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Configure logging
    setup_logging()
    
    # Create config
    config = CRMSeederConfig(
        generation_mode=DataGenerationMode(args.mode),
        storage_backend=StorageBackend(args.storage),
        n_doctors=args.n_doctors,
        n_patients=args.n_patients,
        n_specialties=args.n_specialties,
        n_locations=args.n_locations,
        n_slots_per_doctor=args.n_slots_per_doctor,
        start_date=args.start,
        timezone=args.tz,
        no_overlap=args.no_overlap,
        rand_seed=args.rand_seed,
        output_file=args.output,
    )
    
    # Create seeder and run
    seeder = UnifiedCRMSeeder(config)
    seeder.seed()

    # Flush LangFuse events before exit
    flush()


if __name__ == "__main__":
    main()

