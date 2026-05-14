.PHONY: help install clean status \
        init-supabase test-supabase \
        seed-crm-large seed-crm-xl seed-crm-no-llm seed-procedures \
        query-crm \
        ingest-qdrant ingest-qdrant-recreate qdrant-info \
        mem-dev test-all \
        demo demo-down demo-logs notebooks \
        voice voice-test demo-voice demo-voice-down voice-logs

# ============================================================================
# HELP - Show all available commands
# ============================================================================

help:
	@echo "╔════════════════════════════════════════════════════════════════╗"
	@echo "║        Agentic Memory Design — Makefile Commands                ║"
	@echo "╚════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "SETUP & INSTALLATION"
	@echo "  make install              Install dependencies"
	@echo "  make init-supabase        Initialize Supabase schema (Memory + CRM)"
	@echo "  make test-supabase        Test Supabase connection & pgvector"
	@echo ""
	@echo "QDRANT (Internal KB — CAG + Parent-Child Chunking)"
	@echo "  make ingest-qdrant        Ingest internal KB → Qdrant (parent-child)"
	@echo "  make ingest-qdrant-recreate  Drop + recreate collection + re-ingest"
	@echo "  make qdrant-info          Show Qdrant collection stats"
	@echo ""
	@echo "CRM DATA GENERATION (LLM-Powered)"
	@echo "  make seed-crm-large       Small dataset: 10 doctors, 20 patients (~30s)"
	@echo "  make seed-crm-xl          Large dataset: 50 doctors, 200 patients (~2min)"
	@echo "  make seed-crm-no-llm      Template mode (free, instant, no API)"
	@echo "  make seed-procedures      Seed procedural memory workflows"
	@echo ""
	@echo "CRM QUERIES & STATUS"
	@echo "  make query-crm            Show CRM table counts (Supabase)"
	@echo "  make status               Show all system status"
	@echo ""
	@echo "TESTING"
	@echo "  make test-all             Run all tests"
	@echo "  make mem-dev              Test memory components"
	@echo ""
	@echo "DOCKER DEMO"
	@echo "  make demo                 Build + start api + web (localhost:8080)"
	@echo "  make demo-logs            Tail api container logs"
	@echo "  make demo-down            Stop containers"
	@echo ""
	@echo "VOICE PIPELINE (LiveKit + Deepgram)"
	@echo "  make voice                Run voice worker natively (foreground)"
	@echo "  make voice-test           Validate voice config + env vars"
	@echo "  make demo-voice           Bring up Docker stack with voice worker"
	@echo "  make voice-logs           Tail voice container logs"
	@echo "  make demo-voice-down      Stop the voice-profile stack"
	@echo ""
	@echo "NOTEBOOKS"
	@echo "  make notebooks            Start Jupyter notebooks"
	@echo ""
	@echo "CLEANUP"
	@echo "  make clean                Remove local generated data"
	@echo ""

# ============================================================================
# SETUP & INSTALLATION
# ============================================================================

install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt
	@echo "Installation complete!"

# ============================================================================
# SUPABASE SETUP (Production Database)
# ============================================================================

init-supabase:
	@echo "╔════════════════════════════════════════════════════════════════╗"
	@echo "║        Initializing Supabase Schema                         ║"
	@echo "╚════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "This will create:"
	@echo "   mem_facts (Long-term semantic memory + pgvector)"
	@echo "   mem_episodes (Long-term episodic memory + pgvector)"
	@echo "   locations, specialties, doctors"
	@echo "   patients, bookings"
	@echo "   pgvector indexes (IVFFlat)"
	@echo "   Helper functions for semantic search"
	@echo "   Row Level Security (RLS) policies"
	@echo ""
	@echo "Initializing schema..."
	@PYTHONPATH=src .venv/bin/python scripts/init_supabase.py
	@echo ""
	@echo "Supabase schema initialized successfully!"

test-supabase:
	@echo "Testing Supabase connection..."
	@echo ""
	@PYTHONPATH=src .venv/bin/python scripts/test_supabase.py

# ============================================================================
# QDRANT (RAG Knowledge Base)
# ============================================================================

ingest-qdrant:
	@echo "╔════════════════════════════════════════════════════════════════╗"
	@echo "║  Ingesting Internal KB → Qdrant (Parent-Child Chunking)    ║"
	@echo "╚════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "Configuration:"
	@echo "   - Source: data/knowledge_base/ (internal hospital docs)"
	@echo "   - Strategy: parent_child (children indexed, parents for context)"
	@echo "   - RAG type: CAG (Cache-Augmented Generation)"
	@echo ""
	@PYTHONPATH=src python scripts/ingest_to_qdrant.py --source kb --strategy parent_child

ingest-qdrant-recreate:
	@echo "╔════════════════════════════════════════════════════════════════╗"
	@echo "║  Recreating Qdrant Collection + Re-ingesting              ║"
	@echo "╚════════════════════════════════════════════════════════════════╝"
	@PYTHONPATH=src python scripts/ingest_to_qdrant.py --source kb --strategy parent_child --recreate

qdrant-info:
	@echo "Qdrant Collection Info:"
	@PYTHONPATH=src python -c "\
from infrastructure.db.qdrant_client import collection_info; \
info = collection_info(); \
[print(f'  {k}: {v}') for k, v in info.items()]"

# ============================================================================
# CRM DATA GENERATION (LLM-Powered)
# ============================================================================

seed-crm-large:
	@echo "╔════════════════════════════════════════════════════════════════╗"
	@echo "║  Seeding CRM — Supabase (Small)                            ║"
	@echo "╚════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "Config: LLM mode | 10 doctors | 20 patients | ~30s | ~$$0.01"
	@echo ""
	@PYTHONPATH=src python scripts/seed_crm_unified.py \
		--mode llm \
		--storage database \
		--n-doctors 10 \
		--n-patients 20 \
		--n-specialties 7 \
		--n-locations 4 \
		--n-slots-per-doctor 15 \
		--start 2025-11-03 \
		--tz Asia/Colombo \
		--no-overlap \
		--rand-seed 42

seed-crm-xl:
	@echo "╔════════════════════════════════════════════════════════════════╗"
	@echo "║  Seeding CRM — Supabase (Large)                            ║"
	@echo "╚════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "Config: LLM mode | 50 doctors | 200 patients | ~2min | ~$$0.05"
	@echo ""
	@PYTHONPATH=src python scripts/seed_crm_unified.py \
		--mode llm \
		--storage database \
		--n-doctors 50 \
		--n-patients 200 \
		--n-specialties 10 \
		--n-locations 4 \
		--n-slots-per-doctor 30 \
		--start 2025-11-01 \
		--tz Asia/Colombo \
		--no-overlap \
		--rand-seed 42

seed-crm-no-llm:
	@echo "Seeding CRM — Template mode (no API, instant, free)"
	@echo ""
	@PYTHONPATH=src python scripts/seed_crm_unified.py \
		--mode template \
		--storage database \
		--n-doctors 25 \
		--n-patients 100 \
		--n-specialties 7 \
		--n-locations 4 \
		--n-slots-per-doctor 20 \
		--start 2025-11-03 \
		--tz Asia/Colombo \
		--no-overlap \
		--rand-seed 42

seed-procedures:
	@echo "Seeding procedural memory workflows..."
	@PYTHONPATH=src python scripts/seed_procedures.py

# ============================================================================
# CRM QUERIES & STATUS
# ============================================================================

query-crm:
	@echo "╔════════════════════════════════════════════════════════════════╗"
	@echo "║              CRM Database Statistics (Supabase)               ║"
	@echo "╚════════════════════════════════════════════════════════════════╝"
	@echo ""
	@PYTHONPATH=src python -c "\
from infrastructure.db import get_session; \
from sqlalchemy import text; \
tables = ['locations', 'specialties', 'doctors', 'patients', 'bookings']; \
session = get_session(); \
print('Table Counts:'); \
[print(f'  {t:15s} {session.execute(text(f\"SELECT COUNT(*) FROM {t}\")).scalar():>6}') for t in tables]; \
session.close(); \
print(); \
print('Data is in Supabase PostgreSQL')"

status:
	@echo "╔════════════════════════════════════════════════════════════════╗"
	@echo "║                      System Status                             ║"
	@echo "╚════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "Data Sources:"
	@echo "  CRM + Memory: Supabase PostgreSQL (cloud)"
	@echo "  RAG KB:        Qdrant Cloud"
	@echo "  CAG Cache:     Qdrant Cloud (cag_cache collection)"
	@echo ""
	@echo "Local Data:"
	@if [ -d data/knowledge_base ]; then \
		echo "  Knowledge Base: data/knowledge_base/ ($$(ls data/knowledge_base/ | wc -l | tr -d ' ') docs)"; \
	else \
		echo "  Knowledge Base: Not found"; \
	fi
	@echo ""
	@echo "Configuration:"
	@echo "  Python: $$(python --version 2>&1)"
	@echo "  ST Memory: Supabase (st_turns table)"
	@echo ""
	@echo "Quick Start:"
	@echo "   make init-supabase && make seed-crm-large && make ingest-qdrant"

# ============================================================================
# TESTING
# ============================================================================

test-all:
	@echo "Running all tests..."
	pytest tests/ -v

mem-dev:
	@echo "Testing memory components..."
	pytest tests/test_memory_*.py -q


# ============================================================================
# NOTEBOOKS
# ============================================================================

notebooks:
	@echo "Starting Jupyter notebooks..."
	@echo "   Navigate to: http://localhost:8888"
	@jupyter notebook notebooks/

# ============================================================================
# DOCKER DEMO — one-shot containerised stack
# ============================================================================
# Brings up the full app (FastAPI + nginx/SPA) in two containers.
# Requires: Docker Desktop running, .env populated with real secrets.

demo:
	@if [ ! -f .env ]; then \
		echo "No .env found in $$(pwd)."; \
		echo ""; \
		echo "    Create a .env file with your secrets:"; \
		echo "      SUPABASE_DB_URL  (port 6543!)"; \
		echo "      QDRANT_URL, QDRANT_API_KEY"; \
		echo "      GROQ_API_KEY, OPENROUTER_API_KEY, OPENAI_API_KEY"; \
		echo "      TAVILY_API_KEY, LANGFUSE_* (optional)"; \
		exit 1; \
	fi
	@echo "Building and starting api + web containers..."
	docker compose up --build -d
	@echo ""
	@echo "Stack is up. First boot takes ~60s for lifespan warmup."
	@echo ""
	@echo "    Web UI →  http://localhost:8080"
	@echo "    API   →  http://localhost:8000  (docs at /docs)"
	@echo ""
	@echo "    Try these to see CAG / RAG / CRM each light up:"
	@echo "      • 'What are the opening hours?'           (CAG  ~290 ms)"
	@echo "      • 'Do I have a booking next week?'        (CRM  ~3-5 s)"
	@echo "      • 'How do I claim insurance?'             (CAG)"
	@echo ""
	@echo "    Logs:  make demo-logs    Stop:  make demo-down"

demo-logs:
	docker compose logs -f api

demo-down:
	docker compose down
	@echo "Stack stopped. Use 'docker compose down -v' to also wipe the HF cache volume."

# ============================================================================
# VOICE PIPELINE (LiveKit + Deepgram)
# ============================================================================
# A standalone LiveKit worker that joins LiveKit Cloud rooms and pipes
# audio through Silero VAD → Deepgram STT → orchestrator.achat() →
# Deepgram TTS. Voice is a side-car: the api + web stack runs without it.
#
# Required env vars (in .env): LIVEKIT_URL, LIVEKIT_API_KEY,
# LIVEKIT_API_SECRET, DEEPGRAM_API_KEY (plus the SUPABASE / Qdrant /
# LLM keys that achat() needs).

voice:
	@echo "Starting voice worker (foreground)..."
	@echo "    Connect via: https://agents-playground.livekit.io"
	@echo ""
	@PYTHONPATH=src python -m voice.run start

voice-test:
	@echo "Validating voice config + env vars..."
	@PYTHONPATH=src python -c "\
	from dotenv import load_dotenv; load_dotenv(); \
	from voice.config import load_voice_config, validate_voice_env; \
	cfg = load_voice_config(); \
	print(f'STT          : {cfg.stt_provider}/{cfg.stt_model}'); \
	print(f'TTS          : {cfg.tts_provider}/{cfg.tts_model}'); \
	tts_voice = cfg.tts_voice_id if cfg.tts_provider == 'elevenlabs' else '(n/a)'; \
	print(f'TTS voice ID : {tts_voice}'); \
	print(f'VAD threshold: {cfg.vad_threshold}'); \
	print(f'Silence      : {cfg.silence_threshold_ms} ms'); \
	print(f'Endpointing  : {cfg.min_endpointing_delay} s'); \
	print(f'Interruptions: {cfg.interruption_enabled}'); \
	print(); \
	validate_voice_env(); \
	print('voice env OK')"

demo-voice:
	@if [ ! -f .env ]; then \
		echo "No .env found. Add LIVEKIT_*, DEEPGRAM_API_KEY, and the api/web vars."; \
		exit 1; \
	fi
	@echo "Building and starting api + web + voice containers..."
	docker compose --profile voice up --build -d
	@echo ""
	@echo "Voice stack is up."
	@echo ""
	@echo "    Web UI       →  http://localhost:8080"
	@echo "    API          →  http://localhost:8000"
	@echo "    Voice worker →  outbound-only (joins LiveKit Cloud rooms)"
	@echo ""
	@echo "    Connect a browser via:"
	@echo "      https://agents-playground.livekit.io"
	@echo ""
	@echo "    Logs:  make voice-logs    Stop:  make demo-voice-down"

voice-logs:
	docker compose logs -f voice

demo-voice-down:
	docker compose --profile voice down
	@echo "Voice-profile stack stopped."

# ============================================================================
# CLEANUP
# ============================================================================

clean:
	@echo "Cleaning local generated data..."
	rm -rf /tmp/reminders.log
	rm -rf __pycache__ src/**/__pycache__
	@echo "Cleaned!"
	@echo ""
	@echo "Cloud data (Supabase, Qdrant) is not affected."
	@echo "   To reset CRM: re-run 'make seed-crm-large'"
	@echo "   To reset Qdrant: run 'make ingest-qdrant-recreate'"

# ============================================================================
# DEFAULT TARGET
# ============================================================================

.DEFAULT_GOAL := help
