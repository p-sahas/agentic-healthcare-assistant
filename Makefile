.PHONY: help install clean status \
        init-supabase test-supabase \
        seed-crm-large seed-crm-xl seed-crm-no-llm seed-procedures \
        query-crm \
        ingest-qdrant ingest-qdrant-recreate qdrant-info \
        mem-dev test-all \
        demo notebooks


VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

# ============================================================================
#  HELP - Show all available commands
# ============================================================================

help:
	@echo "╔════════════════════════════════════════════════════════════════╗"
	@echo "║        Agentic Memory Design — Makefile Commands               ║"
	@echo "╚════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo " SETUP & INSTALLATION"
	@echo "  make install              Install dependencies"
	@echo "  make init-supabase        Initialize Supabase schema (Memory + CRM)"
	@echo "  make test-supabase        Test Supabase connection & pgvector"
	@echo ""
	@echo " QDRANT (Internal KB — CAG + Parent-Child Chunking)"
	@echo "  make ingest-qdrant        Ingest internal KB → Qdrant (parent-child)"
	@echo "  make ingest-qdrant-recreate  Drop + recreate collection + re-ingest"
	@echo "  make qdrant-info          Show Qdrant collection stats"
	@echo ""
	@echo " CRM DATA GENERATION (LLM-Powered)"
	@echo "  make seed-crm-large       Small dataset: 10 doctors, 20 patients (~30s)"
	@echo "  make seed-crm-xl          Large dataset: 50 doctors, 200 patients (~2min)"
	@echo "  make seed-crm-no-llm      Template mode (free, instant, no API)"
	@echo "  make seed-procedures      Seed procedural memory workflows"
	@echo ""
	@echo " CRM QUERIES & STATUS"
	@echo "  make query-crm            Show CRM table counts (Supabase)"
	@echo "  make status               Show all system status"
	@echo ""
	@echo " TESTING"
	@echo "  make test-all             Run all tests"
	@echo "  make mem-dev              Test memory components"
	@echo ""
	@echo " DEMOS"
	@echo "  make demo                 Run full workflow demo"
	@echo "  make notebooks            Start Jupyter notebooks"
	@echo ""
	@echo " CLEANUP"
	@echo "  make clean                Remove local generated data"
	@echo ""

# ============================================================================
#  SETUP & INSTALLATION
# ============================================================================

install:
	@echo " Installing dependencies with uv..."
	uv venv $(VENV)
	uv pip install -r requirements.txt
	@echo " Installation complete!"
	@echo " Run 'source .venv/bin/activate' to activate the environ
# ============================================================================
#  SUPABASE SETUP (Production Database)
# ============================================================================

init-supabase:
	@echo "╔════════════════════════════════════════════════════════════════╗"
	@echo "║           Initializing Supabase Schema                         ║"
	@echo "╚════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo " This will create:"
	@echo "    mem_facts (Long-term semantic memory + pgvector)"
	@echo "    mem_episodes (Long-term episodic memory + pgvector)"
	@echo "    locations, specialties, doctors"
	@echo "    patients, bookings"
	@echo "    pgvector indexes (IVFFlat)"
	@echo "    Helper functions for semantic search"
	@echo "    Row Level Security (RLS) policies"
	@echo ""
	@echo " Initializing schema..."
	@PYTHONPATH=src .venv/bin/python scripts/init_supabase.py
	@echo ""
	@echo " Supabase schema initialized successfully!"

test-supabase:
	@echo " Testing Supabase connection..."
	@echo ""
	@PYTHONPATH=src .venv/bin/python scripts/test_supabase.py

# ============================================================================
#  QDRANT (RAG Knowledge Base)
# ============================================================================

ingest-qdrant:
	@echo "╔════════════════════════════════════════════════════════════════╗"
	@echo "║      Ingesting Internal KB → Qdrant (Parent-Child Chunking)    ║"
	@echo "╚════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "  Configuration:"
	@echo "   - Source: data/knowledge_base/ (internal hospital docs)"
	@echo "   - Strategy: parent_child (children indexed, parents for context)"
	@echo "   - RAG type: CAG (Cache-Augmented Generation)"
	@echo ""
	@PYTHONPATH=src python scripts/ingest_to_qdrant.py --source kb --strategy parent_child

ingest-qdrant-recreate:
	@echo "╔════════════════════════════════════════════════════════════════╗"
	@echo "║       Recreating Qdrant Collection + Re-ingesting              ║"
	@echo "╚════════════════════════════════════════════════════════════════╝"
	@PYTHONPATH=src python scripts/ingest_to_qdrant.py --source kb --strategy parent_child --recreate

qdrant-info:
	@echo " Qdrant Collection Info:"
	@PYTHONPATH=src python -c "\
from infrastructure.db.qdrant_client import collection_info; \
info = collection_info(); \
[print(f'  {k}: {v}') for k, v in info.items()]"

# ============================================================================
#  CRM DATA GENERATION (LLM-Powered)
# ============================================================================

seed-crm-large:
	@echo "╔════════════════════════════════════════════════════════════════╗"
	@echo "║      Seeding CRM — Supabase (Small)                            ║"
	@echo "╚════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "  Config: LLM mode | 10 doctors | 20 patients | ~30s | ~$$0.01"
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
	@echo "║      Seeding CRM — Supabase (Large)                            ║"
	@echo "╚════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "  Config: LLM mode | 50 doctors | 200 patients | ~2min | ~$$0.05"
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
	@echo " Seeding CRM — Template mode (no API, instant, free)"
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
	@echo " Seeding procedural memory workflows..."
	@PYTHONPATH=src python scripts/seed_procedures.py

# ============================================================================
#  CRM QUERIES & STATUS
# ============================================================================

query-crm:
	@echo "╔════════════════════════════════════════════════════════════════╗"
	@echo "║               CRM Database Statistics (Supabase)               ║"
	@echo "╚════════════════════════════════════════════════════════════════╝"
	@echo ""
	@PYTHONPATH=src python -c "\
from infrastructure.db import get_session; \
from sqlalchemy import text; \
tables = ['locations', 'specialties', 'doctors', 'patients', 'bookings']; \
session = get_session(); \
print(' Table Counts:'); \
[print(f'  {t:15s} {session.execute(text(f\"SELECT COUNT(*) FROM {t}\")).scalar():>6}') for t in tables]; \
session.close(); \
print(); \
print(' Data is in Supabase PostgreSQL')"

status:
	@echo "╔════════════════════════════════════════════════════════════════╗"
	@echo "║                      System Status                             ║"
	@echo "╚════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo " Data Sources:"
	@echo "   CRM + Memory: Supabase PostgreSQL (cloud)"
	@echo "   RAG KB:        Qdrant Cloud"
	@echo "   CAG Cache:     Qdrant Cloud (cag_cache collection)"
	@echo ""
	@echo " Local Data:"
	@if [ -d data/knowledge_base ]; then \
		echo "   Knowledge Base: data/knowledge_base/ ($$(ls data/knowledge_base/ | wc -l | tr -d ' ') docs)"; \
	else \
		echo "   Knowledge Base: Not found"; \
	fi
	@echo ""
	@echo " Configuration:"
	@echo "  Python: $$(python --version 2>&1)"
	@echo "  ST Memory: Supabase (st_turns table)"
	@echo ""
	@echo " Quick Start:"
	@echo "   make init-supabase && make seed-crm-large && make ingest-qdrant"

# ============================================================================
#  TESTING
# ============================================================================

test-all:
	@echo " Running all tests..."
	pytest tests/ -v

mem-dev:
	@echo " Testing memory components..."
	pytest tests/test_memory_*.py -q


# ============================================================================
#  NOTEBOOKS
# ============================================================================

notebooks:
	@echo " Starting Jupyter notebooks..."
	@echo "   Navigate to: http://localhost:8888"
	@jupyter notebook notebooks/

# ============================================================================
#  CLEANUP
# ============================================================================

clean:
	@echo " Cleaning local generated data..."
	rm -rf /tmp/reminders.log
	rm -rf __pycache__ src/**/__pycache__
	@echo " Cleaned!"
	@echo ""
	@echo "  Cloud data (Supabase, Qdrant) is not affected."
	@echo "   To reset CRM: re-run 'make seed-crm-large'"
	@echo "   To reset Qdrant: run 'make ingest-qdrant-recreate'"

# ============================================================================
#  DEFAULT TARGET
# ============================================================================

.DEFAULT_GOAL := help
