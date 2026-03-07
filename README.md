# Agentic Memory Design

Agentic Memory Design is a sophisticated multi-memory AI assistant built to operate as a healthcare assistant. The system features a powerful routing engine, Retrieval-Augmented Generation (RAG), Customer Relationship Management (CRM) integration, and full observability.

## Table of Contents
- [Architecture](#architecture)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [License](#license)

## Architecture

The system employs a 3-model architecture designed for specialized tasks:
- **Routing Engine**: Utilizes GPT-4o-mini to classify and route user queries effectively.
- **Extraction & Distillation**: Uses Llama 3.1 8B (via Groq) for rapid processing and memory extraction.
- **Chat & Synthesis**: Employs Gemini 2.5 Flash for high-quality conversational responses and synthesizing contextual information.

### Core Components
- **Memory Management**: Seamlessly integrates short-term memory (Supabase) and long-term memory via vector embeddings. Includes episodic and procedural stores. 
- **Tool Routing**: Intelligent dispatching to specialized tools like CRM lookup, web search (Tavily), or RAG queries.
- **RAG & Context-Augmented Generation (CAG)**: Employs Qdrant for semantic search and caching, ensuring fast and relevant context retrieval from the internal knowledge base.
- **Observability**: Fully integrated with LangFuse for real-time tracing of prompts, tool executions, memory operations, and performance metrics.

## Key Features

- **Multi-tiered Memory**: Automatically tracks short-term conversations while extracting and distilling long-term facts in the background.
- **Adaptive Tool Dispatch**: Capable of querying internal patient records (CRM), fetching external information (Web Search), or reading organizational knowledge (RAG).
- **Extensive Tuning**: Robust set of configurations for various chunking strategies (fixed, semantic, sliding, parent-child, and late chunking).
- **Semantic Caching**: Reduces cost and latency by caching frequent responses using Qdrant.

## Project Structure

```text
agentic-healthcare-assistant/
├── config/             # YAML configuration files (models, parameters, FAQs)
├── notebooks/          # Exploratory Jupyter Notebooks for core functionality
├── scripts/            # Database initialization and ingestion scripts
├── sql/                # SQL definitions for Supabase schemas and vector stores
├── src/                # Primary source code
│   ├── agents/         # Orchestrator, router, tools, and prompts
│   ├── infrastructure/ # LLM setups, database clients, loggers, and observability
│   ├── memory/         # Short-term, long-term, and specialized memory stores
│   └── services/       # Chat, CRM, and domain-specific ingestion services
├── pyproject.toml      # Project definitions and dependencies
├── requirements.txt    # Python package dependencies
└── README.md           # Project documentation
```

## Prerequisites

- Python 3.10 or higher
- PostgreSQL instance (e.g., Supabase)
- Qdrant cluster (local or cloud)
- Appropriate API keys for LLMs (OpenRouter, Groq, Google, etc.) and Tavily.

## Installation

1. Clone the repository and navigate into the project directory:
   ```bash
   git clone <repository_url>
   cd agentic-healthcare-assistant
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   Alternatively, if you use Hatch or a similar build tool, you can install via:
   ```bash
   pip install -e .
   ```

## Configuration

1. Create a `.env` file in the root directory and configure the environment variables:
   ```env
   # API Keys
   OPENROUTER_API_KEY=your_openrouter_key
   GROQ_API_KEY=your_groq_key
   GOOGLE_API_KEY=your_google_ai_key
   TAVILY_API_KEY=your_tavily_key

   # Observability
   LANGFUSE_SECRET_KEY=your_langfuse_secret_key
   LANGFUSE_PUBLIC_KEY=your_langfuse_public_key
   LANGFUSE_HOST=https://cloud.langfuse.com

   # Qdrant Database
   QDRANT_API_KEY=your_qdrant_key
   QDRANT_URL=your_qdrant_url

   # Supabase
   SUPABASE_URL=your_supabase_url
   SUPABASE_SERVICE_KEY=your_supabase_service_key
   ```

2. Adjust the system parameters in `config/param.yaml` and model specifications in `config/models.yaml` as required. Sub-configurations such as vector chunking sizes, embedding models, and API endpoints are fully customizable within these files.

## Usage

1. **Initialize the Database**: First, set up Supabase schema by running the initialization scripts.
   ```bash
   python scripts/init_supabase.py
   ```

2. **Ingest Knowledge Base**: Load your domain-specific content into Qdrant.
   ```bash
   python scripts/ingest_to_qdrant.py
   ```

3. **Run the Orchestrator**: You can invoke the main chat service directly or integrate it via your preferred interface. For a basic start, examine the usage patterns in the `notebooks/` directory.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
