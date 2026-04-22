"""
MCP client configuration for the 5-server setup.

Defines how to launch each MCP server as a stdio subprocess. This dict is
consumed by `langchain_mcp_adapters.client.MultiServerMCPClient` inside
`agents/orchestrator.py::build_agent_mcp()`.

Servers:
  1. nawaloka-crm      - custom Python server, wraps CRMTool (5 tools)
  2. nawaloka-memory   - custom Python server, wraps 4-tier memory (6 tools)
  3. nawaloka-kb       - custom Python server, wraps RAGTool (3 tools)
  4. nawaloka-web      - custom Python server, wraps WebSearchTool (1 tool)
  5. postgres          - off-the-shelf @modelcontextprotocol/server-postgres,
                         pointed at Supabase (zero custom code)

Requires:
  - `mcp` and `langchain-mcp-adapters` installed via requirements.txt
  - Node.js + npx available on PATH (for the postgres server)
  - SUPABASE_POSTGRES_URL in .env (or falls back to DATABASE_URL)
"""

import os
import sys

# Absolute path to src/ so the subprocess launches regardless of cwd
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PYTHON = sys.executable  # current interpreter (venv-aware)


def build_mcp_server_config() -> dict:
    """
    Returns a dict suitable for MultiServerMCPClient.

    All 4 custom servers are always included.
    The postgres server is only included if a connection string is
    present in the environment, so the demo still runs without it.
    """
    config = {
        "nawaloka-crm": {
            "command": _PYTHON,
            "args": ["-m", "mcp_servers.crm_server"],
            "transport": "stdio",
            "cwd": _SRC_DIR,
        },
        "nawaloka-memory": {
            "command": _PYTHON,
            "args": ["-m", "mcp_servers.memory_server"],
            "transport": "stdio",
            "cwd": _SRC_DIR,
        },
        "nawaloka-kb": {
            "command": _PYTHON,
            "args": ["-m", "mcp_servers.rag_server"],
            "transport": "stdio",
            "cwd": _SRC_DIR,
        },
        "nawaloka-web": {
            "command": _PYTHON,
            "args": ["-m", "mcp_servers.web_server"],
            "transport": "stdio",
            "cwd": _SRC_DIR,
        },
    }

    pg_url = os.getenv("SUPABASE_POSTGRES_URL") or os.getenv("DATABASE_URL")
    if pg_url:
        config["postgres"] = {
            "command": "npx",
            "args": [
                "-y",
                "@modelcontextprotocol/server-postgres",
                pg_url,
            ],
            "transport": "stdio",
        }

    return config
