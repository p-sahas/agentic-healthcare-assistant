"""
MCP Servers for the Nawaloka Hospital Multi-Agent System.

Five servers expose existing codebase functionality via the Model Context
Protocol, making them consumable by any MCP host (LangGraph agent, Claude
Desktop, Cursor, MCP Inspector, etc.) without custom adapters:

  1. crm_server.py     — wraps src/agents/tools/crm_tool.py       (5 tools)
  2. memory_server.py  — wraps src/memory/memory_ops.py            (6 tools)
  3. rag_server.py     — wraps src/agents/tools/rag_tool.py        (3 tools)
  4. web_server.py     — wraps src/agents/tools/web_search_tool.py (1 tool)
  5. postgres          — off-the-shelf @modelcontextprotocol/server-postgres
                         pointed at Supabase (no code — see mcp_config.py)

Run any server standalone over stdio:
    python -m mcp_servers.crm_server
    python -m mcp_servers.memory_server
    python -m mcp_servers.rag_server
    python -m mcp_servers.web_server

Or inspect with:
    npx @modelcontextprotocol/inspector python -m mcp_servers.crm_server
"""
