"""
client.py — MCP Client (Capstone Task 14)

Connects to the MCP server and calls the lookup_application_status tool
with at least 2 different record IDs, including a found and not-found case.

Uses FastMCP's Client to connect via stdio transport to the server.
"""

import os
import sys
import json
import asyncio
from datetime import datetime

# Ensure project root is on sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import httpx
import uvicorn
from fastmcp import Client
from mcp_tools.server import mcp


# ---------------------------------------------------------------------------
# Test Record IDs
# ---------------------------------------------------------------------------

TEST_RECORDS = [
    {"record_id": "APP-001", "description": "Valid record (should be found)"},
    {"record_id": "APP-010", "description": "Valid record with priority flag"},
    {"record_id": "APP-999", "description": "Invalid record (should not be found)"},
]

TRANSCRIPT_PATH = os.path.join(ROOT_DIR, "transcripts", "task14_mcp.json")
DEFAULT_MCP_URL = "http://127.0.0.1:8001/mcp"


# ---------------------------------------------------------------------------
# MCP Client Runner
# ---------------------------------------------------------------------------

async def run_mcp_client(server_url: str = DEFAULT_MCP_URL, use_http: bool = True):
    """Connect to MCP server via HTTP (/mcp) and call lookup_application_status for test records."""
    print("=" * 75)
    print("TASK 14 — MCP CLIENT CALLING SERVER (HTTP TRANSPORT /mcp)")
    print("=" * 75)

    results = []
    tools_discovered = []
    server = None
    server_task = None

    if use_http:
        # Check if HTTP server is already running on the port
        is_running = False
        try:
            async with httpx.AsyncClient(timeout=0.4) as hclient:
                await hclient.get("http://127.0.0.1:8001")
                is_running = True
        except Exception:
            is_running = False

        if not is_running:
            print("  Starting MCP HTTP server at 127.0.0.1:8001...")
            starlette_app = mcp.http_app(path="/mcp")
            config = uvicorn.Config(starlette_app, host="127.0.0.1", port=8001, log_level="error")
            server = uvicorn.Server(config)
            server_task = asyncio.create_task(server.serve())
            await asyncio.sleep(0.5)

        client_target = server_url
        transport_name = "http"
        print(f"  Connected to MCP server via HTTP: {server_url}")
    else:
        server_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "server.py")
        import pathlib
        client_target = pathlib.Path(server_script)
        transport_name = "stdio"
        print("  Connected to MCP server via stdio")

    try:
        async with Client(client_target) as client:
            # Step 1: Discover available tools
            print("\n--- Tool Discovery ---")
            tools = await client.list_tools()
            for tool in tools:
                tool_info = {
                    "name": tool.name,
                    "description": tool.description or "",
                }
                tools_discovered.append(tool_info)
                print(f"  Tool: {tool.name}")
                print(f"  Description: {tool.description}")
            print(f"  Total tools discovered: {len(tools)}")

            # Step 2: Call the tool for each test record
            print("\n--- Tool Invocations ---")
            for test_case in TEST_RECORDS:
                rid = test_case["record_id"]
                desc = test_case["description"]
                print(f"\n  Calling lookup_application_status(record_id='{rid}')")
                print(f"  Description: {desc}")

                try:
                    result = await client.call_tool(
                        "lookup_application_status",
                        {"record_id": rid}
                    )
                    # Extract text content from CallToolResult
                    result_text = ""
                    if hasattr(result, 'structured_content') and result.structured_content:
                        if isinstance(result.structured_content, dict):
                            parsed = result.structured_content
                            result_text = json.dumps(parsed)
                    elif hasattr(result, 'content'):
                        for item in result.content:
                            if hasattr(item, 'text'):
                                result_text = item.text
                                break
                    elif isinstance(result, list):
                        for item in result:
                            if hasattr(item, 'text'):
                                result_text = item.text
                                break
                    else:
                        result_text = str(result)

                    # Parse the JSON result if not already parsed
                    if 'parsed' not in locals() or parsed is None:
                        try:
                            parsed = json.loads(result_text)
                        except (json.JSONDecodeError, ValueError):
                            try:
                                parsed = json.loads(result_text.replace("'", '"'))
                            except (json.JSONDecodeError, ValueError):
                                parsed = {"raw_response": result_text}

                    record_result = {
                        "record_id": rid,
                        "test_description": desc,
                        "success": True,
                        "result": parsed,
                    }
                    results.append(record_result)

                    found = parsed.get("found", "unknown")
                    status = parsed.get("status", "N/A")
                    escalated = parsed.get("escalated", "N/A")
                    message = parsed.get("message", result_text[:120])

                    print(f"  Found: {found}")
                    print(f"  Status: {status}")
                    print(f"  Escalated: {escalated}")
                    print(f"  Message: {message[:100]}...")

                except Exception as e:
                    record_result = {
                        "record_id": rid,
                        "test_description": desc,
                        "success": False,
                        "error": str(e),
                    }
                    results.append(record_result)
                    print(f"  ERROR: {e}")

    finally:
        if server:
            server.should_exit = True
            if server_task:
                await server_task

    # Build transcript
    transcript = {
        "task": "Task 14 — MCP Tool Lookup",
        "timestamp": datetime.now().isoformat(),
        "transport": "HTTP (/mcp)",
        "server_url": server_url if use_http else "stdio",
        "tools_discovered": tools_discovered,
        "tool_count": len(tools_discovered),
        "test_invocations": results,
        "total_calls": len(results),
        "successful_calls": sum(1 for r in results if r["success"]),
    }

    os.makedirs(os.path.dirname(TRANSCRIPT_PATH), exist_ok=True)
    with open(TRANSCRIPT_PATH, "w", encoding="utf-8") as f:
        json.dump(transcript, f, indent=2, ensure_ascii=False)

    print(f"\n\nTranscript saved to: {TRANSCRIPT_PATH}")
    print("=" * 75)

    return transcript


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    transcript = asyncio.run(run_mcp_client())
    print(f"\nResults: {transcript['successful_calls']}/{transcript['total_calls']} successful")
