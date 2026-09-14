"""
server.py — MCP Server (Capstone Task 14)

Exposes the check_job_application_status tool through FastMCP
so that any MCP-compatible client can discover and use it.

Reuses the existing agent.tools.check_job_application_status function.
"""

import os
import sys

# Ensure project root is on sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from fastmcp import FastMCP
from agent.tools import check_job_application_status

# ---------------------------------------------------------------------------
# MCP Server Definition
# ---------------------------------------------------------------------------

mcp = FastMCP("Naukri Support Agent MCP Server")


@mcp.tool()
def lookup_application_status(record_id: str) -> dict:
    """
    Look up a job application by record_id and return status with escalation scoring.

    Args:
        record_id: The unique application identifier (e.g. 'APP-001').

    Returns:
        Structured result with status, escalation score, and human-readable message.
    """
    return check_job_application_status(record_id)


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Naukri Support Agent MCP Server")
    parser.add_argument("--transport", choices=["http", "stdio", "sse"], default="http", help="Transport protocol (default: http)")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8001, help="Port to bind (default: 8001)")
    parser.add_argument("--path", default="/mcp", help="Endpoint path for http transport (default: /mcp)")
    args = parser.parse_args()

    print("Starting Naukri Support Agent MCP Server...")
    print("Available tools: lookup_application_status")
    print(f"Transport: {args.transport} | Host: {args.host} | Port: {args.port} | Path: {args.path}")

    if args.transport == "http":
        mcp.run(transport="http", host=args.host, port=args.port, path=args.path)
    elif args.transport == "sse":
        mcp.run(transport="sse", host=args.host, port=args.port)
    else:
        mcp.run(transport="stdio")

