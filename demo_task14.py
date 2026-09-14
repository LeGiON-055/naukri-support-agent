"""
demo_task14.py — MCP Tool Lookup Demo (Capstone Task 14)

Demonstrates:
1. MCP server exposing check_job_application_status via FastMCP
2. MCP client connecting via stdio transport
3. Tool discovery (list_tools)
4. 3 tool invocations: found, escalated, and not-found records
5. Transcript saved to transcripts/task14_mcp.json
"""

import os
import sys
import asyncio

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from mcp_tools.client import run_mcp_client


def main():
    print("=" * 75)
    print("CAPSTONE TASK 14 — MCP TOOL LOOKUP DEMO")
    print("=" * 75)
    print()

    transcript = asyncio.run(run_mcp_client())

    print()
    print("=" * 75)
    print("DEMO COMPLETE")
    print("=" * 75)
    print(f"Tools discovered: {transcript['tool_count']}")
    print(f"Successful calls: {transcript['successful_calls']}/{transcript['total_calls']}")
    print(f"Transcript: transcripts/task14_mcp.json")


if __name__ == "__main__":
    main()
