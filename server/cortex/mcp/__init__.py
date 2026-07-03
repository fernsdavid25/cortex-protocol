"""Cortex MCP surface (the product).

Kept import-light on purpose: importing ``cortex.mcp`` does NOT pull in ``fastmcp`` (that
lives in ``cortex.mcp.server``), so the engine and benchmark can be used without the MCP
dependency installed.
"""
