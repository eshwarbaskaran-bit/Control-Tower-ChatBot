"""
graph/
──────
Control Tower LangGraph agent package.

Usage:
    from graph.agent_graph import build_graph
    compiled_graph = build_graph(is_async=True)
"""

from graph.agent_graph import build_graph

__all__ = ["build_graph"]