"""Shared FastAPI dependencies (access process-wide singletons via app state)."""
from __future__ import annotations

from fastapi import Request

from app.config import Settings
from app.services.vector_store import RAGStore


def get_settings_dep(request: Request) -> Settings:
    return request.app.state.settings


def get_store(request: Request) -> RAGStore:
    return request.app.state.store


def get_graph(request: Request):
    return request.app.state.graph
