"""Root API router. Registers all versioned sub-routers."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.v1 import chat, demo, health, knowledge

api_router = APIRouter()

api_router.include_router(health.router)
api_router.include_router(chat.router)
api_router.include_router(knowledge.router)
api_router.include_router(demo.router)

