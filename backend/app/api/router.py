from fastapi import APIRouter

from backend.app.api.health import router as health_router
from backend.app.api.stocks import router as stocks_router

api_router = APIRouter()

api_router.include_router(health_router)
api_router.include_router(stocks_router)