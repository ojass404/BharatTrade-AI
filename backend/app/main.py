from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.router import api_router
from backend.app.config import get_settings

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    print(f"Starting {settings.app_name}")
    yield
    print(f"Stopping {settings.app_name}")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "AI-based Indian stock-market prediction, backtesting "
        "and paper-trading API."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(
    api_router,
    prefix=settings.api_prefix,
)


@app.get("/", tags=["Application"])
def root() -> dict[str, str]:
    return {
        "application": settings.app_name,
        "version": settings.app_version,
        "status": "running",
        "documentation": "/docs",
    }