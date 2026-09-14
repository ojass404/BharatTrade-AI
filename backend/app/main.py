from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.health import router as health_router
from backend.app.config import get_settings


settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Run application startup and shutdown logic."""

    print(
        f"Starting {settings.app_name} "
        f"v{settings.app_version} in {settings.app_env} mode"
    )

    yield

    print(f"Stopping {settings.app_name}")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "AI-based stock movement prediction, backtesting "
        "and paper-trading API for selected NSE equities."
    ),
    debug=settings.debug,
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
    health_router,
    prefix=settings.api_prefix,
)


@app.get("/", tags=["Root"])
def root() -> dict[str, str]:
    """Return basic information about the API."""

    return {
        "application": settings.app_name,
        "version": settings.app_version,
        "message": "BharatTrade AI backend is running.",
        "documentation": "/docs",
        "health_check": f"{settings.api_prefix}/health",
    }