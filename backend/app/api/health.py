from datetime import datetime, timezone

from fastapi import APIRouter

from backend.app.config import get_settings


router = APIRouter(prefix="/health", tags=["Health"])

settings = get_settings()


@router.get("")
def health_check() -> dict[str, str]:
    """Confirm that the backend API is running."""

    return {
        "status": "healthy",
        "application": settings.app_name,
        "version": settings.app_version,
        "environment": settings.app_env,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }