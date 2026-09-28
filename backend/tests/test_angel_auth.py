from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from backend.app.services.angel_auth_service import (
    AngelAuthenticationError,
    AngelAuthService,
)


VALID_TOTP_SECRET = "JBSWY3DPEHPK3PXP"


def create_settings(**overrides: str) -> SimpleNamespace:
    values = {
        "angel_api_key": "test-api-key",
        "angel_client_code": "TEST123",
        "angel_pin": "1234",
        "angel_totp_secret": VALID_TOTP_SECRET,
    }

    values.update(overrides)

    return SimpleNamespace(**values)


def test_generate_totp_returns_six_digits() -> None:
    service = AngelAuthService(settings=create_settings())

    code = service.generate_totp()

    assert code.isdigit()
    assert len(code) == 6


def test_authenticate_returns_session_tokens() -> None:
    mock_client = MagicMock()

    mock_client.generateSession.return_value = {
        "status": True,
        "message": "SUCCESS",
        "data": {
            "jwtToken": "test-jwt-token",
            "refreshToken": "test-refresh-token",
        },
    }

    mock_client.getfeedToken.return_value = "test-feed-token"

    with patch(
        "backend.app.services.angel_auth_service.SmartConnect",
        return_value=mock_client,
    ):
        service = AngelAuthService(settings=create_settings())
        session = service.authenticate()

    assert session.jwt_token == "test-jwt-token"
    assert session.refresh_token == "test-refresh-token"
    assert session.feed_token == "test-feed-token"

    mock_client.generateSession.assert_called_once()


def test_authenticate_rejects_missing_credentials() -> None:
    service = AngelAuthService(
        settings=create_settings(angel_api_key="")
    )

    with pytest.raises(
        AngelAuthenticationError,
        match="ANGEL_API_KEY",
    ):
        service.authenticate()


def test_authenticate_handles_api_rejection() -> None:
    mock_client = MagicMock()

    mock_client.generateSession.return_value = {
        "status": False,
        "message": "Invalid credentials",
        "errorcode": "AG8001",
        "data": None,
    }

    with patch(
        "backend.app.services.angel_auth_service.SmartConnect",
        return_value=mock_client,
    ):
        service = AngelAuthService(settings=create_settings())

        with pytest.raises(
            AngelAuthenticationError,
            match="Invalid credentials",
        ):
            service.authenticate()