from dataclasses import dataclass
from typing import Any

import pyotp
from SmartApi import SmartConnect

from backend.app.config import Settings, get_settings


class AngelAuthenticationError(RuntimeError):
    """Raised when Angel One authentication fails."""


@dataclass(frozen=True)
class AngelSession:
    jwt_token: str
    refresh_token: str
    feed_token: str


class AngelAuthService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.client: SmartConnect | None = None

    def _validate_credentials(self) -> None:
        credentials = {
            "ANGEL_API_KEY": self.settings.angel_api_key,
            "ANGEL_CLIENT_CODE": self.settings.angel_client_code,
            "ANGEL_PIN": self.settings.angel_pin,
            "ANGEL_TOTP_SECRET": self.settings.angel_totp_secret,
        }

        missing = [
            name
            for name, value in credentials.items()
            if not value
        ]

        if missing:
            raise AngelAuthenticationError(
                "Missing Angel One configuration: "
                + ", ".join(missing)
            )

    def generate_totp(self) -> str:
        self._validate_credentials()

        try:
            code = pyotp.TOTP(
                self.settings.angel_totp_secret
            ).now()
        except Exception as exc:
            raise AngelAuthenticationError(
                "Unable to generate Angel One TOTP."
            ) from exc

        if not code.isdigit() or len(code) != 6:
            raise AngelAuthenticationError(
                "Generated Angel One TOTP is invalid."
            )

        return code

    def authenticate(self) -> AngelSession:
        self._validate_credentials()

        try:
            client = SmartConnect(
                api_key=self.settings.angel_api_key
            )

            response: dict[str, Any] = client.generateSession(
                self.settings.angel_client_code,
                self.settings.angel_pin,
                self.generate_totp(),
            )
        except Exception as exc:
            raise AngelAuthenticationError(
                "Could not connect to Angel One SmartAPI."
            ) from exc

        if not response.get("status"):
            message = response.get(
                "message",
                "Angel One rejected the authentication request.",
            )

            error_code = response.get("errorcode")

            if error_code:
                message = f"{message} ({error_code})"

            raise AngelAuthenticationError(message)

        session_data = response.get("data") or {}

        jwt_token = session_data.get("jwtToken")
        refresh_token = session_data.get("refreshToken")

        try:
            feed_token = client.getfeedToken()
        except Exception as exc:
            raise AngelAuthenticationError(
                "Authentication succeeded, but the feed token "
                "could not be generated."
            ) from exc

        if not jwt_token or not refresh_token or not feed_token:
            raise AngelAuthenticationError(
                "Angel One returned an incomplete session."
            )

        self.client = client

        return AngelSession(
            jwt_token=jwt_token,
            refresh_token=refresh_token,
            feed_token=feed_token,
        )