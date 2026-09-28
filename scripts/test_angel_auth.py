from backend.app.services.angel_auth_service import (
    AngelAuthenticationError,
    AngelAuthService,
)


def main() -> None:
    print("Testing Angel One SmartAPI authentication...")

    try:
        service = AngelAuthService()
        session = service.authenticate()
    except AngelAuthenticationError as exc:
        print(f"Authentication failed: {exc}")
        raise SystemExit(1) from exc

    print("Authentication successful.")
    print(f"JWT token received: {bool(session.jwt_token)}")
    print(f"Refresh token received: {bool(session.refresh_token)}")
    print(f"Feed token received: {bool(session.feed_token)}")
    print("No token values were displayed.")


if __name__ == "__main__":
    main()