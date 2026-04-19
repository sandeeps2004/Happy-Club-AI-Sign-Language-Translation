import logging
import time

logger = logging.getLogger("apps.core")


class RequestLoggingMiddleware:
    """Logs request method, path, status code, and duration."""

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        start = time.monotonic()
        response = self.get_response(request)
        duration_ms = (time.monotonic() - start) * 1000

        # Skip static/media files
        if not request.path.startswith(("/static/", "/media/")):
            user = getattr(request, "user", None)
            user_str = user.email if user and user.is_authenticated else "anonymous"
            logger.info(
                "%s %s %s [%s] %.0fms",
                request.method,
                request.path,
                response.status_code,
                user_str,
                duration_ms,
            )

        return response
