from django.conf import settings


def global_context(request):
    """Inject global variables into every template."""
    return {
        "APP_NAME": getattr(settings, "APP_NAME", "AI Sign Language"),
        "DEBUG": settings.DEBUG,
    }
