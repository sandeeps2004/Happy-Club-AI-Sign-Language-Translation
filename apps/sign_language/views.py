from django.contrib.auth.decorators import login_required
from django.shortcuts import render


@login_required
def interpreter_view(request):
    """Real-time sign language interpreter page."""
    return render(request, "sign_language/interpreter.html")
