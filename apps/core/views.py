from django.contrib.auth import get_user_model
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.shortcuts import redirect, render

User = get_user_model()


def root_redirect(request):
    """Redirect root URL to dashboard or login."""
    if request.user.is_authenticated:
        return redirect("dashboard")
    return redirect("login")


@login_required
def dashboard_view(request):
    """Main dashboard — customize per project."""
    context = {
        "page_title": "Dashboard",
    }

    # Example stats for admin/manager roles
    if request.user.has_role("staff"):
        context["total_users"] = User.objects.count()
        context["active_users"] = User.objects.filter(is_active=True).count()

    return render(request, "core/dashboard.html", context)


def health_check(request):
    """Health check endpoint for load balancers / monitoring."""
    return JsonResponse({"status": "ok"})
