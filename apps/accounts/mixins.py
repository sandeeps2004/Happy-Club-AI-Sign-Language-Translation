"""
Class-based view mixins for RBAC.

Usage:
    class MyView(RoleRequiredMixin, TemplateView):
        required_role = Roles.MANAGER
        template_name = "my_template.html"

    class MyView(PermissionRequiredMixin, TemplateView):
        required_permission = "manage_users"
        template_name = "my_template.html"
"""

from django.contrib.auth.mixins import LoginRequiredMixin
from django.core.exceptions import PermissionDenied


class RoleRequiredMixin(LoginRequiredMixin):
    """Require user to have at least the specified role (hierarchical)."""

    required_role = None

    def dispatch(self, request, *args, **kwargs):
        response = super().dispatch(request, *args, **kwargs)
        if hasattr(request, "user") and request.user.is_authenticated:
            if self.required_role and not request.user.has_role(self.required_role):
                raise PermissionDenied("Insufficient role privileges.")
        return response


class ExactRoleRequiredMixin(LoginRequiredMixin):
    """Require user to have exactly the specified role."""

    required_role = None

    def dispatch(self, request, *args, **kwargs):
        response = super().dispatch(request, *args, **kwargs)
        if hasattr(request, "user") and request.user.is_authenticated:
            if self.required_role and request.user.role != self.required_role:
                raise PermissionDenied("You do not have the required role.")
        return response


class AnyRoleRequiredMixin(LoginRequiredMixin):
    """Require user to have any one of the specified roles."""

    required_roles = []

    def dispatch(self, request, *args, **kwargs):
        response = super().dispatch(request, *args, **kwargs)
        if hasattr(request, "user") and request.user.is_authenticated:
            if self.required_roles and request.user.role not in self.required_roles:
                raise PermissionDenied("You do not have a required role.")
        return response


class CustomPermissionRequiredMixin(LoginRequiredMixin):
    """Require user to have a specific permission codename."""

    required_permission = None

    def dispatch(self, request, *args, **kwargs):
        response = super().dispatch(request, *args, **kwargs)
        if hasattr(request, "user") and request.user.is_authenticated:
            if self.required_permission and not request.user.has_perm(
                f"accounts.{self.required_permission}"
            ):
                raise PermissionDenied("You do not have the required permission.")
        return response
