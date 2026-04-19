"""
Role-based access control decorators.

Usage:
    @role_required(Roles.ADMIN)
    def admin_view(request): ...

    @any_role_required(Roles.ADMIN, Roles.MANAGER)
    def manager_view(request): ...

    @permission_required_custom("manage_users")
    def user_management(request): ...
"""

from functools import wraps

from django.contrib.auth.decorators import login_required
from django.http import HttpResponseForbidden
from django.shortcuts import redirect

from .constants import Roles


def role_required(role):
    """
    Require user to have at least this role level (hierarchical).
    Admin > Manager > Staff > User.
    """

    def decorator(view_func):
        @wraps(view_func)
        @login_required
        def wrapper(request, *args, **kwargs):
            if request.user.has_role(role):
                return view_func(request, *args, **kwargs)
            return HttpResponseForbidden(
                "You do not have permission to access this page."
            )

        return wrapper

    return decorator


def exact_role_required(role):
    """Require user to have exactly this role (no hierarchy)."""

    def decorator(view_func):
        @wraps(view_func)
        @login_required
        def wrapper(request, *args, **kwargs):
            if request.user.role == role:
                return view_func(request, *args, **kwargs)
            return HttpResponseForbidden(
                "You do not have permission to access this page."
            )

        return wrapper

    return decorator


def any_role_required(*roles):
    """Require user to have any one of the specified roles."""

    def decorator(view_func):
        @wraps(view_func)
        @login_required
        def wrapper(request, *args, **kwargs):
            if request.user.role in roles:
                return view_func(request, *args, **kwargs)
            return HttpResponseForbidden(
                "You do not have permission to access this page."
            )

        return wrapper

    return decorator


def permission_required_custom(perm_codename):
    """Require user to have a specific permission codename."""

    def decorator(view_func):
        @wraps(view_func)
        @login_required
        def wrapper(request, *args, **kwargs):
            if request.user.has_perm(f"accounts.{perm_codename}"):
                return view_func(request, *args, **kwargs)
            return HttpResponseForbidden(
                "You do not have the required permission."
            )

        return wrapper

    return decorator


# ── Convenience shortcuts ────────────────────────────────────────────────────
admin_required = role_required(Roles.ADMIN)
manager_required = role_required(Roles.MANAGER)
staff_required = role_required(Roles.STAFF)
