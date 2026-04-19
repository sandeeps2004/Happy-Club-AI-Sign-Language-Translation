"""
Custom template tags and filters.

Usage in templates:
    {% load core_tags %}
    {% if user|has_role:"admin" %}...{% endif %}
    {% if user|has_perm_custom:"manage_users" %}...{% endif %}
    {{ some_text|truncate_chars:50 }}
"""

from django import template

register = template.Library()


@register.filter
def has_role(user, role):
    """Check if user has at least the given role level."""
    if not user or not user.is_authenticated:
        return False
    return user.has_role(role)


@register.filter
def has_exact_role(user, role):
    """Check if user has exactly the given role."""
    if not user or not user.is_authenticated:
        return False
    return user.role == role


@register.filter
def has_perm_custom(user, perm):
    """Check if user has a specific permission codename."""
    if not user or not user.is_authenticated:
        return False
    return user.has_perm(f"accounts.{perm}")


@register.filter
def truncate_chars(value, max_length):
    """Truncate a string and add ellipsis."""
    if len(str(value)) <= max_length:
        return value
    return str(value)[: max_length - 1] + "\u2026"


@register.inclusion_tag("includes/sidebar_link.html", takes_context=True)
def sidebar_link(context, url, icon, label, url_name=None):
    """Render a sidebar link with active state detection."""
    request = context.get("request")
    is_active = False
    if request and url_name:
        from django.urls import resolve
        try:
            is_active = resolve(request.path).url_name == url_name
        except Exception:
            is_active = request.path.startswith(url)
    return {"url": url, "icon": icon, "label": label, "is_active": is_active}
