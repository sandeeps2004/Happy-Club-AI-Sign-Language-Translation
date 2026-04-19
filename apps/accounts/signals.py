import logging

from django.contrib.auth import get_user_model
from django.db.models.signals import post_save
from django.dispatch import receiver

from .constants import ROLE_PERMISSIONS, Roles

logger = logging.getLogger("apps.accounts")
User = get_user_model()


@receiver(post_save, sender=User)
def sync_role_permissions(sender, instance, created, **kwargs):
    """
    Auto-sync Django permissions when a user's role changes.
    Ensures the permission set matches ROLE_PERMISSIONS.
    """
    from django.contrib.auth.models import Permission

    role_perms = ROLE_PERMISSIONS.get(instance.role, [])
    permissions = Permission.objects.filter(codename__in=role_perms)

    current = set(instance.user_permissions.values_list("codename", flat=True))
    desired = set(role_perms)

    if current != desired:
        instance.user_permissions.set(permissions)
        logger.debug("Synced permissions for %s (role=%s)", instance.email, instance.role)

    # Admin role gets is_staff for Django admin access
    if instance.role == Roles.ADMIN and not instance.is_staff:
        User.objects.filter(pk=instance.pk).update(is_staff=True)
