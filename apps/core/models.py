"""
Abstract base models for the project.

Usage:
    class MyModel(TimeStampedModel):
        name = models.CharField(max_length=100)

    class MyOwnedModel(OwnedModel):
        title = models.CharField(max_length=200)
"""

from django.conf import settings
from django.db import models


class TimeStampedModel(models.Model):
    """Provides created_at and updated_at fields."""

    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        abstract = True
        ordering = ["-created_at"]


class OwnedModel(TimeStampedModel):
    """
    TimeStamped model that belongs to a user.
    Includes a helper queryset method for data isolation.
    """

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="%(class)ss",
    )

    class Meta:
        abstract = True

    @classmethod
    def for_user(cls, user):
        """Return queryset filtered to this user's records only."""
        return cls.objects.filter(user=user)
