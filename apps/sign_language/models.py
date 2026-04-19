"""Models for sign language interpretation sessions."""
from django.conf import settings
from django.db import models
from apps.core.models import TimeStampedModel

class InterpretationSession(TimeStampedModel):
    """A single recording session of sign language interpretation."""
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="interpretation_sessions",
    )
    sentence = models.TextField(blank=True, default="")
    glosses = models.JSONField(default=list)
    duration_seconds = models.FloatField(null=True, blank=True)
    word_count = models.PositiveIntegerField(default=0)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        preview = self.sentence[:60] or "(no sentence)"
        return f"Session {self.pk} — {preview}"

    @classmethod
    def for_user(cls, user):
        return cls.objects.filter(user=user)


class Prediction(TimeStampedModel):
    """A single word prediction within a session."""
    session = models.ForeignKey(
        InterpretationSession,
        on_delete=models.CASCADE,
        related_name="predictions",
    )
    word = models.CharField(max_length=100)
    confidence = models.FloatField()
    order = models.PositiveIntegerField(default=0)

    class Meta:
        ordering = ["order"]

    def __str__(self):
        return f"{self.word} ({self.confidence:.0%})"
