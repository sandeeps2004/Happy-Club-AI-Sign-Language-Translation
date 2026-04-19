from django.contrib import admin
from .models import InterpretationSession, Prediction

class PredictionInline(admin.TabularInline):
    model = Prediction
    extra = 0
    readonly_fields = ("word", "confidence", "order", "created_at")

@admin.register(InterpretationSession)
class InterpretationSessionAdmin(admin.ModelAdmin):
    list_display = ("id", "user", "word_count", "sentence_preview", "created_at")
    list_filter = ("created_at",)
    search_fields = ("sentence", "user__email")
    readonly_fields = ("glosses", "duration_seconds", "word_count")
    inlines = [PredictionInline]

    def sentence_preview(self, obj):
        return obj.sentence[:80] if obj.sentence else "(empty)"
    sentence_preview.short_description = "Sentence"
