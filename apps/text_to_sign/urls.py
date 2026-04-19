from django.urls import path

from . import views

urlpatterns = [
    path("", views.translator_view, name="text_to_sign_translator"),
    path("translate/", views.translate_endpoint, name="text_to_sign_translate"),
]
