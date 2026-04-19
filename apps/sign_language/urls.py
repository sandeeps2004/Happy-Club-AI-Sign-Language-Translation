from django.urls import path

from . import views

urlpatterns = [
    path("interpreter/", views.interpreter_view, name="sign_language_interpreter"),
]
