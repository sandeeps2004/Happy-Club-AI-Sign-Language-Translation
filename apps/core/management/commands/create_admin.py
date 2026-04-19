"""
Management command to create an admin user.

Usage:
    python manage.py create_admin --email admin@example.com --password admin123
"""

from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand

from apps.accounts.constants import Roles

User = get_user_model()


class Command(BaseCommand):
    help = "Create an admin superuser with the given email and password."

    def add_arguments(self, parser):
        parser.add_argument("--email", type=str, default="admin@admin.com")
        parser.add_argument("--password", type=str, default="admin123")
        parser.add_argument("--first-name", type=str, default="Admin")
        parser.add_argument("--last-name", type=str, default="User")

    def handle(self, *args, **options):
        email = options["email"]
        password = options["password"]

        if User.objects.filter(email=email).exists():
            self.stdout.write(self.style.WARNING(f"User {email} already exists. Skipping."))
            return

        User.objects.create_superuser(
            email=email,
            password=password,
            first_name=options["first_name"],
            last_name=options["last_name"],
            role=Roles.ADMIN,
        )
        self.stdout.write(self.style.SUCCESS(f"Admin user created: {email}"))
