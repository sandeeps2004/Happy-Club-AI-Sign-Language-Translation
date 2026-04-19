"""
Seed demo users for development.

Usage:
    python manage.py seed_demo
"""

from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand

from apps.accounts.constants import Roles

User = get_user_model()

DEMO_USERS = [
    {"email": "admin@demo.com", "password": "demo1234", "first_name": "Admin", "last_name": "User", "role": Roles.ADMIN, "is_staff": True, "is_superuser": True},
    {"email": "manager@demo.com", "password": "demo1234", "first_name": "Manager", "last_name": "User", "role": Roles.MANAGER},
    {"email": "staff@demo.com", "password": "demo1234", "first_name": "Staff", "last_name": "User", "role": Roles.STAFF},
    {"email": "user@demo.com", "password": "demo1234", "first_name": "Regular", "last_name": "User", "role": Roles.USER},
]


class Command(BaseCommand):
    help = "Seed demo users with different roles for development."

    def handle(self, *args, **options):
        for data in DEMO_USERS:
            email = data.pop("email")
            password = data.pop("password")

            if User.objects.filter(email=email).exists():
                self.stdout.write(f"  [SKIP] {email} already exists")
                continue

            user = User.objects.create_user(email=email, password=password, **data)
            self.stdout.write(self.style.SUCCESS(f"  [OK] Created {email} (role={user.role})"))

        self.stdout.write("")
        self.stdout.write(self.style.SUCCESS("Demo users seeded. Password for all: demo1234"))
