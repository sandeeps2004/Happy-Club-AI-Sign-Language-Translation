"""
Management command to seed default roles and permissions.

Usage:
    python manage.py seed_roles
"""

from django.contrib.auth.models import Permission
from django.core.management.base import BaseCommand

from apps.accounts.constants import ROLE_PERMISSIONS, Permissions


class Command(BaseCommand):
    help = "Seed default permissions into the database."

    def handle(self, *args, **options):
        self.stdout.write("Checking permissions...")

        # Verify all custom permissions exist
        for codename, name in Permissions.ALL:
            perm = Permission.objects.filter(codename=codename).first()
            if perm:
                self.stdout.write(f"  [OK] {codename}: {name}")
            else:
                self.stdout.write(self.style.WARNING(f"  [MISSING] {codename} — run migrations first"))

        self.stdout.write("")
        self.stdout.write("Role -> Permission mapping:")
        for role, perms in ROLE_PERMISSIONS.items():
            self.stdout.write(f"  {role}: {', '.join(perms)}")

        self.stdout.write("")
        self.stdout.write(self.style.SUCCESS("Done. Permissions are auto-synced via signal on user save."))
