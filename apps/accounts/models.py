from django.contrib.auth.models import AbstractUser
from django.db import models

from .constants import Permissions, Roles
from .managers import UserManager


class User(AbstractUser):
    """
    Custom User model.
    - Email as login identifier (not username).
    - Role field for RBAC.
    - Avatar + profile fields ready for extension.
    """

    username = None  # Remove username field entirely
    email = models.EmailField("email address", unique=True, db_index=True)
    role = models.CharField(max_length=20, choices=Roles.CHOICES, default=Roles.USER, db_index=True)
    avatar = models.ImageField(upload_to="avatars/", blank=True, null=True)
    phone = models.CharField(max_length=20, blank=True, default="")
    bio = models.TextField(blank=True, default="")
    is_email_verified = models.BooleanField(default=False)

    USERNAME_FIELD = "email"
    REQUIRED_FIELDS = ["first_name", "last_name"]

    objects = UserManager()

    class Meta:
        verbose_name = "user"
        verbose_name_plural = "users"
        ordering = ["-date_joined"]
        permissions = Permissions.ALL

    def __str__(self):
        return self.email

    # ── Role helpers ──────────────────────────────────────────────────────

    @property
    def is_admin(self):
        return self.role == Roles.ADMIN

    @property
    def is_manager(self):
        return self.role == Roles.MANAGER

    @property
    def is_staff_role(self):
        return self.role == Roles.STAFF

    @property
    def is_user_role(self):
        return self.role == Roles.USER

    @property
    def role_priority(self):
        return Roles.PRIORITY.get(self.role, 0)

    def has_role(self, role):
        """Check if user has at least this role level."""
        required = Roles.PRIORITY.get(role, 0)
        return self.role_priority >= required

    def has_higher_role_than(self, other_user):
        return self.role_priority > other_user.role_priority

    @property
    def display_name(self):
        full = self.get_full_name()
        return full if full.strip() else self.email

    @property
    def initials(self):
        if self.first_name and self.last_name:
            return f"{self.first_name[0]}{self.last_name[0]}".upper()
        return self.email[0].upper()
