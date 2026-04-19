"""
Role & Permission constants for the RBAC system.

Roles are hierarchical: ADMIN > MANAGER > STAFF > USER.
Each role inherits all permissions of the roles below it.

To add a new role:
  1. Add it to ROLES dict with a priority (higher = more powerful).
  2. Add its base permissions to ROLE_PERMISSIONS.
  3. Run: python manage.py seed_roles
"""


class Roles:
    ADMIN = "admin"
    MANAGER = "manager"
    STAFF = "staff"
    USER = "user"

    CHOICES = [
        (ADMIN, "Admin"),
        (MANAGER, "Manager"),
        (STAFF, "Staff"),
        (USER, "User"),
    ]

    # Higher number = more privileges. Used for hierarchy checks.
    PRIORITY = {
        ADMIN: 100,
        MANAGER: 75,
        STAFF: 50,
        USER: 10,
    }


class Permissions:
    """
    Custom permission codenames. Django auto-creates add/change/delete/view
    per model; these are extra app-level permissions.
    """

    # User management
    MANAGE_USERS = "manage_users"
    ASSIGN_ROLES = "assign_roles"
    VIEW_ALL_USERS = "view_all_users"

    # Content management
    MANAGE_CONTENT = "manage_content"
    PUBLISH_CONTENT = "publish_content"

    # System
    VIEW_DASHBOARD = "view_dashboard"
    VIEW_ADMIN_PANEL = "view_admin_panel"
    VIEW_ANALYTICS = "view_analytics"
    MANAGE_SETTINGS = "manage_settings"

    ALL = [
        (MANAGE_USERS, "Can manage users"),
        (ASSIGN_ROLES, "Can assign roles to users"),
        (VIEW_ALL_USERS, "Can view all users"),
        (MANAGE_CONTENT, "Can manage all content"),
        (PUBLISH_CONTENT, "Can publish content"),
        (VIEW_DASHBOARD, "Can view dashboard"),
        (VIEW_ADMIN_PANEL, "Can view admin panel"),
        (VIEW_ANALYTICS, "Can view analytics"),
        (MANAGE_SETTINGS, "Can manage system settings"),
    ]


# Maps each role to its base permissions.
# Roles inherit downward: admin gets everything, manager gets staff + user, etc.
ROLE_PERMISSIONS = {
    Roles.USER: [
        Permissions.VIEW_DASHBOARD,
    ],
    Roles.STAFF: [
        Permissions.VIEW_DASHBOARD,
        Permissions.MANAGE_CONTENT,
        Permissions.PUBLISH_CONTENT,
    ],
    Roles.MANAGER: [
        Permissions.VIEW_DASHBOARD,
        Permissions.MANAGE_CONTENT,
        Permissions.PUBLISH_CONTENT,
        Permissions.VIEW_ALL_USERS,
        Permissions.VIEW_ANALYTICS,
    ],
    Roles.ADMIN: [
        Permissions.MANAGE_USERS,
        Permissions.ASSIGN_ROLES,
        Permissions.VIEW_ALL_USERS,
        Permissions.MANAGE_CONTENT,
        Permissions.PUBLISH_CONTENT,
        Permissions.VIEW_DASHBOARD,
        Permissions.VIEW_ADMIN_PANEL,
        Permissions.VIEW_ANALYTICS,
        Permissions.MANAGE_SETTINGS,
    ],
}
