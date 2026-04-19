"""
Base service class for business logic.

Usage:
    class MyService(BaseService):
        def do_thing(self, data):
            self.log.info("Doing thing with %s", data)
            # ... business logic ...
            return result
"""

import logging


class BaseService:
    """Base class providing logger and common patterns."""

    def __init__(self):
        self.log = logging.getLogger(f"services.{self.__class__.__name__}")
