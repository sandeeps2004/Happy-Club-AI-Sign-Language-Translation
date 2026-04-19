"""
Email service — wraps Django's email sending.

Usage:
    email_service = EmailService()
    email_service.send("user@example.com", "Subject", "template_name", context)
"""

from django.conf import settings
from django.core.mail import send_mail
from django.template.loader import render_to_string

from .base import BaseService


class EmailService(BaseService):

    def send(self, to_email, subject, template_name=None, context=None, plain_message=""):
        """Send an email. If template_name is given, renders it as the message body."""
        if template_name and context:
            message = render_to_string(template_name, context)
        else:
            message = plain_message

        try:
            send_mail(
                subject=subject,
                message=message,
                from_email=settings.DEFAULT_FROM_EMAIL,
                recipient_list=[to_email] if isinstance(to_email, str) else to_email,
                fail_silently=False,
            )
            self.log.info("Email sent to %s: %s", to_email, subject)
            return True
        except Exception as e:
            self.log.error("Failed to send email to %s: %s", to_email, e)
            return False
