"""
whatsapp_bot package
================================================================================
Alert notification module providing multi-recipient WhatsApp dispatch
via Twilio REST API with background daemon thread pooling and local fallbacks.
================================================================================
"""

from .sender import WhatsAppDispatcher, AlertPayload, DispatchResult

__all__ = ["WhatsAppDispatcher", "AlertPayload", "DispatchResult"]
