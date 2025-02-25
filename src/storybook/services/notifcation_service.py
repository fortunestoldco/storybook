"""
Notification service module using Twilio.
"""

from twilio.rest import Client
from storybook.config import TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM_NUMBER, TWILIO_TO_NUMBER

class NotificationService:
    def __init__(self):
        self.client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

    def send_message(self, body: str, to: str = TWILIO_TO_NUMBER):
        message = self.client.messages.create(
            body=body,
            from_=TWILIO_FROM_NUMBER,
            to=to
        )
        return message.sid
