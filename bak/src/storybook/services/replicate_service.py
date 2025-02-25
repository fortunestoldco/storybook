"""
Replicate service module.
"""

import replicate
from storybook.config import REPLICATE_API_TOKEN


class ReplicateService:
    def __init__(self):
        self.client = replicate.Client(api_token=REPLICATE_API_TOKEN)

    def generate_image(self, prompt: str):
        model = self.client.models.get("stability-ai/stable-diffusion")
        version = model.versions.get("latest")
        return version.predict(prompt=prompt)
