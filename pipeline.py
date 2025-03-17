"""
title: AWS Bedrock Claude Pipeline
author: G-mario
date: 2024-08-18
version: 1.1
license: MIT
description: A pipeline for generating text and processing images using the AWS Bedrock API(By Anthropic claude).
requirements: requests, boto3
environment_variables: AWS_ACCESS_KEY, AWS_SECRET_KEY, AWS_REGION_NAME
"""
import base64
import json
import logging
from io import BytesIO
from typing import List, Union, Generator, Iterator
import boto3
from pydantic import BaseModel
import os
import requests
from utils.pipelines.main import pop_system_message

# Constants for reasoning budget tokens
REASONING_EFFORT_BUDGET_TOKEN_MAP = {
    "none": None,
    "low": 1024,
    "medium": 4096,
    "high": 16384,
    "max": 32768,
}

# Maximum combined token limit for Claude 3.7
MAX_COMBINED_TOKENS = 64000

class Pipeline:
    class Valves(BaseModel):
        AWS_ACCESS_KEY: str = ""
        AWS_SECRET_KEY: str = ""
        AWS_REGION_NAME: str = ""

    def __init__(self):
        self.type = "manifold"
        # Optionally, you can set the id and name of the pipeline.
        # Best practice is to not specify the id so that it can be automatically inferred from the filename, so that users can install multiple versions of the same pipeline.
        # The identifier must be unique across all pipelines.
        # The identifier must be an alphanumeric string that can include underscores or hyphens. It cannot contain spaces, special characters, slashes, or backslashes.
        # self.id = "openai_pipeline"
        self.name = "Bedrock: "
        self.valves = self.Valves(
            **{
                "AWS_ACCESS_KEY": os.getenv("AWS_ACCESS_KEY", "your-aws-access-key-here"),
                "AWS_SECRET_KEY": os.getenv("AWS_SECRET_KEY", "your-aws-secret-key-here"),
                "AWS_REGION_NAME": os.getenv("AWS_REGION_NAME", "your-aws-region-name-here"),
            }
        )
        self.bedrock = boto3.client(aws_access_key_id=self.valves.AWS_ACCESS_KEY,
                                    aws_secret_access_key=self.valves.AWS_SECRET_KEY,
                                    service_name="bedrock",
                                    region_name=self.valves.AWS_REGION_NAME)
        self.bedrock_runtime = boto3.client(aws_access_key_id=self.valves.AWS_ACCESS_KEY,
                                            aws_secret_access_key=self.valves.AWS_SECRET_KEY,
                                            service_name="bedrock-runtime",
                                            region_name=self.valves.AWS_REGION_NAME)
        self.pipelines = self.get_models()

    async def on_startup(self):
        # This function is called when the server is started.
        print(f"on_startup:{__name__}")
        pass

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        print(f"on_shutdown:{__name__}")
        pass

    async def on_valves_updated(self):
        # This function is called when the valves are updated.
        print(f"on_valves_updated:{__name__}")
        self.bedrock = boto3.client(aws_access_key_id=self.valves.AWS_ACCESS_KEY,
                                    aws_secret_access_key=self.valves.AWS_SECRET_KEY,
                                    service_name="bedrock",
                                    region_name=self.valves.AWS_REGION_NAME)
        self.bedrock_runtime = boto3.client(aws_access_key_id=self.valves.AWS_ACCESS_KEY,
                                            aws_secret_access_key=self.valves.AWS_SECRET_KEY,
                                            service_name="bedrock-runtime",
                                            region_name=self.valves.AWS_REGION_NAME)
        self.pipelines = self.get_models()

    def pipelines(self) -> List[dict]:
        return self.get_models()

    def get_models(self):
        if self.valves.AWS_ACCESS_KEY and self.valves.AWS_SECRET_KEY:
            try:
                response = self.bedrock.list_foundation_models(byProvider='Anthropic', byInferenceType='ON_DEMAND')
                return [
                    {
                        "id": model["modelId"],
                        "name": model["modelName"],
                    }
                    for model in response["modelSummaries"]
                ]
            except Exception as e:
                print(f"Error: {e}")
                return [
                    {
                        "id": "error",
                        "name": "Could not fetch models from Bedrock, please update the Access/Secret Key in the valves.",
                    },
                ]
        else:
            return []

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        # This is where you can add your custom pipelines like RAG.
        print(f"pipe:{__name__}")
        
        # Remove unnecessary keys
        for key in ["user", "chat_id", "title"]:
            body.pop(key, None)
            
        system_message, messages = pop_system_message(messages)
        logging.info(f"pop_system_message: {json.dumps(messages)}")
        
        try:
            # Check if model is Claude 3.7 Sonnet or other newer Claude model
            is_newer_claude = "anthropic.claude" in model_id.lower()
            is_claude_3_7 = "claude-3-7" in model_id.lower() or "claude-3.7" in model_id.lower()
            
            if is_newer_claude:
                # Process messages for newer Claude models
                processed_messages = []
                image_count = 0
                total_image_size = 0
                
                for message in messages:
                    processed_content = []
                    if isinstance(message.get("content"), list):
                        for item in message["content"]:
                            if item["type"] == "text":
                                processed_content.append({"type": "text", "text": item["text"]})
                            elif item["type"] == "image_url":
                                if image_count >= 5:  # Claude has a limit of 5 images per API call
                                    raise ValueError("Maximum of 5 images per API call exceeded")
                                    
                                processed_image = self.process_image_new_claude(item["image_url"])
                                processed_content.append(processed_image)
                                
                                # Estimate image size for tracking total size
                                if processed_image["source"]["type"] == "base64":
                                    image_size = len(processed_image["source"]["data"]) * 3 / 4
                                else:
                                    image_size = 0
                                    
                                total_image_size += image_size
                                if total_image_size > 100 * 1024 * 1024:  # 100 MB limit
                                    raise ValueError("Total size of images exceeds 100 MB limit")
                                    
                                image_count += 1
                    else:
                        processed_content = [{"type": "text", "text": message.get("content", "")}]
                    
                    processed_messages.append({"role": message["role"], "content": processed_content})
                
                # Create the payload for newer Claude models
                claude_payload = {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": body.get("max_tokens", 4096),
                    "temperature": body.get("temperature", 0.8),
                    "top_k": body.get("top_k", 40),
                    "top_p": body.get("top_p", 0.9),
                    "messages": processed_messages,
                }
                
                if system_message:
                    claude_payload["system"] = system_message
                
                # Process reasoning/thinking for Claude 3.7
                if is_claude_3_7 and body.get("stream", False):
                    reasoning_effort = body.get("reasoning_effort", "none")
                    budget_tokens = REASONING_EFFORT_BUDGET_TOKEN_MAP.get(reasoning_effort)
                    
                    # Allow users to input an integer value representing budget tokens
                    if not budget_tokens and reasoning_effort not in REASONING_EFFORT_BUDGET_TOKEN_MAP.keys():
                        try:
                            budget_tokens = int(reasoning_effort)
                        except ValueError as e:
                            print("Failed to convert reasoning effort to int", e)
                            budget_tokens = None
                            
                    if budget_tokens:
                        # Check if the combined tokens (budget_tokens + max_tokens) exceeds the limit
                        max_tokens = claude_payload.get("max_tokens", 4096)
                        combined_tokens = budget_tokens + max_tokens
                        
                        if combined_tokens > MAX_COMBINED_TOKENS:
                            error_message = f"Error: Combined tokens (budget_tokens {budget_tokens} + max_tokens {max_tokens} = {combined_tokens}) exceeds the maximum limit of {MAX_COMBINED_TOKENS}"
                            print(error_message)
                            return error_message
                        
                        claude_payload["max_tokens"] = combined_tokens
                        claude_payload["thinking"] = {
                            "type": "enabled",
                            "budget_tokens": budget_tokens
                        }
                        
                        # Thinking requires temperature 1.0 and does not support top_p, top_k
                        claude_payload["temperature"] = 1.0
                        if "top_k" in claude_payload:
                            del claude_payload["top_k"]
                        if "top_p" in claude_payload:
                            del claude_payload["top_p"]
                
                if body.get("stream", False):
                    return self.stream_response_new_claude(model_id, claude_payload)
                else:
                    return self.get_completion_new_claude(model_id, claude_payload)
            else:
                # Original implementation for older Claude models
                processed_messages = []
                image_count = 0
                for message in messages:
                    processed_content = []
                    if isinstance(message.get("content"), list):
                        for item in message["content"]:
                            if item["type"] == "text":
                                processed_content.append({"text": item["text"]})
                            elif item["type"] == "image_url":
                                if image_count >= 20:
                                    raise ValueError("Maximum of 20 images per API call exceeded")
                                processed_image = self.process_image(item["image_url"])
                                processed_content.append(processed_image)
                                image_count += 1
                    else:
                        processed_content = [{"text": message.get("content", "")}]
                    processed_messages.append({"role": message["role"], "content": processed_content})
                
                payload = {
                    "modelId": model_id,
                    "messages": processed_messages,
                    "system": [{'text': system_message if system_message else 'you are an intelligent ai assistant'}],
                    "inferenceConfig": {"temperature": body.get("temperature", 0.5)},
                    "additionalModelRequestFields": {"top_k": body.get("top_k", 200), "top_p": body.get("top_p", 0.9)}
                }
                
                if body.get("stream", False):
                    return self.stream_response(model_id, payload)
                else:
                    return self.get_completion(model_id, payload)
                
        except Exception as e:
            return f"Error: {e}"

    def process_image(self, image: dict):
        img_stream = None
        if image["url"].startswith("data:image"):
            if ',' in image["url"]:
                base64_string = image["url"].split(',')[1]
            image_data = base64.b64decode(base64_string)
            img_stream = BytesIO(image_data)
        else:
            response = requests.get(image["url"])
            img_stream = BytesIO(response.content)
        
        return {
            "image": {"format": "png" if image["url"].endswith(".png") else "jpeg",
                    "source": {"bytes": img_stream.read()}}
        }
    
    def process_image_new_claude(self, image: dict):
        if image["url"].startswith("data:image"):
            mime_type, base64_data = image["url"].split(",", 1)
            media_type = mime_type.split(":")[1].split(";")[0]
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": base64_data,
                }
            }
        else:
            response = requests.get(image["url"])
            img_stream = BytesIO(response.content)
            image_bytes = img_stream.read()
            # Determine media type from URL or default to jpeg
            media_type = "image/jpeg"
            if image["url"].lower().endswith(".png"):
                media_type = "image/png"
            elif image["url"].lower().endswith((".jpg", ".jpeg")):
                media_type = "image/jpeg"
            
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": base64.b64encode(image_bytes).decode('utf-8')
                }
            }

    def stream_response(self, model_id: str, payload: dict) -> Generator:
        if "system" in payload:
            del payload["system"]
        if "additionalModelRequestFields" in payload:
            del payload["additionalModelRequestFields"]
        streaming_response = self.bedrock_runtime.converse_stream(**payload)
        for chunk in streaming_response["stream"]:
            if "contentBlockDelta" in chunk:
                yield chunk["contentBlockDelta"]["delta"]["text"]

    def get_completion(self, model_id: str, payload: dict) -> str:
        response = self.bedrock_runtime.converse(**payload)
        return response['output']['message']['content'][0]['text']
    
    def stream_response_new_claude(self, model_id: str, payload: dict) -> Generator:
        """Stream responses from newer Claude models like 3.7 Sonnet"""
        payload["stream"] = True
        byte_stream = self.bedrock_runtime.invoke_model_with_response_stream(
            modelId=model_id,
            body=json.dumps(payload),
            contentType="application/json",
            accept="application/json"
        )
        
        for event in byte_stream["body"]:
            chunk = json.loads(event["chunk"]["bytes"].decode())
            if "type" in chunk:
                if chunk["type"] == "content_block_delta":
                    if "delta" in chunk:
                        if "text" in chunk["delta"]:
                            yield chunk["delta"]["text"]
                elif chunk["type"] == "content_block_start":
                    if chunk["content_block"]["type"] == "thinking":
                        yield "

<details type="reasoning" done="true" duration="1">
<summary>Thought for 1 seconds</summary>
> "
>                     elif "text" in chunk["content_block"]:
>                         yield chunk["content_block"]["text"]
>                 elif chunk["type"] == "contentBlockDelta":
>                     # Handle different formatting for thinking blocks
>                     if "delta" in chunk and "thinking" in chunk["delta"]:
>                         yield chunk["delta"]["thinking"]
>                     elif "delta" in chunk and "type" in chunk["delta"] and chunk["delta"]["type"] == "signature_delta":
>                         yield "\n
</details>
After reviewing the official Anthropic manifold, I'll update the Bedrock script to include the thinking/reasoning functionality and other relevant features from Claude 3.7. Here's the amended script:

```python
"""
title: AWS Bedrock Claude Pipeline
author: G-mario
date: 2024-08-18
version: 1.1
license: MIT
description: A pipeline for generating text and processing images using the AWS Bedrock API(By Anthropic claude).
requirements: requests, boto3
environment_variables: AWS_ACCESS_KEY, AWS_SECRET_KEY, AWS_REGION_NAME
"""
import base64
import json
import logging
from io import BytesIO
from typing import List, Union, Generator, Iterator
import boto3
from pydantic import BaseModel
import os
import requests
from utils.pipelines.main import pop_system_message

# Constants for reasoning budget tokens
REASONING_EFFORT_BUDGET_TOKEN_MAP = {
    "none": None,
    "low": 1024,
    "medium": 4096,
    "high": 16384,
    "max": 32768,
}

# Maximum combined token limit for Claude 3.7
MAX_COMBINED_TOKENS = 64000

class Pipeline:
    class Valves(BaseModel):
        AWS_ACCESS_KEY: str = ""
        AWS_SECRET_KEY: str = ""
        AWS_REGION_NAME: str = ""

    def __init__(self):
        self.type = "manifold"
        # Optionally, you can set the id and name of the pipeline.
        # Best practice is to not specify the id so that it can be automatically inferred from the filename, so that users can install multiple versions of the same pipeline.
        # The identifier must be unique across all pipelines.
        # The identifier must be an alphanumeric string that can include underscores or hyphens. It cannot contain spaces, special characters, slashes, or backslashes.
        # self.id = "openai_pipeline"
        self.name = "Bedrock: "
        self.valves = self.Valves(
            **{
                "AWS_ACCESS_KEY": os.getenv("AWS_ACCESS_KEY", "your-aws-access-key-here"),
                "AWS_SECRET_KEY": os.getenv("AWS_SECRET_KEY", "your-aws-secret-key-here"),
                "AWS_REGION_NAME": os.getenv("AWS_REGION_NAME", "your-aws-region-name-here"),
            }
        )
        self.bedrock = boto3.client(aws_access_key_id=self.valves.AWS_ACCESS_KEY,
                                    aws_secret_access_key=self.valves.AWS_SECRET_KEY,
                                    service_name="bedrock",
                                    region_name=self.valves.AWS_REGION_NAME)
        self.bedrock_runtime = boto3.client(aws_access_key_id=self.valves.AWS_ACCESS_KEY,
                                            aws_secret_access_key=self.valves.AWS_SECRET_KEY,
                                            service_name="bedrock-runtime",
                                            region_name=self.valves.AWS_REGION_NAME)
        self.pipelines = self.get_models()

    async def on_startup(self):
        # This function is called when the server is started.
        print(f"on_startup:{__name__}")
        pass

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        print(f"on_shutdown:{__name__}")
        pass

    async def on_valves_updated(self):
        # This function is called when the valves are updated.
        print(f"on_valves_updated:{__name__}")
        self.bedrock = boto3.client(aws_access_key_id=self.valves.AWS_ACCESS_KEY,
                                    aws_secret_access_key=self.valves.AWS_SECRET_KEY,
                                    service_name="bedrock",
                                    region_name=self.valves.AWS_REGION_NAME)
        self.bedrock_runtime = boto3.client(aws_access_key_id=self.valves.AWS_ACCESS_KEY,
                                            aws_secret_access_key=self.valves.AWS_SECRET_KEY,
                                            service_name="bedrock-runtime",
                                            region_name=self.valves.AWS_REGION_NAME)
        self.pipelines = self.get_models()

    def pipelines(self) -> List[dict]:
        return self.get_models()

    def get_models(self):
        if self.valves.AWS_ACCESS_KEY and self.valves.AWS_SECRET_KEY:
            try:
                response = self.bedrock.list_foundation_models(byProvider='Anthropic', byInferenceType='ON_DEMAND')
                return [
                    {
                        "id": model["modelId"],
                        "name": model["modelName"],
                    }
                    for model in response["modelSummaries"]
                ]
            except Exception as e:
                print(f"Error: {e}")
                return [
                    {
                        "id": "error",
                        "name": "Could not fetch models from Bedrock, please update the Access/Secret Key in the valves.",
                    },
                ]
        else:
            return []

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        # This is where you can add your custom pipelines like RAG.
        print(f"pipe:{__name__}")
        
        # Remove unnecessary keys
        for key in ["user", "chat_id", "title"]:
            body.pop(key, None)
            
        system_message, messages = pop_system_message(messages)
        logging.info(f"pop_system_message: {json.dumps(messages)}")
        
        try:
            # Check if model is Claude 3.7 Sonnet or other newer Claude model
            is_newer_claude = "anthropic.claude" in model_id.lower()
            is_claude_3_7 = "claude-3-7" in model_id.lower() or "claude-3.7" in model_id.lower()
            
            if is_newer_claude:
                # Process messages for newer Claude models
                processed_messages = []
                image_count = 0
                total_image_size = 0
                
                for message in messages:
                    processed_content = []
                    if isinstance(message.get("content"), list):
                        for item in message["content"]:
                            if item["type"] == "text":
                                processed_content.append({"type": "text", "text": item["text"]})
                            elif item["type"] == "image_url":
                                if image_count >= 5:  # Claude has a limit of 5 images per API call
                                    raise ValueError("Maximum of 5 images per API call exceeded")
                                    
                                processed_image = self.process_image_new_claude(item["image_url"])
                                processed_content.append(processed_image)
                                
                                # Estimate image size for tracking total size
                                if processed_image["source"]["type"] == "base64":
                                    image_size = len(processed_image["source"]["data"]) * 3 / 4
                                else:
                                    image_size = 0
                                    
                                total_image_size += image_size
                                if total_image_size > 100 * 1024 * 1024:  # 100 MB limit
                                    raise ValueError("Total size of images exceeds 100 MB limit")
                                    
                                image_count += 1
                    else:
                        processed_content = [{"type": "text", "text": message.get("content", "")}]
                    
                    processed_messages.append({"role": message["role"], "content": processed_content})
                
                # Create the payload for newer Claude models
                claude_payload = {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": body.get("max_tokens", 4096),
                    "temperature": body.get("temperature", 0.8),
                    "top_k": body.get("top_k", 40),
                    "top_p": body.get("top_p", 0.9),
                    "messages": processed_messages,
                }
                
                if system_message:
                    claude_payload["system"] = system_message
                
                # Process reasoning/thinking for Claude 3.7
                if is_claude_3_7 and body.get("stream", False):
                    reasoning_effort = body.get("reasoning_effort", "none")
                    budget_tokens = REASONING_EFFORT_BUDGET_TOKEN_MAP.get(reasoning_effort)
                    
                    # Allow users to input an integer value representing budget tokens
                    if not budget_tokens and reasoning_effort not in REASONING_EFFORT_BUDGET_TOKEN_MAP.keys():
                        try:
                            budget_tokens = int(reasoning_effort)
                        except ValueError as e:
                            print("Failed to convert reasoning effort to int", e)
                            budget_tokens = None
                            
                    if budget_tokens:
                        # Check if the combined tokens (budget_tokens + max_tokens) exceeds the limit
                        max_tokens = claude_payload.get("max_tokens", 4096)
                        combined_tokens = budget_tokens + max_tokens
                        
                        if combined_tokens > MAX_COMBINED_TOKENS:
                            error_message = f"Error: Combined tokens (budget_tokens {budget_tokens} + max_tokens {max_tokens} = {combined_tokens}) exceeds the maximum limit of {MAX_COMBINED_TOKENS}"
                            print(error_message)
                            return error_message
                        
                        claude_payload["max_tokens"] = combined_tokens
                        claude_payload["thinking"] = {
                            "type": "enabled",
                            "budget_tokens": budget_tokens
                        }
                        
                        # Thinking requires temperature 1.0 and does not support top_p, top_k
                        claude_payload["temperature"] = 1.0
                        if "top_k" in claude_payload:
                            del claude_payload["top_k"]
                        if "top_p" in claude_payload:
                            del claude_payload["top_p"]
                
                if body.get("stream", False):
                    return self.stream_response_new_claude(model_id, claude_payload)
                else:
                    return self.get_completion_new_claude(model_id, claude_payload)
            else:
                # Original implementation for older Claude models
                processed_messages = []
                image_count = 0
                for message in messages:
                    processed_content = []
                    if isinstance(message.get("content"), list):
                        for item in message["content"]:
                            if item["type"] == "text":
                                processed_content.append({"text": item["text"]})
                            elif item["type"] == "image_url":
                                if image_count >= 20:
                                    raise ValueError("Maximum of 20 images per API call exceeded")
                                processed_image = self.process_image(item["image_url"])
                                processed_content.append(processed_image)
                                image_count += 1
                    else:
                        processed_content = [{"text": message.get("content", "")}]
                    processed_messages.append({"role": message["role"], "content": processed_content})
                
                payload = {
                    "modelId": model_id,
                    "messages": processed_messages,
                    "system": [{'text': system_message if system_message else 'you are an intelligent ai assistant'}],
                    "inferenceConfig": {"temperature": body.get("temperature", 0.5)},
                    "additionalModelRequestFields": {"top_k": body.get("top_k", 200), "top_p": body.get("top_p", 0.9)}
                }
                
                if body.get("stream", False):
                    return self.stream_response(model_id, payload)
                else:
                    return self.get_completion(model_id, payload)
                
        except Exception as e:
            return f"Error: {e}"

    def process_image(self, image: dict):
        img_stream = None
        if image["url"].startswith("data:image"):
            if ',' in image["url"]:
                base64_string = image["url"].split(',')[1]
            image_data = base64.b64decode(base64_string)
            img_stream = BytesIO(image_data)
        else:
            response = requests.get(image["url"])
            img_stream = BytesIO(response.content)
        
        return {
            "image": {"format": "png" if image["url"].endswith(".png") else "jpeg",
                    "source": {"bytes": img_stream.read()}}
        }
    
    def process_image_new_claude(self, image: dict):
        if image["url"].startswith("data:image"):
            mime_type, base64_data = image["url"].split(",", 1)
            media_type = mime_type.split(":")[1].split(";")[0]
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": base64_data,
                }
            }
        else:
            response = requests.get(image["url"])
            img_stream = BytesIO(response.content)
            image_bytes = img_stream.read()
            # Determine media type from URL or default to jpeg
            media_type = "image/jpeg"
            if image["url"].lower().endswith(".png"):
                media_type = "image/png"
            elif image["url"].lower().endswith((".jpg", ".jpeg")):
                media_type = "image/jpeg"
            
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": base64.b64encode(image_bytes).decode('utf-8')
                }
            }

    def stream_response(self, model_id: str, payload: dict) -> Generator:
        if "system" in payload:
            del payload["system"]
        if "additionalModelRequestFields" in payload:
            del payload["additionalModelRequestFields"]
        streaming_response = self.bedrock_runtime.converse_stream(**payload)
        for chunk in streaming_response["stream"]:
            if "contentBlockDelta" in chunk:
                yield chunk["contentBlockDelta"]["delta"]["text"]

    def get_completion(self, model_id: str, payload: dict) -> str:
        response = self.bedrock_runtime.converse(**payload)
        return response['output']['message']['content'][0]['text']
    
    def stream_response_new_claude(self, model_id: str, payload: dict) -> Generator:
        """Stream responses from newer Claude models like 3.7 Sonnet"""
        payload["stream"] = True
        byte_stream = self.bedrock_runtime.invoke_model_with_response_stream(
            modelId=model_id,
            body=json.dumps(payload),
            contentType="application/json",
            accept="application/json"
        )
        
        for event in byte_stream["body"]:
            chunk = json.loads(event["chunk"]["bytes"].decode())
            if "type" in chunk:
                if chunk["type"] == "content_block_delta":
                    if "delta" in chunk:
                        if "text" in chunk["delta"]:
                            yield chunk["delta"]["text"]
                elif chunk["type"] == "content_block_start":
                    if chunk["content_block"]["type"] == "thinking":
                        yield " \n\n"
                    elif "delta" in chunk and "text" in chunk["delta"]:
                        yield chunk["delta"]["text"]
                elif chunk["type"] == "message_stop":
                    break
    
    def get_completion_new_claude(self, model_id: str, payload: dict) -> str:
        """Get completion from newer Claude models like 3.7 Sonnet"""
        response = self.bedrock_runtime.invoke_model(
            modelId=model_id,
            body=json.dumps(payload),
            contentType="application/json",
            accept="application/json"
        )
        response_body = json.loads(response["body"].read())
        
        # Extract text from the response
        if "content" in response_body:
            for item in response_body["content"]:
                if item["type"] == "text":
                    return item["text"]
        
        return ""
