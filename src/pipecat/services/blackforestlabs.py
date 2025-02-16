#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import io
import os
from typing import AsyncGenerator, Dict, Optional, Union

import aiohttp
from enum import Enum

from loguru import logger
from PIL import Image
from pydantic import BaseModel

from pipecat.frames.frames import ErrorFrame, Frame, URLImageRawFrame
from pipecat.services.ai_services import ImageGenService

import time
import requests

try:
    os.environ["BLACK_FOREST_LABS_KEY"]
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use Black Forest Labs API, you need to `pip install pipecat-ai[blackforestlabs]`. Also, set `BLACK_FOREST_LABS_KEY` environment variable."
    )
    raise Exception(f"Missing module: {e}")

class Status(Enum):
    PENDING = "Pending"
    READY = "Ready"
    ERROR = "Error"

class BFLImageGenService(ImageGenService):
    class InputParams(BaseModel):
        width: int = 1024
        height: int = 768
        steps: int = 40
        prompt_upsampling: bool = False
        seed: Optional[int] = None
        guidance: float = 2.5
        safety_tolerance: float = 2
        interval: int = 2
        output_format: str = "jpeg"
        image_prompt: str = ""

    def __init__(
        self,
        *,
        params: InputParams,
        aiohttp_session: aiohttp.ClientSession,
        model: str = "flux-pro",
        key: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.set_model_name(model)
        self._params = params
        self._aiohttp_session = aiohttp_session
        self.base_url = "api.us1.bfl.ai/v1"
        if key:
            os.environ["BLACK_FOREST_LABS_KEY"] = key

    def get_result(self, task_id, polling_url, attempt=1, max_attempts=15) -> Optional[str]:
       """
        Poll for task result and return sample URL if available.
        
        Args:
            task_id: Unique identifier for the task
            polling_url: URL to poll for results 
            attempt: Current attempt number
            max_attempts: Maximum number of polling attempts
            
        Returns:
            Optional[str]: Sample URL if available, None otherwise
        """
       if attempt > max_attempts:
           print(f"[FLUX API] Max attempts reached for task_id {task_id}")
           return None

       try:
           headers = {"x-key": os.environ["X_KEY"]}
           
           wait_time = min(2 ** attempt + 5, 30)
           print(f"[FLUX API] Waiting {wait_time} seconds before attempt {attempt}")
           time.sleep(wait_time)
           
           print(f"[FLUX API] Attempt {attempt}: Checking result for task {task_id}")
           response = requests.get(polling_url, headers=headers, timeout=30)
           print(f"[FLUX API] Response Status: {response.status_code}")
           
           if response.status_code == 200:
               result = response.json()
               status = result.get("status")
               print(f"[FLUX API] Task Status: {status}")
               
               if status == Status.READY.value:
                   sample_url = result.get('result', {}).get('sample')
                   if not sample_url:
                       print("[FLUX API] Error: No sample URL in response")
                       print(f"[FLUX API] Response data: {result}")
                       return None
                       
                       
               elif status == Status.PENDING.value:
                   print(f"[FLUX API] Attempt {attempt}: Image not ready. Retrying...")
                   return self.get_result(task_id, polling_url, attempt + 1)
               else:
                   print(f"[FLUX API] Unexpected status: {status}")
                   print(f"[FLUX API] Full response: {result}")
                   return None
                   
           else:
               print(f"[FLUX API] Error retrieving result: {response.status_code}")
               print(f"[FLUX API] Response: {response.text}")
               if attempt < max_attempts:
                   return self.get_result(task_id, polling_url, attempt + 1)
               
       except Exception as e:
           print(f"[FLUX API] Error retrieving result: {str(e)}")
           print(f"[FLUX API] Error Type: {type(e).__name__}")
           if attempt < max_attempts:
               return self.get_result(task_id, polling_url, attempt + 1)
               
       return None

    async def run_image_gen(self, prompt: str) -> AsyncGenerator[Frame, None]:
        def load_image_bytes(encoded_image: bytes):
            buffer = io.BytesIO(encoded_image)
            image = Image.open(buffer)
            return image.tobytes(), image.size, image.format

        logger.debug(f"Generating image from prompt: {prompt}")
        

        headers = {
            "Content-Type": "application/json",
            "X-Key": self.key
        }

        payload = {
            "prompt": prompt,
            **self._params.model_dump(exclude_none=True)
        }
        generation_url = f"https://{self.base_url}/{self.model_name}"
        
        async with self._aiohttp_session.post(generation_url, json=payload, headers=headers) as response:
            if response.status != 200:
                error_text = await response.text()
                logger.error(f"{self} error: {error_text}")
                yield ErrorFrame(f"Image generation failed: {error_text}")
                return
        initial_response = await response.json()
        polling_url = initial_response.get("polling_url")
        task_id = initial_response.get("id")

        if not polling_url:
            logger.error(f"{self} error: No polling URL received")
            yield ErrorFrame("Image generation failed: No polling URL received")
            return
        
        # Poll for results
        image_url = self.get_result(task_id=task_id, polling_url=polling_url)

        if not image_url:
            logger.error(f"{self} error: image generation failed")
            yield ErrorFrame("Image generation failed")
            return

        logger.debug(f"Image generated at: {image_url}")

        # Load the image from the url
        logger.debug(f"Downloading image {image_url} ...")
        async with self._aiohttp_session.get(image_url) as response:
            logger.debug(f"Downloaded image {image_url}")
            encoded_image = await response.content.read()
            (image_bytes, size, format) = await asyncio.to_thread(load_image_bytes, encoded_image)

            frame = URLImageRawFrame(url=image_url, image=image_bytes, size=size, format=format)
            yield frame
