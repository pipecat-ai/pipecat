#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""
Example demonstrating the Service Resilience Framework.

This example shows how to use the new resilience framework to make services
more robust with automatic retries, circuit breakers, and fallback mechanisms.

The resilience framework provides:
- Automatic retries with exponential backoff
- Circuit breaker pattern to prevent cascading failures
- Service health monitoring and metrics
- Standardized error handling across services

Run with:
    python examples/foundational/43-service-resilience.py
"""

import argparse
import asyncio
import os
import random
from typing import Any, Optional

from dotenv import load_dotenv
from loguru import logger

from pipecat.frames.frames import EndFrame, ErrorFrame, TextFrame, TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.resilience import (
    CircuitBreakerConfig,
    RetryPolicy,
    ServiceResilience,
    create_llm_resilience,
)
from pipecat.services.resilience.resilient_service import ResilientServiceMixin
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.network.fastapi_websocket import FastAPIWebsocketParams
from pipecat.transports.services.daily import DailyParams

load_dotenv(override=True)


class UnreliableService(ResilientServiceMixin, FrameProcessor):
    """Example service that simulates unreliable behavior to demonstrate resilience patterns.
    
    This service will randomly fail to show how the resilience framework handles:
    - Transient failures with automatic retries
    - Persistent failures with circuit breaker protection
    - Fallback mechanisms when primary operations fail
    """
    
    def __init__(self, failure_rate: float = 0.3, **kwargs):
        super().__init__(**kwargs)
        self.failure_rate = failure_rate
        self.call_count = 0
        
        # Setup resilience framework with custom configuration
        self._setup_resilience(
            service_name="unreliable-service",
            retry_policy=RetryPolicy(
                max_attempts=3,
                base_delay=0.5,
                max_delay=5.0,
            ),
            circuit_breaker_config=CircuitBreakerConfig(
                failure_threshold=3,
                recovery_timeout=10.0,
                success_threshold=2,
            ),
        )
        
        logger.info(f"UnreliableService initialized with {failure_rate*100}% failure rate")

    async def process_frame(self, frame, direction: FrameDirection):
        """Process frames with resilience patterns."""
        if isinstance(frame, TextFrame):
            try:
                # Execute the unreliable operation with resilience
                result = await self._execute_resilient(
                    operation=lambda: self._unreliable_text_processing(frame.text),
                    operation_name="text-processing",
                    context=f"text-length-{len(frame.text)}",
                )
                
                # Push the successful result
                await self.push_frame(TTSSpeakFrame(result))
                
            except Exception as e:
                logger.error(f"Text processing failed after all resilience attempts: {e}")
                # Push error message to TTS
                await self.push_frame(
                    TTSSpeakFrame("Sorry, I'm having technical difficulties right now.")
                )
        else:
            await super().process_frame(frame, direction)

    async def _unreliable_text_processing(self, text: str) -> str:
        """Simulate an unreliable operation that sometimes fails."""
        self.call_count += 1
        
        # Simulate processing delay
        await asyncio.sleep(0.1)
        
        # Randomly fail based on failure rate
        if random.random() < self.failure_rate:
            error_types = [
                ConnectionError("Network connection failed"),
                TimeoutError("Operation timed out"),
                RuntimeError("Service temporarily unavailable"),
            ]
            raise random.choice(error_types)
        
        # Success case
        processed_text = f"Processed (attempt {self.call_count}): {text}"
        logger.info(f"Successfully processed text: {text}")
        return processed_text


class FallbackDemoService(ResilientServiceMixin, FrameProcessor):
    """Demonstrates fallback mechanisms when primary services fail."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._setup_resilience("fallback-demo")
        
    async def process_frame(self, frame, direction: FrameDirection):
        if isinstance(frame, TextFrame) and frame.text.startswith("fallback:"):
            # Extract the actual text after "fallback:" prefix
            actual_text = frame.text[9:]  # Remove "fallback:" prefix
            
            try:
                # Try primary operation with fallback
                result = await self._execute_with_fallback(
                    primary_operation=lambda: self._primary_processing(actual_text),
                    fallback_operation=lambda: self._fallback_processing(actual_text),
                    operation_name="text-processing-with-fallback",
                )
                
                await self.push_frame(TTSSpeakFrame(result))
                
            except Exception as e:
                logger.error(f"Both primary and fallback operations failed: {e}")
                await self.push_frame(
                    TTSSpeakFrame("All processing methods are currently unavailable.")
                )
        else:
            await super().process_frame(frame, direction)

    async def _primary_processing(self, text: str) -> str:
        """Primary processing that fails 80% of the time."""
        await asyncio.sleep(0.1)
        
        if random.random() < 0.8:  # 80% failure rate
            raise ConnectionError("Primary service is down")
        
        return f"Primary processed: {text}"

    async def _fallback_processing(self, text: str) -> str:
        """Fallback processing that's more reliable."""
        await asyncio.sleep(0.05)  # Faster fallback
        
        if random.random() < 0.1:  # 10% failure rate
            raise RuntimeError("Fallback service also failed")
        
        return f"Fallback processed: {text}"


async def demonstrate_resilience_patterns():
    """Demonstrate various resilience patterns."""
    
    logger.info("\n=== Service Resilience Framework Demo ===\n")
    
    # 1. Basic retry with exponential backoff
    logger.info("1. Testing retry policy with exponential backoff...")
    
    resilience = ServiceResilience("demo-service")
    
    async def flaky_operation():
        if random.random() < 0.7:  # 70% failure rate
            raise ConnectionError("Simulated network error")
        return "Success!"
    
    try:
        result = await resilience.execute(flaky_operation, operation_name="flaky-test")
        logger.info(f"Retry demo result: {result}")
    except Exception as e:
        logger.error(f"Retry demo failed: {e}")
    
    # 2. Circuit breaker demo
    logger.info("\n2. Testing circuit breaker pattern...")
    
    circuit_resilience = ServiceResilience(
        "circuit-demo",
        circuit_breaker_config=CircuitBreakerConfig(
            failure_threshold=2,  # Open after 2 failures
            recovery_timeout=5.0,  # Try recovery after 5 seconds
            success_threshold=1,   # Close after 1 success
        )
    )
    
    async def always_failing_operation():
        raise RuntimeError("This always fails")
    
    # Trigger circuit breaker
    for i in range(5):
        try:
            await circuit_resilience.execute(
                always_failing_operation, 
                operation_name=f"circuit-test-{i}"
            )
        except Exception as e:
            logger.info(f"Circuit test {i+1}: {type(e).__name__}: {e}")
        
        await asyncio.sleep(0.5)
    
    # 3. Health check demo
    logger.info("\n3. Testing health check functionality...")
    
    async def health_check():
        # Simulate health check that sometimes fails
        if random.random() < 0.3:
            raise ConnectionError("Health check failed")
        return "Healthy"
    
    is_healthy = await resilience.health_check(health_check)
    logger.info(f"Service health status: {'Healthy' if is_healthy else 'Unhealthy'}")
    
    # 4. Show resilience statistics
    logger.info("\n4. Resilience statistics:")
    stats = resilience.get_stats()
    for key, value in stats.items():
        if isinstance(value, dict):
            logger.info(f"  {key}:")
            for sub_key, sub_value in value.items():
                logger.info(f"    {sub_key}: {sub_value}")
        else:
            logger.info(f"  {key}: {value}")


# We store functions so objects don't get instantiated. The function will be
# called when the desired transport gets selected.
transport_params = {
    "daily": lambda: DailyParams(audio_out_enabled=True),
    "twilio": lambda: FastAPIWebsocketParams(audio_out_enabled=True), 
    "webrtc": lambda: TransportParams(audio_out_enabled=True),
}


async def run_example(transport: BaseTransport, args: argparse.Namespace, handle_sigint: bool):
    """Run the service resilience example."""
    
    # First, demonstrate resilience patterns in isolation
    await demonstrate_resilience_patterns()
    
    logger.info(f"\n=== Starting resilient pipeline demo ===\n")
    
    # Create TTS service
    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
    )
    
    # Create resilient services
    unreliable_service = UnreliableService(failure_rate=0.4, name="unreliable-processor")
    fallback_service = FallbackDemoService(name="fallback-processor")
    
    # Create pipeline with resilient services
    pipeline = Pipeline([
        unreliable_service,
        fallback_service,
        tts,
        transport.output(),
    ])
    
    task = PipelineTask(pipeline)
    
    # Register event handler for when client connects
    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        """Send demo messages when client connects."""
        messages = [
            # Test basic resilience
            TextFrame("Hello! This message tests basic retry functionality."),
            
            # Test fallback mechanism  
            TextFrame("fallback:This message demonstrates fallback processing."),
            
            # Test multiple messages to trigger circuit breaker
            TextFrame("This is message 1 to test circuit breaker behavior."),
            TextFrame("This is message 2 to test circuit breaker behavior."), 
            TextFrame("This is message 3 to test circuit breaker behavior."),
            
            # Final message
            TextFrame("Demo complete! Check the logs to see resilience patterns in action."),
            
            EndFrame(),
        ]
        
        for message in messages:
            await task.queue_frame(message)
            await asyncio.sleep(2)  # Pause between messages
    
    runner = PipelineRunner(handle_sigint=handle_sigint)
    await runner.run(task)


if __name__ == "__main__":
    from pipecat.examples.run import main

    main(run_example, transport_params=transport_params) 