# test_jaeger.py
import time

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

# Set up a resource (this is required for the trace to be properly displayed in Jaeger)
resource = Resource.create({"service.name": "test-jaeger-connection"})

# Set up a tracer provider with the resource
provider = TracerProvider(resource=resource)

# Set up a console exporter
console_exporter = ConsoleSpanExporter()
provider.add_span_processor(BatchSpanProcessor(console_exporter))

# Set up the OTLP exporter to Jaeger
otlp_exporter = OTLPSpanExporter(endpoint="http://localhost:4317", insecure=True)
provider.add_span_processor(BatchSpanProcessor(otlp_exporter))

# Set the global tracer provider
trace.set_tracer_provider(provider)

# Get a tracer
tracer = trace.get_tracer("test-tracer")

print("Creating test spans...")

# Create parent span
with tracer.start_as_current_span("parent-span") as parent:
    parent.set_attribute("parent.attribute", "parent-value")
    print("Created parent span")
    time.sleep(1)  # Simulate work

    # Create child span
    with tracer.start_as_current_span("child-span") as child:
        child.set_attribute("child.attribute", "child-value")
        child.set_attribute("child.number", 42)
        print("Created child span")
        time.sleep(1)  # Simulate more work

print("Test complete. Check Jaeger UI for traces with service 'test-jaeger-connection'")
# Sleep to allow the batch processor time to export
time.sleep(5)
