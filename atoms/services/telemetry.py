from loguru import logger
from logging.handlers import QueueHandler, QueueListener
import queue
import contextvars
import os

from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor, ConsoleLogExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.semconv.resource import ResourceAttributes

from dotenv import load_dotenv

load_dotenv(override=True)

call_id_context: contextvars.ContextVar[str] = contextvars.ContextVar('call_id', default='unknown')

def format_with_call_id(record):
    call_id = call_id_context.get('none')
    record["extra"]["call_id"] = call_id
    return record

def get_otel_exporter(environment: str):
    if environment == 'local':
        return ConsoleLogExporter()
    else:
        return OTLPLogExporter(endpoint=f"{os.getenv('OTEL_BASE_URL')}/v1/logs")

def instrument_app(environment: str):
    logger.remove()

    resource = Resource(
        attributes={
            ResourceAttributes.SERVICE_NAME: "pipecat",
            ResourceAttributes.DEPLOYMENT_ENVIRONMENT: environment,
        }
    )
    logger_provider = LoggerProvider(
        resource=resource
    )
    otlp_exporter = get_otel_exporter(environment)
    processor = BatchLogRecordProcessor(otlp_exporter)
    logger_provider.add_log_record_processor(processor)
    otel_handler = LoggingHandler(logger_provider=logger_provider)

    log_queue = queue.Queue()
    queue_listener = QueueListener(log_queue, otel_handler)
    queue_listener.start()

    logger.add(
        QueueHandler(log_queue),
        level="DEBUG",
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>call_id:{extra[call_id]}</cyan> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        filter=format_with_call_id
    )
