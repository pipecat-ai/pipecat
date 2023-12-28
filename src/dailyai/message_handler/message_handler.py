import logging
import time

from dataclasses import dataclass
from queue import Queue, Empty
from threading import Thread

from dailyai.storage.search import SearchIndexer
from dailyai.services.ai_services import AIServiceConfig


@dataclass
class Message:
    type: str
    timestamp: float
    message: str


class MessageHandler:
    def __init__(self, intro):
        self.messages: list[Message] = [Message("system", time.time(), intro)]
        self.last_user_message_idx:int | None = None
        self.finalized_user_message_idx: int | None = None

    def add_user_message(self, message) -> None:
        if self.last_user_message_idx is not None and self.last_user_message_idx != self.finalized_user_message_idx:
            previous_message: str = self.messages[self.last_user_message_idx].message
            self.messages[self.last_user_message_idx] = Message(
                "user", time.time(), ' '.join([previous_message, message])
            )
            self.messages = self.messages[: self.last_user_message_idx + 1]
        else:
            self.messages.append(Message("user", time.time(), message))

        self.last_user_message_idx = len(self.messages) - 1

    def add_assistant_message(self, message) -> None:
        if self.messages[-1].type == "assistant":
            self.messages[-1].message += " " + message
        else:
            self.messages.append(Message("assistant", time.time(), message))

    def add_assistant_messages(self, messages) -> None:
        self.messages.append(Message("assistant", time.time(), " ".join(messages)))

    def get_llm_messages(self) -> list[dict[str, str]]:
        return [{"role": m.type, "content": m.message} for m in self.messages]

    def finalize_user_message(self) -> None:
        self.finalized_user_message_idx = self.last_user_message_idx

    def shutdown(self) -> None:
        pass

class IndexingMessageHandler(MessageHandler):
    def __init__(
        self, intro, services: AIServiceConfig, indexer: SearchIndexer
    ) -> None:
        super().__init__(intro)
        self.services = services

        self.search_indexer = indexer

        self.last_written_idx = 0
        self.storage_message_queue = Queue()

        self.index_writer_thread = Thread(target=self.storage_writer, daemon=True)
        self.index_writer_thread.start()

        self.logger = logging.getLogger("bot-instance")

    def shutdown(self):
        self.finalize_user_message()
        self.storage_message_queue.put(None)
        self.index_writer_thread.join()

    def storage_writer(self) -> None:
        while True:
            try:
                message_idx = self.storage_message_queue.get()
                self.storage_message_queue.task_done()

                if message_idx is None:
                    return

                if message_idx <= self.last_written_idx:
                    continue

                self.last_written_idx = message_idx

                message = self.messages[message_idx]
                content = message.message
                if message.type == "user":
                    content = self.cleanup_user_message(content)

                    # sometimes the LLM returns a string wrapped in quotes and sometimes it doesn't.
                    # if it didn't, wrap it in quotes
                    if content[0] != '"':
                        content = '"' + content + '"'

                self.search_indexer.index_text(content)
            except Empty:
                pass

    def cleanup_user_message(self, user_message) -> str:
        return user_message

    def finalize_user_message(self):
        super().finalize_user_message()
        self.write_messages_to_storage()

    def write_messages_to_storage(self):
        if self.finalized_user_message_idx is None:
            return

        for idx in range(self.last_written_idx, len(self.messages)):
            self.logger.info(
                f"Writing to storage: {self.messages[idx].type} {self.messages[idx].message}"
            )
            if (
                self.messages[idx].type == "user"
                and idx > self.finalized_user_message_idx
            ):
                break

            if self.messages[idx].type != "system":
                self.storage_message_queue.put(idx)
