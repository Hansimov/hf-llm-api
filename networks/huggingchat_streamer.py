import json
import re
import requests


from tclogger import logger
from transformers import AutoTokenizer

from constants.models import (
    MODEL_MAP,
    STOP_SEQUENCES_MAP,
    TOKEN_LIMIT_MAP,
    TOKEN_RESERVED,
)
from constants.envs import PROXIES
from constants.networks import REQUESTS_HEADERS
from messagers.message_outputer import OpenaiStreamOutputer


class HuggingchatStreamer:
    def __init__(self, model: str):
        if model in MODEL_MAP.keys():
            self.model = model
        else:
            self.model = "mixtral-8x7b"
        self.model_fullname = MODEL_MAP[self.model]
        self.message_outputer = OpenaiStreamOutputer(model=self.model)
        # export HF_ENDPOINT=https://hf-mirror.com
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_fullname)

    def count_tokens(self, text):
        tokens = self.tokenizer.encode(text)
        token_count = len(tokens)
        logger.note(f"Prompt Token Count: {token_count}")
        return token_count

    def get_conversation_id(self, preprompt: str = ""):
        request_url = "https://huggingface.co/chat/conversation"
        request_body = {
            "model": self.model_fullname,
            "preprompt": preprompt,
        }
        logger.note(f"> Conversation ID:", end=" ")
        res = requests.post(
            request_url,
            headers=REQUESTS_HEADERS,
            json=request_body,
            proxies=PROXIES,
            timeout=10,
        )
        if res.status_code == 200:
            conversation_id = res.json()["conversationId"]
            logger.success(f"[{conversation_id}]")
        else:
            logger.warn(f"[{res.status_code}]")
            raise ValueError("Failed to get conversation ID!")
        self.conversation_id = conversation_id

    def chat_response(
        self,
        prompt: str = None,
        temperature: float = 0.5,
        top_p: float = 0.95,
        max_new_tokens: int = None,
        api_key: str = None,
        use_cache: bool = False,
    ):
        pass

    def chat_return_dict(self, stream_response):
        pass

    def chat_return_generator(self, stream_response):
        pass


if __name__ == "__main__":
    streamer = HuggingchatStreamer(model="mixtral-8x7b")
    conversation_id = streamer.get_conversation_id()
    # python -m networks.huggingchat_streamer
