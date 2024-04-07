import copy
import json
import re
import uuid

from pathlib import Path

from curl_cffi import requests
from tclogger import logger, OSEnver
from constants.envs import PROXIES


class OpenaiAPI:
    def __init__(self):
        self.init_requests_params()

    def init_requests_params(self):
        self.api_base = "https://chat.openai.com/backend-anon"
        self.api_me = f"{self.api_base}/me"
        self.api_models = f"{self.api_base}/models"
        self.api_chat_requirements = f"{self.api_base}/sentinel/chat-requirements"
        self.api_conversation = f"{self.api_base}/conversation"
        self.uuid = str(uuid.uuid4())
        self.requests_headers = {
            # "Accept": "*/*",
            "Accept-Encoding": "gzip, deflate, br, zstd",
            "Accept-Language": "en-US,en;q=0.9",
            "Cache-Control": "no-cache",
            "Content-Type": "application/json",
            "Oai-Device-Id": self.uuid,
            "Oai-Language": "en-US",
            "Pragma": "no-cache",
            "Referer": "https://chat.openai.com/",
            "Sec-Ch-Ua": 'Google Chrome";v="123", "Not:A-Brand";v="8", "Chromium";v="123"',
            "Sec-Ch-Ua-Mobile": "?0",
            "Sec-Ch-Ua-Platform": '"Windows"',
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
        }

    def log_request(self, url, method="GET"):
        logger.note(f"> {method}:", end=" ")
        logger.mesg(f"{url}", end=" ")

    def log_response(self, res: requests.Response, stream=False, verbose=False):
        status_code = res.status_code
        status_code_str = f"[{status_code}]"

        if status_code == 200:
            logger_func = logger.success
        else:
            logger_func = logger.warn

        logger_func(status_code_str)

        if verbose:
            if stream:
                if not hasattr(self, "content_offset"):
                    self.content_offset = 0

                for line in res.iter_lines():
                    line = line.decode("utf-8")
                    line = re.sub(r"^data:\s*", "", line)
                    if re.match(r"^\[DONE\]", line):
                        logger.success("\n[Finished]")
                        break
                    line = line.strip()
                    if line:
                        try:
                            data = json.loads(line, strict=False)
                            message_role = data["message"]["author"]["role"]
                            message_status = data["message"]["status"]
                            if (
                                message_role == "assistant"
                                and message_status == "in_progress"
                            ):
                                content = data["message"]["content"]["parts"][0]
                                delta_content = content[self.content_offset :]
                                self.content_offset = len(content)
                                logger_func(delta_content, end="")
                        except Exception as e:
                            logger.warn(e)
            else:
                logger_func(res.json())

    def get_models(self):
        self.log_request(self.api_models)
        res = requests.get(
            self.api_models,
            headers=self.requests_headers,
            proxies=PROXIES,
            timeout=10,
            impersonate="chrome120",
        )
        self.log_response(res)

    def auth(self):
        self.log_request(self.api_chat_requirements, method="POST")
        res = requests.post(
            self.api_chat_requirements,
            headers=self.requests_headers,
            proxies=PROXIES,
            timeout=10,
            impersonate="chrome120",
        )
        self.chat_requirements_token = res.json()["token"]
        self.log_response(res)

    def transform_messages(self, messages: list[dict]):
        def get_role(role):
            if role in ["system", "user", "assistant"]:
                return role
            else:
                return "system"

        new_messages = [
            {
                "author": {"role": get_role(message["role"])},
                "content": {"content_type": "text", "parts": [message["content"]]},
                "metadata": {},
            }
            for message in messages
        ]
        return new_messages

    def chat_completions(self, messages: list[dict]):
        new_headers = {
            "Accept": "text/event-stream",
            "Openai-Sentinel-Chat-Requirements-Token": self.chat_requirements_token,
        }
        requests_headers = copy.deepcopy(self.requests_headers)
        requests_headers.update(new_headers)
        post_data = {
            "action": "next",
            "messages": self.transform_messages(messages),
            "parent_message_id": "",
            "model": "text-davinci-002-render-sha",
            "timezone_offset_min": -480,
            "suggestions": [],
            "history_and_training_disabled": False,
            "conversation_mode": {"kind": "primary_assistant"},
            "force_paragen": False,
            "force_paragen_model_slug": "",
            "force_nulligen": False,
            "force_rate_limit": False,
            "websocket_request_id": str(uuid.uuid4()),
        }
        self.log_request(self.api_conversation, method="POST")
        s = requests.Session()
        res = s.post(
            self.api_conversation,
            headers=requests_headers,
            json=post_data,
            proxies=PROXIES,
            timeout=10,
            impersonate="chrome120",
            stream=True,
        )
        self.log_response(res, stream=True, verbose=True)


if __name__ == "__main__":
    api = OpenaiAPI()
    # api.get_models()
    api.auth()
    messages = [
        {"role": "system", "content": "i am Hansimov"},
        {"role": "system", "content": "i have a cat named lucky"},
        {"role": "user", "content": "Repeat my name and my cat's name"},
        {
            "role": "assistant",
            "content": "Your name is Hansimov and your cat's name is Lucky.",
        },
        {"role": "user", "content": "summarize our conversation"},
    ]
    api.chat_completions(messages)

    # python -m tests.openai
