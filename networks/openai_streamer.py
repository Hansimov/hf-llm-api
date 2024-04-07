import copy
import json
import re
import tiktoken
import uuid

from curl_cffi import requests
from tclogger import logger

from constants.envs import PROXIES
from constants.headers import OPENAI_GET_HEADERS, OPENAI_POST_DATA
from constants.models import TOKEN_LIMIT_MAP, TOKEN_RESERVED

from messagers.message_outputer import OpenaiStreamOutputer


class OpenaiRequester:
    def __init__(self):
        self.init_requests_params()

    def init_requests_params(self):
        self.api_base = "https://chat.openai.com/backend-anon"
        self.api_me = f"{self.api_base}/me"
        self.api_models = f"{self.api_base}/models"
        self.api_chat_requirements = f"{self.api_base}/sentinel/chat-requirements"
        self.api_conversation = f"{self.api_base}/conversation"
        self.uuid = str(uuid.uuid4())
        self.requests_headers = copy.deepcopy(OPENAI_GET_HEADERS)
        extra_headers = {
            "Oai-Device-Id": self.uuid,
        }
        self.requests_headers.update(extra_headers)

    def log_request(self, url, method="GET"):
        logger.note(f"> {method}:", end=" ")
        logger.mesg(f"{url}", end=" ")

    def log_response(
        self, res: requests.Response, stream=False, iter_lines=False, verbose=False
    ):
        status_code = res.status_code
        status_code_str = f"[{status_code}]"

        if status_code == 200:
            logger_func = logger.success
        else:
            logger_func = logger.warn

        logger_func(status_code_str)

        logger.enter_quiet(not verbose)

        if stream:
            if not iter_lines:
                return

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

        logger.exit_quiet(not verbose)

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

    def chat_completions(self, messages: list[dict], verbose=False):
        extra_headers = {
            "Accept": "text/event-stream",
            "Openai-Sentinel-Chat-Requirements-Token": self.chat_requirements_token,
        }
        requests_headers = copy.deepcopy(self.requests_headers)
        requests_headers.update(extra_headers)

        post_data = copy.deepcopy(OPENAI_POST_DATA)
        extra_data = {
            "messages": self.transform_messages(messages),
            "websocket_request_id": str(uuid.uuid4()),
        }
        post_data.update(extra_data)

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
        self.log_response(res, stream=True, iter_lines=False)
        return res


class OpenaiStreamer:
    def __init__(self):
        self.model = "gpt-3.5-turbo"
        self.message_outputer = OpenaiStreamOutputer(
            owned_by="openai", model="gpt-3.5-turbo"
        )
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, messages: list[dict]):
        token_count = sum(
            len(self.tokenizer.encode(message["content"])) for message in messages
        )
        logger.note(f"Prompt Token Count: {token_count}")
        return token_count

    def check_token_limit(self, messages: list[dict]):
        token_limit = TOKEN_LIMIT_MAP[self.model]
        token_redundancy = int(
            token_limit - TOKEN_RESERVED - self.count_tokens(messages)
        )
        if token_redundancy <= 0:
            raise ValueError(f"Prompt exceeded token limit: {token_limit}")
        return True

    def chat_response(self, messages: list[dict]):
        self.check_token_limit(messages)
        requester = OpenaiRequester()
        requester.auth()
        return requester.chat_completions(messages, verbose=False)

    def chat_return_generator(self, stream_response: requests.Response, verbose=False):
        content_offset = 0
        is_finished = False

        for line in stream_response.iter_lines():
            line = line.decode("utf-8")
            line = re.sub(r"^data:\s*", "", line)
            line = line.strip()

            if not line:
                continue

            if re.match(r"^\[DONE\]", line):
                content_type = "Finished"
                delta_content = ""
                logger.success("\n[Finished]")
                is_finished = True
            else:
                content_type = "Completions"
                try:
                    data = json.loads(line, strict=False)
                    message_role = data["message"]["author"]["role"]
                    message_status = data["message"]["status"]
                    if message_role == "assistant" and message_status == "in_progress":
                        content = data["message"]["content"]["parts"][0]
                        if not len(content):
                            continue
                        delta_content = content[content_offset:]
                        content_offset = len(content)
                        if verbose:
                            logger.success(delta_content, end="")
                    else:
                        continue
                except Exception as e:
                    logger.warn(e)

            output = self.message_outputer.output(
                content=delta_content, content_type=content_type
            )
            yield output

        if not is_finished:
            yield self.message_outputer.output(content="", content_type="Finished")

    def chat_return_dict(self, stream_response: requests.Response):
        final_output = self.message_outputer.default_data.copy()
        final_output["choices"] = [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": ""},
            }
        ]
        final_content = ""
        for item in self.chat_return_generator(stream_response):
            try:
                data = json.loads(item)
                delta = data["choices"][0]["delta"]
                delta_content = delta.get("content", "")
                if delta_content:
                    final_content += delta_content
            except Exception as e:
                logger.warn(e)
        final_output["choices"][0]["message"]["content"] = final_content.strip()
        return final_output
