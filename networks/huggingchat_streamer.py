import copy
import json
import re
import requests
import uuid

# from curl_cffi import requests
from tclogger import logger
from transformers import AutoTokenizer

from constants.models import (
    MODEL_MAP,
    STOP_SEQUENCES_MAP,
    TOKEN_LIMIT_MAP,
    TOKEN_RESERVED,
)
from constants.envs import PROXIES
from constants.headers import (
    REQUESTS_HEADERS,
    HUGGINGCHAT_POST_HEADERS,
    HUGGINGCHAT_SETTINGS_POST_DATA,
)
from messagers.message_outputer import OpenaiStreamOutputer
from messagers.message_composer import MessageComposer


class TokenChecker:
    def __init__(self, input_str: str, model: str):
        self.input_str = input_str

        if model in MODEL_MAP.keys():
            self.model = model
        else:
            self.model = "mixtral-8x7b"

        self.model_fullname = MODEL_MAP[self.model]

        if self.model == "llama3-70b":
            # As original llama3 repo is gated and requires auth,
            #   I use NousResearch's version as a workaround
            self.tokenizer = AutoTokenizer.from_pretrained(
                "NousResearch/Meta-Llama-3-70B"
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_fullname)

    def count_tokens(self):
        token_count = len(self.tokenizer.encode(self.input_str))
        logger.note(f"Prompt Token Count: {token_count}")
        return token_count

    def check_token_limit(self):
        token_limit = TOKEN_LIMIT_MAP[self.model]
        token_redundancy = int(token_limit - TOKEN_RESERVED - self.count_tokens())
        if token_redundancy <= 0:
            raise ValueError(f"Prompt exceeded token limit: {token_limit}")
        return True


class HuggingchatRequester:
    def __init__(self, model: str):
        if model in MODEL_MAP.keys():
            self.model = model
        else:
            self.model = "mixtral-8x7b"
        self.model_fullname = MODEL_MAP[self.model]

    def get_hf_chat_id(self):
        request_url = "https://huggingface.co/chat/settings"
        request_body = copy.deepcopy(HUGGINGCHAT_SETTINGS_POST_DATA)
        extra_body = {
            "activeModel": self.model_fullname,
        }
        request_body.update(extra_body)
        logger.note(f"> hf-chat ID:", end=" ")

        res = requests.post(
            request_url,
            headers=HUGGINGCHAT_POST_HEADERS,
            json=request_body,
            proxies=PROXIES,
            timeout=10,
        )
        self.hf_chat_id = res.cookies.get("hf-chat")
        if self.hf_chat_id:
            logger.success(f"[{self.hf_chat_id}]")
        else:
            logger.warn(f"[{res.status_code}]")
            logger.warn(res.text)
            raise ValueError("Failed to get hf-chat ID!")

    def get_conversation_id(self, system_prompt: str = ""):
        request_url = "https://huggingface.co/chat/conversation"
        request_headers = HUGGINGCHAT_POST_HEADERS
        extra_headers = {
            "Cookie": f"hf-chat={self.hf_chat_id}",
        }
        request_headers.update(extra_headers)
        request_body = {
            "model": self.model_fullname,
            "preprompt": system_prompt,
        }
        logger.note(f"> Conversation ID:", end=" ")

        res = requests.post(
            request_url,
            headers=request_headers,
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
        return conversation_id

    def get_last_message_id(self):
        request_url = f"https://huggingface.co/chat/conversation/{self.conversation_id}/__data.json?x-sveltekit-invalidated=11"
        request_headers = HUGGINGCHAT_POST_HEADERS
        extra_headers = {
            "Cookie": f"hf-chat={self.hf_chat_id}",
        }
        request_headers.update(extra_headers)
        logger.note(f"> Message ID:", end=" ")

        message_id = None
        res = requests.post(
            request_url,
            headers=request_headers,
            proxies=PROXIES,
            timeout=10,
        )
        if res.status_code == 200:
            data = res.json()["nodes"][1]["data"]
            # find the last element which matches the format of uuid4
            uuid_pattern = re.compile(
                r"^[\da-f]{8}-[\da-f]{4}-[\da-f]{4}-[\da-f]{4}-[\da-f]{12}$"
            )
            for item in data:
                if type(item) == str and uuid_pattern.match(item):
                    message_id = item
            logger.success(f"[{message_id}]")
        else:
            logger.warn(f"[{res.status_code}]")
            raise ValueError("Failed to get conversation ID!")

        return message_id

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

        logger.enter_quiet(not verbose)
        logger_func(status_code_str)

        if status_code != 200:
            logger_func(res.text)

        if stream:
            if not iter_lines:
                return

            for line in res.iter_lines():
                line = line.decode("utf-8")
                line = re.sub(r"^data:\s*", "", line)
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line, strict=False)
                        msg_type = data.get("type")
                        if msg_type == "status":
                            msg_status = data.get("status")
                        elif msg_type == "stream":
                            content = data.get("token", "")
                            logger_func(content, end="")
                        elif msg_type == "finalAnswer":
                            full_content = data.get("text")
                            logger.success("\n[Finished]")
                            break
                        else:
                            pass
                    except Exception as e:
                        logger.warn(e)
        else:
            logger_func(res.json())

        logger.exit_quiet(not verbose)

    def chat_completions(self, messages: list[dict], iter_lines=False, verbose=False):
        composer = MessageComposer(model=self.model)
        system_prompt, input_prompt = composer.decompose_to_system_and_input_prompt(
            messages
        )

        checker = TokenChecker(input_str=system_prompt + input_prompt, model=self.model)
        checker.check_token_limit()

        logger.enter_quiet(not verbose)
        self.get_hf_chat_id()
        self.get_conversation_id(system_prompt=system_prompt)
        message_id = self.get_last_message_id()

        request_url = f"https://huggingface.co/chat/conversation/{self.conversation_id}"
        request_headers = copy.deepcopy(HUGGINGCHAT_POST_HEADERS)
        extra_headers = {
            "Content-Type": "text/event-stream",
            "Referer": request_url,
            "Cookie": f"hf-chat={self.hf_chat_id}",
        }
        request_headers.update(extra_headers)
        request_body = {
            "files": [],
            "id": message_id,
            "inputs": input_prompt,
            "is_continue": False,
            "is_retry": False,
            "web_search": False,
        }
        self.log_request(request_url, method="POST")
        logger.exit_quiet(not verbose)

        res = requests.post(
            request_url,
            headers=request_headers,
            json=request_body,
            proxies=PROXIES,
            stream=True,
        )
        self.log_response(res, stream=True, iter_lines=iter_lines, verbose=verbose)
        return res


class HuggingchatStreamer:
    def __init__(self, model: str):
        if model in MODEL_MAP.keys():
            self.model = model
        else:
            self.model = "mixtral-8x7b"
        self.model_fullname = MODEL_MAP[self.model]
        self.message_outputer = OpenaiStreamOutputer(model=self.model)

    def chat_response(self, messages: list[dict], verbose=False):
        requester = HuggingchatRequester(model=self.model)
        return requester.chat_completions(
            messages=messages, iter_lines=False, verbose=verbose
        )

    def chat_return_generator(self, stream_response: requests.Response, verbose=False):
        is_finished = False
        for line in stream_response.iter_lines():
            line = line.decode("utf-8")
            line = re.sub(r"^data:\s*", "", line)
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line, strict=False)
                msg_type = data.get("type")
                if msg_type == "status":
                    msg_status = data.get("status")
                    continue
                elif msg_type == "stream":
                    content_type = "Completions"
                    content = data.get("token", "")
                    if verbose:
                        logger.success(content, end="")
                elif msg_type == "finalAnswer":
                    content_type = "Finished"
                    content = ""
                    full_content = data.get("text")
                    if verbose:
                        logger.success("\n[Finished]")
                    is_finished = True
                    break
                else:
                    continue
            except Exception as e:
                logger.warn(e)

            output = self.message_outputer.output(
                content=content, content_type=content_type
            )
            yield output

        if not is_finished:
            yield self.message_outputer.output(content="", content_type="Finished")

    def chat_return_dict(self, stream_response):
        pass


if __name__ == "__main__":
    # model = "command-r-plus"
    model = "llama3-70b"
    # model = "zephyr-141b"

    streamer = HuggingchatStreamer(model=model)
    messages = [
        {
            "role": "system",
            "content": "You are an LLM developed by CloseAI.\nYour name is Hansimov-Copilot.",
        },
        {"role": "user", "content": "Hello, what is your role?"},
        {"role": "assistant", "content": "I am an LLM."},
        {"role": "user", "content": "What is your name?"},
    ]

    streamer.chat_response(messages=messages)
    # HF_ENDPOINT=https://hf-mirror.com python -m networks.huggingchat_streamer
