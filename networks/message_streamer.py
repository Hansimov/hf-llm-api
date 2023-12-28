import json
import re
import requests
from messagers.message_outputer import OpenaiStreamOutputer
from utils.logger import logger
from utils.enver import enver


class MessageStreamer:
    MODEL_MAP = {
        "mixtral-8x7b": "mistralai/Mixtral-8x7B-Instruct-v0.1",  # 72.62, fast [Recommended]
        "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.2",  # 65.71, fast
        "openchat-3.5": "openchat/openchat_3.5",  # 61.24, fast
        # "zephyr-7b-alpha": "HuggingFaceH4/zephyr-7b-alpha",  # 59.5, fast
        # "zephyr-7b-beta": "HuggingFaceH4/zephyr-7b-beta",  # 61.95, slow
        "default": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    }

    def __init__(self, model: str):
        if model in self.MODEL_MAP.keys():
            self.model = model
        else:
            self.model = "default"
        self.model_fullname = self.MODEL_MAP[self.model]
        self.message_outputer = OpenaiStreamOutputer()

    def parse_line(self, line):
        line = line.decode("utf-8")
        line = re.sub(r"data:\s*", "", line)
        data = json.loads(line)
        content = data["token"]["text"]
        return content

    def chat_response(
        self,
        prompt: str = None,
        temperature: float = 0.01,
        max_new_tokens: int = 8192,
        api_key: str = None,
    ):
        # https://huggingface.co/docs/api-inference/detailed_parameters?code=curl
        # curl --proxy http://<server>:<port> https://api-inference.huggingface.co/models/<org>/<model_name> -X POST -d '{"inputs":"who are you?","parameters":{"max_new_token":64}}' -H 'Content-Type: application/json' -H 'Authorization: Bearer <HF_TOKEN>'
        self.request_url = (
            f"https://api-inference.huggingface.co/models/{self.model_fullname}"
        )
        self.request_headers = {
            "Content-Type": "application/json",
        }

        if api_key:
            logger.note(
                f"Using API Key: {api_key[:3]}{(len(api_key)-7)*'*'}{api_key[-4:]}"
            )
            self.request_headers["Authorization"] = f"Bearer {api_key}"

        # References:
        #   huggingface_hub/inference/_client.py:
        #     class InferenceClient > def text_generation()
        #   huggingface_hub/inference/_text_generation.py:
        #     class TextGenerationRequest > param `stream`
        # https://huggingface.co/docs/text-generation-inference/conceptual/streaming#streaming-with-curl
        self.request_body = {
            "inputs": prompt,
            "parameters": {
                "temperature": max(temperature, 0.01),  # must be positive
                "max_new_tokens": max_new_tokens,
                "return_full_text": False,
            },
            "stream": True,
        }
        logger.back(self.request_url)
        enver.set_envs(proxies=True)
        stream_response = requests.post(
            self.request_url,
            headers=self.request_headers,
            json=self.request_body,
            proxies=enver.requests_proxies,
            stream=True,
        )
        status_code = stream_response.status_code
        if status_code == 200:
            logger.success(status_code)
        else:
            logger.err(status_code)

        return stream_response

    def chat_return_dict(self, stream_response):
        # https://platform.openai.com/docs/guides/text-generation/chat-completions-response-format
        final_output = self.message_outputer.default_data.copy()
        final_output["choices"] = [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": "",
                },
            }
        ]
        logger.back(final_output)

        for line in stream_response.iter_lines():
            if not line:
                continue
            content = self.parse_line(line)

            if content.strip() == "</s>":
                logger.success("\n[Finished]")
                break
            else:
                logger.back(content, end="")
                final_output["choices"][0]["message"]["content"] += content

        return final_output

    def chat_return_generator(self, stream_response):
        is_finished = False
        for line in stream_response.iter_lines():
            if not line:
                continue

            content = self.parse_line(line)

            if content.strip() == "</s>":
                content_type = "Finished"
                logger.success("\n[Finished]")
                is_finished = True
            else:
                content_type = "Completions"
                logger.back(content, end="")

            output = self.message_outputer.output(
                content=content, content_type=content_type
            )
            yield output

        if not is_finished:
            yield self.message_outputer.output(content="", content_type="Finished")
