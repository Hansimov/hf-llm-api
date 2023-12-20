import json
import re
import requests
from messagers.message_outputer import OpenaiStreamOutputer
from utils.logger import logger
from utils.enver import enver


class MessageStreamer:
    MODEL_MAP = {
        "mixtral-8x7b": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "default": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    }

    def __init__(self, model: str):
        if model in self.MODEL_MAP.keys():
            self.model = model
        else:
            self.model = "default"
        self.model_fullname = self.MODEL_MAP[self.model]

    def parse_line(self, line):
        line = line.decode("utf-8")
        line = re.sub(r"data:\s*", "", line)
        data = json.loads(line)
        content = data["token"]["text"]
        return content

    def chat(
        self,
        prompt: str = None,
        temperature: float = 0.01,
        max_new_tokens: int = 8192,
        stream: bool = True,
        yield_output: bool = False,
    ):
        # https://huggingface.co/docs/text-generation-inference/conceptual/streaming#streaming-with-curl
        self.request_url = (
            f"https://api-inference.huggingface.co/models/{self.model_fullname}"
        )
        self.message_outputer = OpenaiStreamOutputer()
        self.request_headers = {
            "Content-Type": "application/json",
        }
        # huggingface_hub/inference/_client.py: class InferenceClient > def text_generation()
        self.request_body = {
            "inputs": prompt,
            "parameters": {
                "temperature": max(temperature, 0.01),  # must be positive
                "max_new_tokens": max_new_tokens,
                "return_full_text": False,
            },
            "stream": stream,
        }
        print(self.request_url)
        enver.set_envs(proxies=True)
        stream = requests.post(
            self.request_url,
            headers=self.request_headers,
            json=self.request_body,
            proxies=enver.requests_proxies,
            stream=stream,
        )
        print(stream.status_code)
        for line in stream.iter_lines():
            if not line:
                continue

            content = self.parse_line(line)

            if content.strip() == "</s>":
                content_type = "Finished"
                logger.mesg("\n[Finished]")
            else:
                content_type = "Completions"
                logger.mesg(content, end="")

            if yield_output:
                output = self.message_outputer.output(
                    content=content, content_type=content_type
                )
                yield output
