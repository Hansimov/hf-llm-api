import json
import re
import requests
from tiktoken import get_encoding as tiktoken_get_encoding
from messagers.message_outputer import OpenaiStreamOutputer
from utils.logger import logger
from utils.enver import enver


class MessageStreamer:
    MODEL_MAP = {
        "mixtral-8x7b": "mistralai/Mixtral-8x7B-Instruct-v0.1",  # 72.62, fast [Recommended]
        "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.2",  # 65.71, fast
        "nous-mixtral-8x7b": "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
        # "openchat-3.5": "openchat/openchat-3.5-1210",  # 68.89, fast
        # "zephyr-7b-beta": "HuggingFaceH4/zephyr-7b-beta",  # ❌ Too Slow
        # "llama-70b": "meta-llama/Llama-2-70b-chat-hf",  # ❌ Require Pro User
        # "codellama-34b": "codellama/CodeLlama-34b-Instruct-hf",  # ❌ Low Score
        # "falcon-180b": "tiiuae/falcon-180B-chat",  # ❌ Require Pro User
        "default": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    }
    STOP_SEQUENCES_MAP = {
        "mixtral-8x7b": "</s>",
        "mistral-7b": "</s>",
        "nous-mixtral-8x7b": "<|im_end|>",
        "openchat-3.5": "<|end_of_turn|>",
    }
    TOKEN_LIMIT_MAP = {
        "mixtral-8x7b": 32768,
        "mistral-7b": 32768,
        "nous-mixtral-8x7b": 32768,
        "openchat-3.5": 8192,
    }
    TOKEN_RESERVED = 100

    def __init__(self, model: str):
        if model in self.MODEL_MAP.keys():
            self.model = model
        else:
            self.model = "default"
        self.model_fullname = self.MODEL_MAP[self.model]
        self.message_outputer = OpenaiStreamOutputer()
        self.tokenizer = tiktoken_get_encoding("cl100k_base")

    def parse_line(self, line):
        line = line.decode("utf-8")
        line = re.sub(r"data:\s*", "", line)
        data = json.loads(line)
        try:
            content = data["token"]["text"]
        except:
            logger.err(data)
        return content

    def count_tokens(self, text):
        tokens = self.tokenizer.encode(text)
        token_count = len(tokens)
        logger.note(f"Prompt Token Count: {token_count}")
        return token_count

    def chat_response(
        self,
        prompt: str = None,
        temperature: float = 0.5,
        top_p: float = 0.95,
        max_new_tokens: int = None,
        api_key: str = None,
        use_cache: bool = False,
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

        if temperature is None or temperature < 0:
            temperature = 0.0
        # temperature must  0 < and < 1 for HF LLM models
        temperature = max(temperature, 0.01)
        temperature = min(temperature, 0.99)
        top_p = max(top_p, 0.01)
        top_p = min(top_p, 0.99)

        token_limit = int(
            self.TOKEN_LIMIT_MAP[self.model]
            - self.TOKEN_RESERVED
            - self.count_tokens(prompt) * 1.35
        )
        if token_limit <= 0:
            raise ValueError("Prompt exceeded token limit!")

        if max_new_tokens is None or max_new_tokens <= 0:
            max_new_tokens = token_limit
        else:
            max_new_tokens = min(max_new_tokens, token_limit)

        # References:
        #   huggingface_hub/inference/_client.py:
        #     class InferenceClient > def text_generation()
        #   huggingface_hub/inference/_text_generation.py:
        #     class TextGenerationRequest > param `stream`
        # https://huggingface.co/docs/text-generation-inference/conceptual/streaming#streaming-with-curl
        # https://huggingface.co/docs/api-inference/detailed_parameters#text-generation-task
        self.request_body = {
            "inputs": prompt,
            "parameters": {
                "temperature": temperature,
                "top_p": top_p,
                "max_new_tokens": max_new_tokens,
                "return_full_text": False,
            },
            "options": {
                "use_cache": use_cache,
            },
            "stream": True,
        }

        if self.model in self.STOP_SEQUENCES_MAP.keys():
            self.stop_sequences = self.STOP_SEQUENCES_MAP[self.model]
        #     self.request_body["parameters"]["stop_sequences"] = [
        #         self.STOP_SEQUENCES[self.model]
        #     ]

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

        final_content = ""
        for line in stream_response.iter_lines():
            if not line:
                continue
            content = self.parse_line(line)

            if content.strip() == self.stop_sequences:
                logger.success("\n[Finished]")
                break
            else:
                logger.back(content, end="")
                final_content += content

        if self.model in self.STOP_SEQUENCES_MAP.keys():
            final_content = final_content.replace(self.stop_sequences, "")

        final_content = final_content.strip()
        final_output["choices"][0]["message"]["content"] = final_content
        return final_output

    def chat_return_generator(self, stream_response):
        is_finished = False
        line_count = 0
        for line in stream_response.iter_lines():
            if line:
                line_count += 1
            else:
                continue

            content = self.parse_line(line)

            if content.strip() == self.stop_sequences:
                content_type = "Finished"
                logger.success("\n[Finished]")
                is_finished = True
            else:
                content_type = "Completions"
                if line_count == 1:
                    content = content.lstrip()
                logger.back(content, end="")

            output = self.message_outputer.output(
                content=content, content_type=content_type
            )
            yield output

        if not is_finished:
            yield self.message_outputer.output(content="", content_type="Finished")
