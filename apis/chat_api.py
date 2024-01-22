import argparse
import markdown2
import os
import sys
import uvicorn

from pathlib import Path
from fastapi import FastAPI, Depends
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import Union
from sse_starlette.sse import EventSourceResponse, ServerSentEvent
from utils.logger import logger
from networks.message_streamer import MessageStreamer
from messagers.message_composer import MessageComposer
from mocks.stream_chat_mocker import stream_chat_mock


class ChatAPIApp:
    def __init__(self):
        self.app = FastAPI(
            docs_url="/",
            title="HuggingFace LLM API",
            swagger_ui_parameters={"defaultModelsExpandDepth": -1},
            version="1.0",
        )
        self.setup_routes()

    def get_available_models(self):
        # https://platform.openai.com/docs/api-reference/models/list
        # ANCHOR[id=available-models]: Available models
        self.available_models = {
            "object": "list",
            "data": [
                {
                    "id": "mixtral-8x7b",
                    "description": "[mistralai/Mixtral-8x7B-Instruct-v0.1]: https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1",
                    "object": "model",
                    "created": 1700000000,
                    "owned_by": "mistralai",
                },
                {
                    "id": "mistral-7b",
                    "description": "[mistralai/Mistral-7B-Instruct-v0.2]: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2",
                    "object": "model",
                    "created": 1700000000,
                    "owned_by": "mistralai",
                },
                {
                    "id": "nous-mixtral-8x7b",
                    "description": "[NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO]: https://huggingface.co/NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
                    "object": "model",
                    "created": 1700000000,
                    "owned_by": "NousResearch",
                },
            ],
        }
        return self.available_models

    def extract_api_key(
        credentials: HTTPAuthorizationCredentials = Depends(
            HTTPBearer(auto_error=False)
        ),
    ):
        api_key = None
        if credentials:
            api_key = credentials.credentials
        else:
            api_key = os.getenv("HF_TOKEN")

        if api_key:
            if api_key.startswith("hf_"):
                return api_key
            else:
                logger.warn(f"Invalid HF Token!")
        else:
            logger.warn("Not provide HF Token!")
        return None

    class ChatCompletionsPostItem(BaseModel):
        model: str = Field(
            default="mixtral-8x7b",
            description="(str) `mixtral-8x7b`",
        )
        messages: list = Field(
            default=[{"role": "user", "content": "Hello, who are you?"}],
            description="(list) Messages",
        )
        temperature: Union[float, None] = Field(
            default=0,
            description="(float) Temperature",
        )
        max_tokens: Union[int, None] = Field(
            default=-1,
            description="(int) Max tokens",
        )
        stream: bool = Field(
            default=True,
            description="(bool) Stream",
        )

    def chat_completions(
        self, item: ChatCompletionsPostItem, api_key: str = Depends(extract_api_key)
    ):
        streamer = MessageStreamer(model=item.model)
        composer = MessageComposer(model=item.model)
        composer.merge(messages=item.messages)
        # streamer.chat = stream_chat_mock

        stream_response = streamer.chat_response(
            prompt=composer.merged_str,
            temperature=item.temperature,
            max_new_tokens=item.max_tokens,
            api_key=api_key,
        )
        if item.stream:
            event_source_response = EventSourceResponse(
                streamer.chat_return_generator(stream_response),
                media_type="text/event-stream",
                ping=2000,
                ping_message_factory=lambda: ServerSentEvent(**{"comment": ""}),
            )
            return event_source_response
        else:
            data_response = streamer.chat_return_dict(stream_response)
            return data_response

    def get_readme(self):
        readme_path = Path(__file__).parents[1] / "README.md"
        with open(readme_path, "r", encoding="utf-8") as rf:
            readme_str = rf.read()
        readme_html = markdown2.markdown(
            readme_str, extras=["table", "fenced-code-blocks", "highlightjs-lang"]
        )
        return readme_html

    def setup_routes(self):
        for prefix in ["", "/v1", "/api", "/api/v1"]:
            if prefix in ["/api/v1"]:
                include_in_schema = True
            else:
                include_in_schema = False

            self.app.get(
                prefix + "/models",
                summary="Get available models",
                include_in_schema=include_in_schema,
            )(self.get_available_models)

            self.app.post(
                prefix + "/chat/completions",
                summary="Chat completions in conversation session",
                include_in_schema=include_in_schema,
            )(self.chat_completions)
        self.app.get(
            "/readme",
            summary="README of HF LLM API",
            response_class=HTMLResponse,
            include_in_schema=False,
        )(self.get_readme)


class ArgParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super(ArgParser, self).__init__(*args, **kwargs)

        self.add_argument(
            "-s",
            "--server",
            type=str,
            default="0.0.0.0",
            help="Server IP for HF LLM Chat API",
        )
        self.add_argument(
            "-p",
            "--port",
            type=int,
            default=23333,
            help="Server Port for HF LLM Chat API",
        )

        self.add_argument(
            "-d",
            "--dev",
            default=False,
            action="store_true",
            help="Run in dev mode",
        )

        self.args = self.parse_args(sys.argv[1:])


app = ChatAPIApp().app

if __name__ == "__main__":
    args = ArgParser().args
    if args.dev:
        uvicorn.run("__main__:app", host=args.server, port=args.port, reload=True)
    else:
        uvicorn.run("__main__:app", host=args.server, port=args.port, reload=False)

    # python -m apis.chat_api      # [Docker] on product mode
    # python -m apis.chat_api -d   # [Dev]    on develop mode
