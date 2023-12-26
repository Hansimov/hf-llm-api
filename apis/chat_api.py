import argparse
import uvicorn
import sys

from fastapi import FastAPI
from pydantic import BaseModel, Field
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
        self.available_models = [
            {
                "id": "mixtral-8x7b",
                "description": "[Mixtral-8x7B-Instruct-v0.1]: https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1",
            },
            {
                "id": "mistral-7b",
                "description": "[Mistral-7B-Instruct-v0.2]: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2",
            },
            {
                "id": "openchat-3.5",
                "description": "[openchat_3.5]: https://huggingface.co/openchat/openchat_3.5",
            },
        ]
        return self.available_models

    class ChatCompletionsPostItem(BaseModel):
        model: str = Field(
            default="mixtral-8x7b",
            description="(str) `mixtral-8x7b`",
        )
        messages: list = Field(
            default=[{"role": "user", "content": "Hello, who are you?"}],
            description="(list) Messages",
        )
        temperature: float = Field(
            default=0.01,
            description="(float) Temperature",
        )
        max_tokens: int = Field(
            default=8192,
            description="(int) Max tokens",
        )
        stream: bool = Field(
            default=True,
            description="(bool) Stream",
        )

    def chat_completions(self, item: ChatCompletionsPostItem):
        streamer = MessageStreamer(model=item.model)
        composer = MessageComposer(model=item.model)
        composer.merge(messages=item.messages)
        event_source_response = EventSourceResponse(
            streamer.chat(
                prompt=composer.merged_str,
                temperature=item.temperature,
                max_new_tokens=item.max_tokens,
                stream=item.stream,
                yield_output=True,
            ),
            media_type="text/event-stream",
            ping=2000,
            ping_message_factory=lambda: ServerSentEvent(**{"comment": ""}),
        )
        return event_source_response

    def setup_routes(self):
        for prefix in ["", "/v1"]:
            self.app.get(
                prefix + "/models",
                summary="Get available models",
            )(self.get_available_models)

            self.app.post(
                prefix + "/chat/completions",
                summary="Chat completions in conversation session",
            )(self.chat_completions)


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
