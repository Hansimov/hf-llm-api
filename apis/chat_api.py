import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse
from utils.logger import logger
from networks.message_streamer import MessageStreamer
from messagers.message_composer import MessageComposer


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
                "description": "Mixtral-8x7B: https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1",
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
            default=32000,
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
        return EventSourceResponse(
            streamer.chat(
                prompt=composer.merged_str,
                temperature=item.temperature,
                max_new_tokens=item.max_tokens,
                stream=item.stream,
                yield_output=True,
            ),
            media_type="text/event-stream",
        )

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


app = ChatAPIApp().app

if __name__ == "__main__":
    uvicorn.run("__main__:app", host="0.0.0.0", port=23333, reload=True)
