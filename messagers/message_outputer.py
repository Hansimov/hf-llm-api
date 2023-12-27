import json


class OpenaiStreamOutputer:
    """
    Create chat completion - OpenAI API Documentation
    * https://platform.openai.com/docs/api-reference/chat/create
    """

    def __init__(self):
        self.default_data = {
            "created": 1700000000,
            "id": "chatcmpl-hugginface",
            "object": "chat.completion.chunk",
            # "content_type": "Completions",
            "model": "hugginface",
            "choices": [],
        }

    def data_to_string(self, data={}, content_type=""):
        data_str = f"{json.dumps(data)}"
        return data_str

    def output(self, content=None, content_type="Completions") -> str:
        data = self.default_data.copy()
        if content_type == "Role":
            data["choices"] = [
                {
                    "index": 0,
                    "delta": {"role": "assistant"},
                    "finish_reason": None,
                }
            ]
        elif content_type in [
            "Completions",
            "InternalSearchQuery",
            "InternalSearchResult",
            "SuggestedResponses",
        ]:
            if content_type in ["InternalSearchQuery", "InternalSearchResult"]:
                content += "\n"
            data["choices"] = [
                {
                    "index": 0,
                    "delta": {"content": content},
                    "finish_reason": None,
                }
            ]
        elif content_type == "Finished":
            data["choices"] = [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }
            ]
        else:
            data["choices"] = [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": None,
                }
            ]
        return self.data_to_string(data, content_type)
