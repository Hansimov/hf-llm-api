---
title: HF LLM API
emoji: ☯️
colorFrom: gray
colorTo: gray
sdk: docker
app_port: 23333
---

## HF-LLM-API
Huggingface LLM Inference API in OpenAI message format.

## Features

✅ Implemented:

- Support Models
  - `mixtral-8x7b`, `mistral-7b`
- Support OpenAI API format
  - Can use api endpoint via official `openai-python` package
- Stream response
- Docker deployment

🔨 In progress:
- [x] Support more models

## Run API service

### Run in Command Line

**Install dependencies:**

```bash
# pipreqs . --force --mode no-pin
pip install -r requirements.txt
```

**Run API:**

```bash
python -m apis.chat_api
```

## Run via Docker

**Docker build:**

```bash
sudo docker build -t hf-llm-api:1.0 . --build-arg http_proxy=$http_proxy --build-arg https_proxy=$https_proxy
```

**Docker run:**

```bash
# no proxy
sudo docker run -p 23333:23333 hf-llm-api:1.0

# with proxy
sudo docker run -p 23333:23333 --env http_proxy="http://<server>:<port>" hf-llm-api:1.0
```

## API Usage

### Using `openai-python`

See: [examples/chat_with_openai.py](https://github.com/Hansimov/hf-llm-api/blob/main/examples/chat_with_openai.py)

```py
from openai import OpenAI

# If runnning this service with proxy, you might need to unset `http(s)_proxy`.
base_url = "http://127.0.0.1:23333"
api_key = "sk-xxxxx"

client = OpenAI(base_url=base_url, api_key=api_key)
response = client.chat.completions.create(
    model="mixtral-8x7b",
    messages=[
        {
            "role": "user",
            "content": "what is your model",
        }
    ],
    stream=True,
)

for chunk in response:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="", flush=True)
    elif chunk.choices[0].finish_reason == "stop":
        print()
    else:
        pass
```

### Using post requests

See: [examples/chat_with_post.py](https://github.com/Hansimov/hf-llm-api/blob/main/examples/chat_with_post.py)


```py
import ast
import httpx
import json
import re

# If runnning this service with proxy, you might need to unset `http(s)_proxy`.
chat_api = "http://127.0.0.1:23333"
api_key = "sk-xxxxx"
requests_headers = {}
requests_payload = {
    "model": "mixtral-8x7b",
    "messages": [
        {
            "role": "user",
            "content": "what is your model",
        }
    ],
    "stream": True,
}

with httpx.stream(
    "POST",
    chat_api + "/chat/completions",
    headers=requests_headers,
    json=requests_payload,
    timeout=httpx.Timeout(connect=20, read=60, write=20, pool=None),
) as response:
    # https://docs.aiohttp.org/en/stable/streams.html
    # https://github.com/openai/openai-cookbook/blob/main/examples/How_to_stream_completions.ipynb
    response_content = ""
    for line in response.iter_lines():
        remove_patterns = [r"^\s*data:\s*", r"^\s*\[DONE\]\s*"]
        for pattern in remove_patterns:
            line = re.sub(pattern, "", line).strip()

        if line:
            try:
                line_data = json.loads(line)
            except Exception as e:
                try:
                    line_data = ast.literal_eval(line)
                except:
                    print(f"Error: {line}")
                    raise e
            # print(f"line: {line_data}")
            delta_data = line_data["choices"][0]["delta"]
            finish_reason = line_data["choices"][0]["finish_reason"]
            if "role" in delta_data:
                role = delta_data["role"]
            if "content" in delta_data:
                delta_content = delta_data["content"]
                response_content += delta_content
                print(delta_content, end="", flush=True)
            if finish_reason == "stop":
                print()

```
