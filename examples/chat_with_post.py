import ast
import httpx
import json
import re

# If runnning this service with proxy, you might need to unset `http(s)_proxy`.
chat_api = "http://127.0.0.1:23333"
api_key = "sk-xxxxx"
requests_headers = {}
requests_payload = {
    "model": "nous-mixtral-8x7b",
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
