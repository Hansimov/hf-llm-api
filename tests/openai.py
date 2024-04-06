import uuid

from pathlib import Path

from curl_cffi import requests
from tclogger import logger, OSEnver

secrets_path = Path(__file__).parents[1] / "secrets.json"
ENVER = OSEnver(secrets_path)


class OpenaiAPI:
    def __init__(self):
        self.init_requests_params()

    def init_requests_params(self):
        self.api_base = "https://chat.openai.com/backend-anon"
        self.api_me = f"{self.api_base}/me"
        self.api_models = f"{self.api_base}/models"
        self.api_chat_requirements = f"{self.api_base}/sentinel/chat-requirements"
        self.uuid = str(uuid.uuid4())
        self.requests_headers = {
            "Accept": "*/*",
            "Accept-Encoding": "gzip, deflate, br, zstd",
            "Accept-Language": "en-US,en;q=0.9",
            "Cache-Control": "no-cache",
            "Oai-Device-Id": self.uuid,
            "Oai-Language": "en-US",
            "Pragma": "no-cache",
            "Referer": "https://chat.openai.com/",
            "Sec-Ch-Ua": 'Google Chrome";v="123", "Not:A-Brand";v="8", "Chromium";v="123"',
            "Sec-Ch-Ua-Mobile": "?0",
            "Sec-Ch-Ua-Platform": '"Windows"',
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
        }

        http_proxy = ENVER["http_proxy"]
        self.requests_proxies = {
            "http": http_proxy,
            "https": http_proxy,
        }

    def log_request(self, url, method="GET"):
        if ENVER["http_proxy"]:
            logger.note(f"> Using Proxy: {ENVER['http_proxy']}")
        logger.note(f"> {method}: {url}", end=" ")

    def log_response(self, res: requests.Response):
        status_code = res.status_code
        status_code_str = f"[{status_code}]"
        if status_code == 200:
            logger.success(status_code_str)
        else:
            logger.warn(f"uuid: {self.uuid}")
            logger.warn(status_code_str)

        logger.mesg(res.json())

    def get_models(self):
        self.log_request(self.api_models)
        res = requests.get(
            self.api_models,
            headers=self.requests_headers,
            proxies=self.requests_proxies,
            timeout=10,
            impersonate="chrome120",
        )

        self.log_response(res)

    def auth(self):
        self.log_request(self.api_models, method="POST")
        res = requests.post(
            self.api_chat_requirements,
            headers=self.requests_headers,
            proxies=self.requests_proxies,
            timeout=10,
            impersonate="chrome120",
        )

        self.log_response(res)


if __name__ == "__main__":
    api = OpenaiAPI()
    api.auth()

    # python -m tests.openai
