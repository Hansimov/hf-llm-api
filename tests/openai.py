import uuid

from pathlib import Path

from curl_cffi import requests
from tclogger import logger, OSEnver

secrets_path = Path(__file__).parents[1] / "secrets.json"
ENVER = OSEnver(secrets_path)


class OpenaiAPI:
    def __init__(self):
        self.api_me = "https://chat.openai.com/backend-anon/me"
        self.api_models = "https://chat.openai.com/backend-anon/models"

    def auth(self):
        http_proxy = ENVER["http_proxy"]
        if http_proxy:
            logger.note(f"> Using Proxy: {http_proxy}")
        requests_proxies = {
            "http": http_proxy,
            "https": http_proxy,
        }
        uuid_str = str(uuid.uuid4())
        requests_headers = {
            "Accept": "*/*",
            "Accept-Encoding": "gzip, deflate, br, zstd",
            "Accept-Language": "en-US,en;q=0.9",
            "Cache-Control": "no-cache",
            "Oai-Device-Id": uuid_str,
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

        logger.note(f"> Get: {self.api_models}")

        res = requests.get(
            self.api_models,
            headers=requests_headers,
            proxies=requests_proxies,
            timeout=10,
            impersonate="chrome120",
        )

        logger.warn(res.status_code)
        logger.mesg(res.json())


if __name__ == "__main__":
    api = OpenaiAPI()
    api.auth()

    # python -m tests.openai
