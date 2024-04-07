from pathlib import Path
from tclogger import logger, OSEnver

secrets_path = Path(__file__).parents[1] / "secrets.json"
ENVER = OSEnver(secrets_path)

http_proxy = ENVER["http_proxy"]
if http_proxy:
    logger.note(f"> Using proxy: {http_proxy}")
    PROXIES = {
        "http": http_proxy,
        "https": http_proxy,
    }
else:
    PROXIES = None
