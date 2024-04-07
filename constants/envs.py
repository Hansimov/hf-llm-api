from pathlib import Path
from tclogger import logger, OSEnver


config_root = Path(__file__).parents[1] / "configs"

secrets_path = config_root / "secrets.json"
SECRETS = OSEnver(secrets_path)

http_proxy = SECRETS["http_proxy"]
if http_proxy:
    logger.note(f"> Using proxy: {http_proxy}")
    PROXIES = {
        "http": http_proxy,
        "https": http_proxy,
    }
else:
    PROXIES = None

config_path = config_root / "config.json"
CONFIG = OSEnver(config_path)
