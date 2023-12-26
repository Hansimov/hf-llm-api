import time
from utils.logger import logger


def stream_chat_mock():
    for i in range(8):
        content = f"W{i+1} "
        time.sleep(1.5)
        logger.mesg(content, end="")
        yield content
    logger.mesg("")
    yield ""
