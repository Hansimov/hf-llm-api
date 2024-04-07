import time
from tclogger import logger


def stream_chat_mock(*args, **kwargs):
    logger.note(msg=str(args) + str(kwargs))
    for i in range(10):
        content = f"W{i+1} "
        time.sleep(0.1)
        logger.mesg(content, end="")
        yield content
    logger.mesg("")
    yield ""
