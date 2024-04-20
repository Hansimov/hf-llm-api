import http

from typing import Optional

from fastapi import HTTPException, status


class HfApiException(Exception):
    def __init__(
        self,
        status_code: int,
        detail: Optional[str] = None,
    ) -> None:
        if detail is None:
            self.detail = http.HTTPStatus(status_code).phrase
        else:
            self.detail = detail
        self.status_code = status_code

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        return f"{class_name}(status_code={self.status_code!r}, detail={self.detail!r})"

    def __str__(self) -> str:
        return self.__repr__()


INVALID_API_KEY_ERROR = HfApiException(
    status_code=status.HTTP_403_FORBIDDEN,
    detail="Invalid API Key",
)
