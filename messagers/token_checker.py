from tclogger import logger
from transformers import AutoTokenizer

from constants.models import MODEL_MAP, TOKEN_LIMIT_MAP, TOKEN_RESERVED


class TokenChecker:
    def __init__(self, input_str: str, model: str):
        self.input_str = input_str

        if model in MODEL_MAP.keys():
            self.model = model
        else:
            self.model = "nous-mixtral-8x7b"

        self.model_fullname = MODEL_MAP[self.model]

        # As some models are gated, we need to fetch tokenizers from alternatives
        GATED_MODEL_MAP = {
            "llama3-70b": "NousResearch/Meta-Llama-3-70B",
            "gemma-7b": "unsloth/gemma-7b",
            "mistral-7b": "dfurman/Mistral-7B-Instruct-v0.2",
            "mixtral-8x7b": "dfurman/Mixtral-8x7B-Instruct-v0.1",
        }
        if self.model in GATED_MODEL_MAP.keys():
            self.tokenizer = AutoTokenizer.from_pretrained(GATED_MODEL_MAP[self.model])
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_fullname)

    def count_tokens(self):
        token_count = len(self.tokenizer.encode(self.input_str))
        logger.note(f"Prompt Token Count: {token_count}")
        return token_count

    def get_token_limit(self):
        return TOKEN_LIMIT_MAP[self.model]

    def get_token_redundancy(self):
        return int(self.get_token_limit() - TOKEN_RESERVED - self.count_tokens())

    def check_token_limit(self):
        if self.get_token_redundancy() <= 0:
            raise ValueError(
                f"Prompt exceeded token limit: {self.count_tokens()} > {self.get_token_limit()}"
            )
        return True
