MODEL_MAP = {
    "mixtral-8x7b": "mistralai/Mixtral-8x7B-Instruct-v0.1",  # [Recommended]
    "nous-mixtral-8x7b": "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.2",
    "openchat-3.5": "openchat/openchat-3.5-0106",
    "gemma-7b": "google/gemma-7b-it",
    "default": "mistralai/Mixtral-8x7B-Instruct-v0.1",
}


STOP_SEQUENCES_MAP = {
    "mixtral-8x7b": "</s>",
    "nous-mixtral-8x7b": "<|im_end|>",
    "mistral-7b": "</s>",
    "openchat-3.5": "<|end_of_turn|>",
    "gemma-7b": "<eos>",
}

TOKEN_LIMIT_MAP = {
    "mixtral-8x7b": 32768,
    "nous-mixtral-8x7b": 32768,
    "mistral-7b": 32768,
    "openchat-3.5": 8192,
    "gemma-7b": 8192,
}

TOKEN_RESERVED = 20
