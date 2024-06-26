MODEL_MAP = {
    "mixtral-8x7b": "mistralai/Mixtral-8x7B-Instruct-v0.1",  # [Recommended]
    "nous-mixtral-8x7b": "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.2",
    "yi-1.5-34b": "01-ai/Yi-1.5-34B-Chat",
    "gemma-7b": "google/gemma-1.1-7b-it",
    # "openchat-3.5": "openchat/openchat-3.5-0106",
    # "command-r-plus": "CohereForAI/c4ai-command-r-plus",
    # "llama3-70b": "meta-llama/Meta-Llama-3-70B-Instruct",
    # "zephyr-141b": "HuggingFaceH4/zephyr-orpo-141b-A35b-v0.1",
    "default": "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
}

AVAILABLE_MODELS = list(MODEL_MAP.keys())

PRO_MODELS = ["command-r-plus", "llama3-70b", "zephyr-141b"]

STOP_SEQUENCES_MAP = {
    # https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1/blob/main/tokenizer_config.json#L33
    "mixtral-8x7b": "</s>",
    # https://huggingface.co/NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO/blob/main/tokenizer_config.json#L50
    "nous-mixtral-8x7b": "<|im_end|>",
    # https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2/blob/main/tokenizer_config.json#L33
    "mistral-7b": "</s>",
    # https://huggingface.co/01-ai/Yi-1.5-34B-Chat/blob/main/tokenizer_config.json#L42
    "yi-1.5-34b": "<|im_end|>",
    # https://huggingface.co/google/gemma-1.1-7b-it/blob/main/tokenizer_config.json#L1509
    "gemma-7b": "<eos>",
    # "openchat-3.5": "<|end_of_turn|>",
    # "command-r-plus": "<|END_OF_TURN_TOKEN|>",
}

TOKEN_LIMIT_MAP = {
    "mixtral-8x7b": 32768,
    "nous-mixtral-8x7b": 32768,
    "mistral-7b": 32768,
    "yi-1.5-34b": 4096,
    "gemma-7b": 8192,
    # "openchat-3.5": 8192,
    # "command-r-plus": 32768,
    # "llama3-70b": 8192,
    # "zephyr-141b": 2048,
    # "gpt-3.5-turbo": 8192,
}

TOKEN_RESERVED = 20


# https://platform.openai.com/docs/api-reference/models/list
AVAILABLE_MODELS_DICTS = [
    {
        "id": "mixtral-8x7b",
        "description": "[mistralai/Mixtral-8x7B-Instruct-v0.1]: https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1",
        "object": "model",
        "created": 1700000000,
        "owned_by": "mistralai",
    },
    {
        "id": "nous-mixtral-8x7b",
        "description": "[NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO]: https://huggingface.co/NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
        "object": "model",
        "created": 1700000000,
        "owned_by": "NousResearch",
    },
    {
        "id": "mistral-7b",
        "description": "[mistralai/Mistral-7B-Instruct-v0.2]: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2",
        "object": "model",
        "created": 1700000000,
        "owned_by": "mistralai",
    },
    {
        "id": "yi-1.5-34b",
        "description": "[01-ai/Yi-1.5-34B-Chat]: https://huggingface.co/01-ai/Yi-1.5-34B-Chat",
        "object": "model",
        "created": 1700000000,
        "owned_by": "01-ai",
    },
    {
        "id": "gemma-7b",
        "description": "[google/gemma-1.1-7b-it]: https://huggingface.co/google/gemma-1.1-7b-it",
        "object": "model",
        "created": 1700000000,
        "owned_by": "Google",
    },
    # {
    #     "id": "openchat-3.5",
    #     "description": "[openchat/openchat-3.5-0106]: https://huggingface.co/openchat/openchat-3.5-0106",
    #     "object": "model",
    #     "created": 1700000000,
    #     "owned_by": "openchat",
    # },
    # {
    #     "id": "command-r-plus",
    #     "description": "[CohereForAI/c4ai-command-r-plus]: https://huggingface.co/CohereForAI/c4ai-command-r-plus",
    #     "object": "model",
    #     "created": 1700000000,
    #     "owned_by": "CohereForAI",
    # },
    # {
    #     "id": "llama3-70b",
    #     "description": "[meta-llama/Meta-Llama-3-70B]: https://huggingface.co/meta-llama/Meta-Llama-3-70B",
    #     "object": "model",
    #     "created": 1700000000,
    #     "owned_by": "Meta",
    # },
    # {
    #     "id": "zephyr-141b",
    #     "description": "[HuggingFaceH4/zephyr-orpo-141b-A35b-v0.1]: https://huggingface.co/HuggingFaceH4/zephyr-orpo-141b-A35b-v0.1",
    #     "object": "model",
    #     "created": 1700000000,
    #     "owned_by": "Huggingface",
    # },
    # {
    #     "id": "gpt-3.5-turbo",
    #     "description": "[openai/gpt-3.5-turbo]: https://platform.openai.com/docs/models/gpt-3-5-turbo",
    #     "object": "model",
    #     "created": 1700000000,
    #     "owned_by": "OpenAI",
    # },
]
