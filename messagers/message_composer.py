import re
from pprint import pprint
from utils.logger import logger
from transformers import AutoTokenizer


class MessageComposer:
    # LINK - apis/chat_api.py#available-models
    AVALAIBLE_MODELS = [
        "mixtral-8x7b",
        "mistral-7b",
        "nous-mixtral-8x7b",
        "openchat-3.5",
        "gemma-7b",
    ]

    def __init__(self, model: str = None):
        if model in self.AVALAIBLE_MODELS:
            self.model = model
        else:
            self.model = "mixtral-8x7b"
        self.system_roles = ["system"]
        self.inst_roles = ["user", "system", "inst"]
        self.answer_roles = ["assistant", "bot", "answer", "model"]
        self.default_role = "user"

    def concat_messages_by_role(self, messages):
        def is_same_role(role1, role2):
            if (
                (role1 == role2)
                or (role1 in self.inst_roles and role2 in self.inst_roles)
                or (role1 in self.answer_roles and role2 in self.answer_roles)
            ):
                return True
            else:
                return False

        concat_messages = []
        for message in messages:
            role = message["role"]
            content = message["content"]
            if concat_messages and is_same_role(role, concat_messages[-1]["role"]):
                concat_messages[-1]["content"] += "\n" + content
            else:
                if role in self.inst_roles:
                    message["role"] = "inst"
                elif role in self.answer_roles:
                    message["role"] = "answer"
                else:
                    message["role"] = "inst"
                concat_messages.append(message)
        return concat_messages

    def merge(self, messages) -> str:
        # Mistral and Mixtral:
        #   <s> [INST] Instruction [/INST] Model answer </s> [INST] Follow-up instruction [/INST]

        # OpenChat:
        #   GPT4 Correct User: Hello<|end_of_turn|>GPT4 Correct Assistant: Hi<|end_of_turn|>GPT4 Correct User: How are you today?<|end_of_turn|>GPT4 Correct Assistant:

        # Nous Mixtral:
        #   <|im_start|>system
        #   You are "Hermes 2".<|im_end|>
        #   <|im_start|>user
        #   Hello, who are you?<|im_end|>
        #   <|im_start|>assistant

        # Google Gemma-it
        # <start_of_turn>user
        # How does the brain work?<end_of_turn>
        # <start_of_turn>model

        self.messages = messages
        self.merged_str = ""

        # https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1#instruction-format
        if self.model in ["mixtral-8x7b", "mistral-7b"]:
            self.messages = self.concat_messages_by_role(messages)
            self.cached_str = ""
            for message in self.messages:
                role = message["role"]
                content = message["content"]
                if role in self.inst_roles:
                    self.cached_str = f"[INST] {content} [/INST]"
                elif role in self.answer_roles:
                    self.merged_str += f"<s> {self.cached_str} {content} </s>\n"
                    self.cached_str = ""
                else:
                    self.cached_str = f"[INST] {content} [/INST]"
            if self.cached_str:
                self.merged_str += f"{self.cached_str}"
        # https://huggingface.co/NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO#prompt-format
        elif self.model in ["nous-mixtral-8x7b"]:
            self.merged_str_list = []
            for message in self.messages:
                role = message["role"]
                content = message["content"]
                if role not in ["system", "user", "assistant"]:
                    role = self.default_role
                message_line = f"<|im_start|>{role}\n{content}<|im_end|>"
                self.merged_str_list.append(message_line)
            self.merged_str_list.append("<|im_start|>assistant")
            self.merged_str = "\n".join(self.merged_str_list)
        # https://huggingface.co/openchat/openchat-3.5-0106
        elif self.model in ["openchat-3.5"]:
            tokenizer = AutoTokenizer.from_pretrained("openchat/openchat-3.5-0106")
            self.merged_str = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            # self.messages = self.concat_messages_by_role(messages)
            # self.merged_str_list = []
            # self.end_of_turn = "<|end_of_turn|>"
            # for message in self.messages:
            #     role = message["role"]
            #     content = message["content"]
            #     if role in self.inst_roles:
            #         self.merged_str_list.append(
            #             f"GPT4 Correct User:\n{content}{self.end_of_turn}"
            #         )
            #     elif role in self.answer_roles:
            #         self.merged_str_list.append(
            #             f"GPT4 Correct Assistant:\n{content}{self.end_of_turn}"
            #         )
            #     else:
            #         self.merged_str_list.append(
            #             f"GPT4 Correct User: {content}{self.end_of_turn}"
            #         )
            # self.merged_str_list.append(f"GPT4 Correct Assistant:\n")
            # self.merged_str = "\n".join(self.merged_str_list)
        # https://huggingface.co/google/gemma-7b-it#chat-template
        elif self.model in ["gemma-7b"]:
            self.messages = self.concat_messages_by_role(messages)
            self.merged_str_list = []
            self.end_of_turn = "<end_of_turn>"
            self.start_of_turn = "<start_of_turn>"
            for message in self.messages:
                role = message["role"]
                content = message["content"]
                if role in self.inst_roles:
                    self.merged_str_list.append(
                        f"{self.start_of_turn}user\n{content}{self.end_of_turn}"
                    )
                elif role in self.answer_roles:
                    self.merged_str_list.append(
                        f"{self.start_of_turn}model\n{content}{self.end_of_turn}"
                    )
                else:
                    self.merged_str_list.append(
                        f"{self.start_of_turn}user\n{content}{self.end_of_turn}"
                    )
            self.merged_str_list.append(f"{self.start_of_turn}model\n")
            self.merged_str = "\n".join(self.merged_str_list)
        else:
            self.merged_str = "\n".join(
                [
                    f'`{message["role"]}`:\n{message["content"]}\n'
                    for message in self.messages
                ]
            )

        return self.merged_str

    def convert_pair_matches_to_messages(self, pair_matches_list):
        messages = []
        if len(pair_matches_list) <= 0:
            messages = [
                {
                    "role": "user",
                    "content": self.merged_str,
                }
            ]
        else:
            for match in pair_matches_list:
                inst = match.group("inst")
                answer = match.group("answer")
                messages.extend(
                    [
                        {"role": "user", "content": inst.strip()},
                        {"role": "assistant", "content": answer.strip()},
                    ]
                )
        return messages

    def append_last_instruction_to_messages(self, inst_matches_list, pair_matches_list):
        if len(inst_matches_list) > len(pair_matches_list):
            self.messages.extend(
                [
                    {
                        "role": "user",
                        "content": inst_matches_list[-1].group("inst").strip(),
                    }
                ]
            )

    def split(self, merged_str) -> list:
        self.merged_str = merged_str
        self.messages = []

        if self.model in ["mixtral-8x7b", "mistral-7b"]:
            pair_pattern = (
                r"<s>\s*\[INST\](?P<inst>[\s\S]*?)\[/INST\](?P<answer>[\s\S]*?)</s>"
            )
            pair_matches = re.finditer(pair_pattern, self.merged_str, re.MULTILINE)
            pair_matches_list = list(pair_matches)

            self.messages = self.convert_pair_matches_to_messages(pair_matches_list)

            inst_pattern = r"\[INST\](?P<inst>[\s\S]*?)\[/INST\]"
            inst_matches = re.finditer(inst_pattern, self.merged_str, re.MULTILINE)
            inst_matches_list = list(inst_matches)

            self.append_last_instruction_to_messages(
                inst_matches_list, pair_matches_list
            )
        elif self.model in ["nous-mixtral-8x7b"]:
            # https://huggingface.co/NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO#prompt-format
            # message_pattern = r"<\|im_start\|>(?P<role>system|user|assistant)[\s\n]*(?P<content>[\s\S]*?)<\|im_end\|>"
            message_pattern = r"<\|im_start\|>(?P<role>system|user|assistant)[\s\n]*(?P<content>[\s\S]*?)<\|im_end\|>"
            message_matches = re.finditer(
                message_pattern, self.merged_str, flags=re.MULTILINE | re.IGNORECASE
            )
            message_matches_list = list(message_matches)
            logger.note(f"message_matches_list: {message_matches_list}")
            for match in message_matches_list:
                role = match.group("role")
                content = match.group("content")
                self.messages.append({"role": role, "content": content.strip()})
        elif self.model in ["openchat-3.5"]:
            pair_pattern = r"GPT4 Correct User:(?P<inst>[\s\S]*?)<\|end_of_turn\|>\s*GPT4 Correct Assistant:(?P<answer>[\s\S]*?)<\|end_of_turn\|>"
            pair_matches = re.finditer(
                pair_pattern, self.merged_str, flags=re.MULTILINE | re.IGNORECASE
            )
            pair_matches_list = list(pair_matches)
            self.messages = self.convert_pair_matches_to_messages(pair_matches_list)
            inst_pattern = r"GPT4 Correct User:(?P<inst>[\s\S]*?)<\|end_of_turn\|>"
            inst_matches = re.finditer(
                inst_pattern, self.merged_str, flags=re.MULTILINE | re.IGNORECASE
            )
            inst_matches_list = list(inst_matches)
            self.append_last_instruction_to_messages(
                inst_matches_list, pair_matches_list
            )
        # https://huggingface.co/google/gemma-7b-it#chat-template
        elif self.model in ["gemma-7b"]:
            pair_pattern = r"<start_of_turn>user[\s\n]*(?P<inst>[\s\S]*?)<end_of_turn>[\s\n]*<start_of_turn>model(?P<answer>[\s\S]*?)<end_of_turn>"
            pair_matches = re.finditer(
                pair_pattern, self.merged_str, flags=re.MULTILINE | re.IGNORECASE
            )
            pair_matches_list = list(pair_matches)
            self.messages = self.convert_pair_matches_to_messages(pair_matches_list)
            inst_pattern = r"<start_of_turn>user\n(?P<inst>[\s\S]*?)<end_of_turn>"
            inst_matches = re.finditer(
                inst_pattern, self.merged_str, flags=re.MULTILINE | re.IGNORECASE
            )
            inst_matches_list = list(inst_matches)
            self.append_last_instruction_to_messages(
                inst_matches_list, pair_matches_list
            )
        else:
            self.messages = [
                {
                    "role": "user",
                    "content": self.merged_str,
                }
            ]

        return self.messages


if __name__ == "__main__":
    # model = "mixtral-8x7b"
    # model = "nous-mixtral-8x7b"
    # model = "gemma-7b"
    model = "openchat-3.5"
    composer = MessageComposer(model)
    messages = [
        {
            "role": "system",
            "content": "You are a LLM developed by OpenAI.\nYour name is GPT-4.",
        },
        {"role": "user", "content": "Hello, who are you?"},
        {"role": "assistant", "content": "I am a bot."},
        {"role": "user", "content": "What is your name?"},
        # {"role": "assistant", "content": "My name is Bing."},
        # {"role": "user", "content": "Tell me a joke."},
        # {"role": "assistant", "content": "What is a robot's favorite type of music?"},
        # {
        #     "role": "user",
        #     "content": "How many questions have I asked? Please list them.",
        # },
    ]
    logger.note(f"model: {composer.model}")
    merged_str = composer.merge(messages)
    logger.note("merged_str:")
    logger.mesg(merged_str)
    logger.note("splitted messages:")
    pprint(composer.split(merged_str))
    # logger.note("merged merged_str:")
    # logger.mesg(composer.merge(composer.split(merged_str)))

    # python -m messagers.message_composer
