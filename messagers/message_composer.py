import re
from pprint import pprint


class MessageComposer:
    """
    models:
    - mixtral-8x7b (mistralai/Mixtral-8x7B-Instruct-v0.1)
    """

    def __init__(self, model: str = None):
        self.inst_roles = ["user", "system", "inst"]
        self.answer_roles = ["assistant", "bot", "answer"]

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
        # <s> [INST] Instruction [/INST] Model answer </s> [INST] Follow-up instruction [/INST]

        self.messages = self.concat_messages_by_role(messages)
        self.merged_str = ""
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

        return self.merged_str

    def split(self, merged_str) -> list:
        self.messages = []
        self.merged_str = merged_str
        pair_pattern = (
            r"<s>\s*\[INST\](?P<inst>[\s\S]*?)\[/INST\](?P<answer>[\s\S]*?)</s>"
        )
        pair_matches = re.finditer(pair_pattern, self.merged_str, re.MULTILINE)
        pair_matches_list = list(pair_matches)

        if len(pair_matches_list) <= 0:
            self.messages = [
                {
                    "role": "user",
                    "content": self.merged_str,
                }
            ]
        else:
            for match in pair_matches_list:
                inst = match.group("inst")
                answer = match.group("answer")
                self.messages.extend(
                    [
                        {"role": "user", "content": inst.strip()},
                        {"role": "assistant", "content": answer.strip()},
                    ]
                )

        inst_pattern = r"\[INST\](?P<inst>[\s\S]*?)\[/INST\]"
        inst_matches = re.finditer(inst_pattern, self.merged_str, re.MULTILINE)
        inst_matches_list = list(inst_matches)

        if len(inst_matches_list) > len(pair_matches_list):
            self.messages.extend(
                [
                    {
                        "role": "user",
                        "content": inst_matches_list[-1].group("inst").strip(),
                    }
                ]
            )

        return self.messages


if __name__ == "__main__":
    composer = MessageComposer()
    messages = [
        {
            "role": "system",
            "content": "You are a LLM developed by OpenAI. Your name is GPT-4.",
        },
        {"role": "user", "content": "Hello, who are you?"},
        {"role": "assistant", "content": "I am a bot."},
        # {"role": "user", "content": "What is your name?"},
        {"role": "assistant", "content": "My name is Bing."},
        # {"role": "user", "content": "Tell me a joke."},
        # {"role": "assistant", "content": "What is a robot's favorite type of music?"},
        # {
        #     "role": "user",
        #     "content": "How many questions have I asked? Please list them.",
        # },
    ]
    merged_str = composer.merge(messages)
    print(merged_str)
    pprint(composer.split(merged_str))
    # print(composer.merge(composer.split(merged_str)))
