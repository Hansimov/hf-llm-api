import json
import base64
import random
from datetime import datetime, timedelta, timezone
from Crypto.Hash import SHA3_512
from constants.headers import OPENAI_GET_HEADERS


class ProofWorker:
    def __init__(self, difficulty=None, required=False, seed=None):
        self.difficulty = difficulty
        self.required = required
        self.seed = seed
        self.proof_token_prefix = "gAAAAABwQ8Lk5FbGpA2NcR9dShT6gYjU7VxZ4D"

    def get_parse_time(self):
        now = datetime.now()
        tz = timezone(timedelta(hours=8))
        now = now.astimezone(tz)
        time_format = "%a %b %d %Y %H:%M:%S"
        return now.strftime(time_format) + " GMT+0800 (中国标准时间)"

    def get_config(self):
        cores = [8, 12, 16, 24]
        core = random.choice(cores)
        screens = [3000, 4000, 6000]
        screen = random.choice(screens)
        return [
            str(core) + str(screen),
            self.get_parse_time(),
            4294705152,
            0,
            OPENAI_GET_HEADERS["User-Agent"],
        ]

    def calc_proof_token(self, seed: str, difficulty: str):
        config = self.get_config()
        diff_len = len(difficulty) // 2
        for i in range(100000):
            config[3] = i
            json_str = json.dumps(config)
            base = base64.b64encode(json_str.encode()).decode()
            hasher = SHA3_512.new()
            hasher.update((seed + base).encode())
            hash = hasher.digest()
            if hash.hex()[:diff_len] <= difficulty:
                return "gAAAAAB" + base
        self.proof_token = (
            self.proof_token_prefix + base64.b64encode(seed.encode()).decode()
        )
        return self.proof_token


if __name__ == "__main__":
    seed, difficulty = "0.42665582693491433", "05cdf2"
    worker = ProofWorker()
    proof_token = worker.calc_proof_token(seed, difficulty)
    print(f"proof_token: {proof_token}")
    # python -m networks.proof_of_work
