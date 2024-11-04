from tokenmix2.misc import get_model_and_tokenizer, get_env_conf
from profiler import WallTime
import argparse
import json


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_conf", type=str, default=None)
    parser.add_argument("--prompt_length", type=str, default=None)
    args = parser.parse_args()

    env_conf = get_env_conf(args.env_conf)
    tokenizer, model = get_model_and_tokenizer(**env_conf['model'])

    for prompt_length in json.loads(args.prompt_length):

        input_ids = [42] * prompt_length
        walltime = WallTime(f"{args.env_conf}/prompt-{prompt_length}", cuda=0)

        for _ in range(10):
            output = model.generate(input_ids, max_new_tokens=128, eos_token_id=[], prof=walltime)

        walltime.result(detail=True)
