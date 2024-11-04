from tokenmix2.misc import get_model_and_tokenizer, get_env_conf
from profiler import WallTime
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_conf", type=str, default=None)
    parser.add_argument("--hist", action='store_true')
    args = parser.parse_args()

    env_conf = get_env_conf(args.env_conf)

    tokenizer, model = get_model_and_tokenizer(**env_conf['model'])

    prompt = """
    Write an essay of 1,000 words to describe China.
    """

    input_ids = tokenizer(prompt)
    walltime = WallTime("128t decoding", cuda=0)

    for _ in range(10):
        output = model.generate(input_ids, max_new_tokens=128, prof=walltime)

    walltime.result("repeat-10")

    if args.hist:
        walltime.hist("repeat-10")
