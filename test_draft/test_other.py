from tokenmix2.misc import get_model_and_tokenizer
from tokenmix2.misc import get_env_conf
from tokenmix2.misc import Evaluator
import argparse, os

from profiler import WallTime


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_conf", type=str, default=None)
    parser.add_argument("--use_env_conf_tasks", action="store_true", default=False)
    args = parser.parse_args()

    # WallTime("sdpa", cuda=[0,1,2,3])
    # WallTime("draft_score", cuda=[0,1,2,3])

    env_conf = get_env_conf(args.env_conf)
    test_conf = get_env_conf("test_draft/eval.json")

    tokenizer, model = get_model_and_tokenizer(**env_conf["model"])
    model.eval()

    def get_color(value):
        from pygments.console import colorize
        if value > 75:
            return colorize("green", f"{value: 2d}")
        elif value > 50:
            return colorize("yellow", f"{value: 2d}")
        else:
            return colorize("red", f"{value: 2d}")

    def callback(outputs):
        if not model.model.decoder.is_benchmark_mode():
            return

        ratios = model.model.decoder.get_ratios(reset=True)
        num_heads = 32

        print("      ", end='')
        for head_id in range(num_heads):
            print(f'#{head_id:<2}', end=' ')
        print(f"avg", end=None)

        mean_ratios = [[] for _ in range(num_heads + 1)]

        for idx, layer_ratio in enumerate(ratios):
            if layer_ratio is not None:
                print(f"{idx: >3}", end=': ')
                for hid, head_ratio in enumerate(layer_ratio):
                    value = int(head_ratio * 100)
                    print(get_color(value), end=' ')
                    mean_ratios[hid].append(value)
                value = int(sum(layer_ratio) / len(layer_ratio) * 100)
                print(get_color(value))
                mean_ratios[-1].append(value)
        
        print(f"     ", end='')
        for head_ratio in mean_ratios:
            head_ratio = sum(head_ratio) // len(head_ratio)
            print(get_color(head_ratio), end=' ')


    ckp_file = env_conf['model']['save_ckp']
    if os.path.exists(ckp_file):
        print(f"load checkpoint {ckp_file}")
        model.load_checkpoint(ckp_file)
    else:
        print(f"{ckp_file} dose not exists")

    evaluator_class = Evaluator

    if args.use_env_conf_tasks:
        evaluator = evaluator_class(model, tokenizer, **env_conf["train"], callback=callback)
    else:
        evaluator = evaluator_class(model, tokenizer, eval=None, tasks=test_conf, callback=callback)
    
    evaluator.evaluate()

    # for value in WallTime.regist.values():
    #     value.result(detail=True)
