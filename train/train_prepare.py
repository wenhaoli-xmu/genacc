"""
intro
-----
* v4版本使用ranknet的pairwise ranking loss来学习排序
"""

from tokenmix2.misc import get_model_and_tokenizer, get_env_conf, Saver, Evaluator, get_optimizer_and_lr_adjuster

import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader, ConcatDataset
import torch
from corpus import LazyRandomSampleCorpus, get_processor


def build_dataset(env_conf, tokenizer):
    sum_partition = 0

    num_iters = env_conf["train"]["train_iters"]
    corpus = []
    for info in env_conf["train"]["corpus"]:
        sum_partition += info["partition"]
        num_instance = int(info["partition"] * num_iters)

        proc = get_processor(info["conf"], tokenizer)
        corp = LazyRandomSampleCorpus(info["data"], proc, max_instance=num_instance, use_cache=False)
        corpus.append(corp)

    assert sum_partition == 1
    return ConcatDataset(corpus)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_conf', type=str, default=None)
    args = parser.parse_args()


    env_conf = get_env_conf(args.env_conf)
    tokenizer, model = get_model_and_tokenizer(**env_conf['model'])
    dataset = build_dataset(env_conf, tokenizer)

    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    model.eval()

    inputs_record = []
    counter = 0

    for step, batch in tqdm(enumerate(loader), disable=False):

        batch.update({"return_inputs": True})

        with torch.no_grad():
            outputs = model(**batch)

        inputs = torch.cat([x.cpu() for x in outputs.hidden_states], dim=0)
        inputs_record.append(inputs)

        if len(inputs_record) >= 10:
            torch.save(inputs_record, f"train_cache/{counter:0>3}.pth")
            inputs_record = []
            counter += 1
