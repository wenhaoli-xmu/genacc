# import subprocess
import multiprocessing
import torch
from tokenmix2.misc import get_model_and_tokenizer, get_env_conf, adjust_lr, get_torch_dtype
import gc, time
import os, sys
import math

import deepspeed
from corpus import LazyRandomSampleCorpus, get_processor
from torch.utils.data import DataLoader, ConcatDataset
import json
from functools import partial
from torch.utils.checkpoint import checkpoint
from multiprocessing import shared_memory
import numpy as np


def compute_attn_supervise_loss(draft_attn, true_attn, max_top, max_oth, maskout):
    loss = torch.tensor(0, dtype=torch.float32)
    criterion = torch.nn.BCEWithLogitsLoss()

    # 计算出true attn的sort index
    mask = torch.triu(torch.ones(true_attn.shape[-2:], dtype=torch.bool, device=true_attn.device), diagonal=1)[None, None, :, :]
    true_attn = torch.masked_fill(true_attn, mask, value=torch.finfo(true_attn.dtype).min)
    indices = torch.argsort(true_attn, dim=-1, descending=True)

    # 切分出来top 0.01 的indices，和other 0.98的indices
    top_cnt = int(indices.shape[-1] * (1 - maskout))
    top_indices = indices[..., :top_cnt]
    oth_indices = indices[..., top_cnt:]

    if max_top is not None:
        top_rnd_indices = torch.randperm(top_cnt, dtype=torch.int64, device=indices.device)[:max_top]
        top_indices = top_indices[..., top_rnd_indices]
    if max_oth is not None:
        oth_rnd_indices = torch.randperm(indices.shape[-1] - top_cnt, dtype=torch.int64, device=indices.device)[:max_oth]
        oth_indices = oth_indices[..., oth_rnd_indices]

    top_mask = torch.gather(mask.expand_as(true_attn), dim=-1, index=top_indices)[..., :, None]
    oth_mask = torch.gather(mask.expand_as(true_attn), dim=-1, index=oth_indices)[..., None, :]

    top_draft_attn = torch.gather(draft_attn, dim=-1, index=top_indices)[..., :, None]
    oth_draft_attn = torch.gather(draft_attn, dim=-1, index=oth_indices)[..., None, :]

    residual = top_draft_attn - oth_draft_attn
    residual_mask = (top_mask | oth_mask).expand_as(residual).flatten(-3)

    logits = residual.flatten(-3)[~residual_mask.bool()]
    labels = torch.ones_like(logits, dtype=torch.float32)
    loss += criterion(logits, labels.type(torch.float32)).cpu()

    # 算一下排序误差
    diff = torch.count_nonzero(logits < 0) / logits.numel()

    return diff, loss


def build_dataset(env_conf, tokenizer):
    sum_partition = 0

    num_iters = env_conf['train']['train_iters']
    corpus = []
    for info in env_conf['train']['corpus']:
        sum_partition += info['partition']
        num_instance = int(info['partition'] * num_iters)

        proc = get_processor(info['conf'], tokenizer)
        corp = LazyRandomSampleCorpus(info['data'], proc, max_instance=num_instance, use_cache=False)
        corpus.append(corp)

    assert sum_partition == 1
    return ConcatDataset(corpus)


def get_optimizer_and_lr_adjuster(max_lr, train_iters, warmup, weight_decay, beta1, beta2, params, **kwargs):
    optim = torch.optim.AdamW(params, lr=max_lr, betas=[beta1, beta2], weight_decay=weight_decay)
    lr_adjuster = partial(adjust_lr, optim=optim, total=train_iters, max_lr=max_lr, min_lr=0, restart=1, warmup=warmup, plateau=0)
    return optim, lr_adjuster


def clear_cache(local_rank, max_trial=10):
    torch.cuda.set_device(local_rank)
    while max_trial > 0:
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()
        time.sleep(0.1)
        max_trial -= 1


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()

    # 和模型结构有关的参数，需要根据模型的不同而相应地调整
    parser.add_argument("--num_layers", type=int, default=32)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--hidden_size", type=int, default=4096)
    parser.add_argument("--maskout", type=float, default=0.98)

    # 和模型无关的参数
    parser.add_argument("--env_conf", type=str, default=None)
    parser.add_argument("--local_rank", type=int, default=-1)

    # 和资源开销有关的参数
    parser.add_argument("--instance_per_cycle", type=int, default=1000)
    parser.add_argument("--max_top", type=int, default=None)
    parser.add_argument("--max_oth", type=int, default=1024)
    
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    deepspeed.init_distributed()


    # 计算一些变量 & 例行检查
    env_conf = get_env_conf(args.env_conf)
    num_gpus = torch.distributed.get_world_size()

    assert env_conf['train']['train_iters'] % args.instance_per_cycle == 0
    assert args.num_layers % num_gpus == 0
    assert env_conf['model']['model_dtype'] in ('bf16', 'fp16')

    num_inn_cycle = env_conf['train']['train_iters'] // args.instance_per_cycle
    num_out_cycle = args.num_layers // num_gpus
    inputs_shape = [num_gpus, args.instance_per_cycle, args.max_tokens, args.hidden_size]
    

    # 创建共享内存
    if args.local_rank == 0:
        shm = shared_memory.SharedMemory(
            name='inputs_record', 
            create=True, 
            size=np.prod(inputs_shape) * 2)
        shm2 = shared_memory.SharedMemory(
            name='inputs_length',
            create=True,
            size=args.instance_per_cycle * 8
        )
        print(f"RANK0 shared memory size: {np.prod(inputs_shape) * 2 // 1024 ** 3} GB")
        
    torch.distributed.barrier()

    if args.local_rank != 0:
        shm = shared_memory.SharedMemory(name="inputs_record")
        shm2 = shared_memory.SharedMemory(name="inputs_length")
        
    inputs_record = torch.frombuffer(
        shm.buf, 
        dtype=get_torch_dtype(env_conf['model']['model_dtype']),
        ).reshape(*inputs_shape)

    inputs_length = torch.frombuffer(
        shm2.buf,
        dtype=torch.int64,
    ).reshape(args.instance_per_cycle)

    torch.distributed.barrier()


    # 开始训练pipeline
    for out_cycle_idx in range(num_out_cycle):
        torch.manual_seed(42)

        # 加载模型 & tokenizer
        layer_idx = out_cycle_idx * num_gpus + args.local_rank
        layer_indices = [out_cycle_idx * num_gpus + i for i in range(num_gpus)]
        env_conf["model"]["device_map"] = {"": args.local_rank}
        tokenizer, model = get_model_and_tokenizer(**env_conf['model'])

        # 将模型只保存某个layer
        model.train()
        model.freeze_model()
        model.unfreeze_model()
        layer = model.dump_as_attn_modules()[layer_idx]
        params = model.layer_ft_params(layer_idx)
        del model
        clear_cache(args.local_rank)
        print(f"RANK{args.local_rank} training started !")

        # 构造数据集
        if args.local_rank == 0:
            corpus = build_dataset(env_conf, tokenizer)
            loader = DataLoader(corpus, batch_size=1, shuffle=True)
            data_iter = iter(loader)
        torch.distributed.barrier()
        
        # 构造优化器 & 学习率调节器
        optim, lr_adjust = get_optimizer_and_lr_adjuster(**env_conf['train'], params=params)
        
        # 一些参数
        accum_grad = env_conf["train"]["accum_grad"]
        clip_grad = env_conf["train"]["clip_grad"]
        history_loss = []
        history_diff = []
        step = 0

        for inn_cycle_idx in range(num_inn_cycle):

            if args.local_rank == 0:

                # 同时使用全部卡来准备训练数据
                env_conf["model"]["device_map"] = None
                _, model = get_model_and_tokenizer(**env_conf['model'])
                model.eval()

                # 准备数据
                for idx in range(args.instance_per_cycle):
                    inputs = next(data_iter)
                    inputs.update({"return_inputs": True})

                    with torch.no_grad():
                        #前向传播
                        outputs = model(**inputs)
                        inputs = [outputs.hidden_states[i].cpu()for i in layer_indices]
                        inputs = torch.stack(inputs, dim=0)

                        # 写入共享内存
                        inputs_record[:, idx: idx + 1, :inputs.shape[-2], :] = inputs
                        inputs_length[idx] = inputs.shape[-2]

                # 数据准备完毕之后直接删除模型
                del model, inputs, outputs
                clear_cache(args.local_rank)

            torch.distributed.barrier()

            for length, hidden_states in zip(inputs_length, inputs_record[args.local_rank]):

                hidden_states = hidden_states[:length, ...].unsqueeze(0)

                lr_adjust(step=step)

                # forward & backward
                _, _, draft_attn, true_attn = layer(hidden_states=hidden_states.to(args.local_rank))
                grad = torch.zeros_like(draft_attn)

                for head_idx, (draft_attn_head, true_attn_head) in enumerate(zip(
                    torch.chunk(draft_attn, chunks=draft_attn.shape[1], dim=1),
                    torch.chunk(true_attn, chunks=draft_attn.shape[1], dim=1),
                )):
                    draft_attn_head = draft_attn_head.detach()
                    true_attn_head = true_attn_head.detach()
                    draft_attn_head.requires_grad_(True)

                    diff, loss = compute_attn_supervise_loss(draft_attn_head, true_attn_head, args.max_top, args.max_oth, args.maskout)
                    loss.backward()

                    grad[:, head_idx, ...] = draft_attn_head.grad.data[:]

                    history_loss.append(loss.item())
                    history_diff.append(diff.item())

                grad /= accum_grad
                draft_attn.backward(gradient=grad)

                # update the parameters
                if (step + 1) % accum_grad == 0:
                    if clip_grad is not None:
                        torch.nn.utils.clip_grad_norm_(params, max_norm=clip_grad)
                    optim.step()
                    optim.zero_grad()

                step += 1

            print(f"layer: {layer_idx}\tstep: {step}\tloss: {sum(history_loss) / len(history_loss):<.3f}\tdiff: {sum(history_diff) / len(history_diff):<.3f}", flush=True)
            history_loss = []
            history_diff = []
            clear_cache(args.local_rank)

        # overall save
        save_path = args.env_conf.split('/')[-1]
        if not os.path.exists(f"train_results/{save_path}"):
            os.mkdir(f"train_results/{save_path}")
        torch.save(params, f"train_results/{save_path}/{layer_idx}.pth")
        print(f"RANK{args.local_rank} training done !")
        torch.distributed.barrier()

    if args.local_rank == 0:
        shm.unlink()
        shm2.unlink()
