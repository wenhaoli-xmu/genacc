"""
intro
-----
* v4版本使用ranknet的pairwise ranking loss来学习排序
"""

from tokenmix2.misc import get_model_and_tokenizer, get_env_conf, adjust_lr

import argparse
from tqdm import tqdm
import torch
import os
from functools import partial

import deepspeed
import time, gc


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


def get_optimizer_and_lr_adjuster(model, max_lr, train_iters, warmup, weight_decay, beta1, beta2, params, **kwargs):
    optim = torch.optim.AdamW(params, lr=max_lr, betas=[beta1, beta2], weight_decay=weight_decay)
    lr_adjuster = partial(adjust_lr, optim=optim, total=train_iters, max_lr=max_lr, min_lr=0, restart=1, warmup=warmup, plateau=0)
    return optim, lr_adjuster


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_conf', type=str, default=None)
    parser.add_argument('--max_top', type=int, default=None)
    parser.add_argument('--max_oth', type=int, default=None)
    parser.add_argument('--maskout', type=float, default=0.98)
    parser.add_argument('--layer', type=int, default=None)
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    env_conf = get_env_conf(args.env_conf)
    tokenizer, model = get_model_and_tokenizer(**env_conf['model'])

    params = model.layer_ft_params(args.layer)
    optim, lr_adjust = get_optimizer_and_lr_adjuster(
        model, 
        **env_conf['train'], 
        params=params)
    
    model.train()
    model.freeze_model()
    model.unfreeze_model()

    layer = model.dump_as_attn_modules()[args.layer]
    assert layer.is_fix_layer is False, f"{args.layer} is fix layer, and dose not requrie post training."
    
    del model

    while torch.cuda.memory_reserved() / 1024 ** 2 > 1000:
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()
        time.sleep(0.1)
        print("model was deleted, waiting for GPU memory release")

    device = next(layer.parameters()).device
    files = sorted(os.listdir("train_cache"))
    accum_grad = env_conf["train"]["accum_grad"]
    clip_grad = env_conf["train"]["clip_grad"]
    step = 0
    
    history_loss = []
    history_diff = []
    epochs = env_conf['train']['train_iters'] // 1000


    for _ in range(epochs):
        for file in tqdm(files):
            path = os.path.join("train_cache", file)
            inputs = torch.load(path)

            for hidden_states in inputs:
                hidden_states = hidden_states[args.layer: args.layer + 1, ...]

                lr_adjust(step=step)

                # forward & backward
                _, _, draft_attn, true_attn = layer(hidden_states=hidden_states.to(device))
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

                torch.cuda.empty_cache()
                step += 1

            print(f"layer: {args.layer}\tstep: {step}\tloss: {sum(history_loss) / len(history_loss):<.3f}\tdiff: {sum(history_diff) / len(history_diff):<.3f}", flush=True)
            history_loss = []
            history_diff = []

    # overall save
    save_path = args.env_conf.split('/')[-1]
    if not os.path.exists(f"train_results/{save_path}"):
        os.mkdir(f"train_results/{save_path}")
    torch.save(params, f"train_results/{save_path}/{args.layer}.pth")
