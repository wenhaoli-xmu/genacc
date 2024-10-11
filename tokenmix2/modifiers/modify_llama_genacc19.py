import torch
import types
from .modify_llama import do_sdpa_attn, get_attn_score, check_and_apply_qk_rope
from transformers.models.llama.modeling_llama import CausalLMOutputWithPast, repeat_kv, CrossEntropyLoss
from torch.utils.checkpoint import checkpoint
from ..modifier import Modifier
from peft import get_peft_model, LoraConfig, TaskType

from typing import List, Tuple


def random_rotation_matrix(dim, dtype, device):
    """
    随机生成一个 n 维旋转矩阵
    :param dim: 维度大小 (n)
    :return: n x n 随机旋转矩阵
    """
    # 使用QR分解生成随机正交矩阵
    random_matrix = torch.randn((dim, dim), dtype=torch.float64)
    q, r = torch.linalg.qr(random_matrix)
    
    # 调整使其行列式为1
    if torch.det(q) < 0:
        q[:, 0] *= -1

    return q.type(dtype).to(device)


def model_forward(
    self,
    input_ids: torch.LongTensor,
    labels: torch.Tensor = None,
    kv_cache: List[Tuple[torch.Tensor, torch.Tensor]] = None,
    return_inputs: bool = False,
    **kwargs
):
    # model forward function
    hidden_states, kv_cache, draft_attn, true_attn, inputs_records = self.model(
        input_ids=input_ids,
        kv_cache=kv_cache,
        return_inputs=return_inputs)
    
    logits = self.lm_head(hidden_states).float()

    loss = None
    if labels is not None:
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)

        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

    return CausalLMOutputWithPast(
        loss=loss, 
        logits=logits, 
        past_key_values=kv_cache,
        attentions=(draft_attn, true_attn),
        hidden_states=inputs_records)


def model_model_forward(
    self,
    input_ids: torch.LongTensor = None,
    kv_cache: List[Tuple[torch.Tensor, torch.Tensor]] = None,
    return_inputs: bool = False
):
    inputs_embeds = self.embed_tokens(input_ids)
    hidden_states = inputs_embeds

    if kv_cache is None:
        kv_cache = [None] * len(self.layers)

    draft_attns = []
    true_attns = []
    inputs_records = []

    for layer_idx, (decoder_layer, kv_cache_layer) in enumerate(zip(self.layers, kv_cache)):

        if torch.is_grad_enabled():
            layer_output = checkpoint(
                decoder_layer,
                hidden_states,
                kv_cache_layer,
                return_inputs,
                use_reentrant=False)
        else:
            layer_output = decoder_layer(
                hidden_states, 
                kv_cache_layer,
                return_inputs)

        hidden_states, kv_cache_layer, draft_attn, true_attn, inputs_record = layer_output
        draft_attns.append(draft_attn)
        true_attns.append(true_attn)
        inputs_records.append(inputs_record)

        kv_cache[layer_idx] = kv_cache_layer

    hidden_states = self.norm(hidden_states)

    return hidden_states, kv_cache, draft_attns, true_attns, inputs_records

def layer_forward(
    self,
    hidden_states: torch.Tensor,
    kv_cache: Tuple[torch.Tensor, torch.Tensor] = None,
    return_inputs: bool = False,
):
    device = self.self_attn.q_proj.weight.data.device
    if hidden_states.device != device:
        hidden_states = hidden_states.to(device)

    # do the self attention mechanism
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)

    if return_inputs:
        inputs_record = hidden_states.data.cpu().clone()
    else:
        inputs_record = None

    hidden_states, kv_cache, draft_attn, true_attn = self.self_attn(
        hidden_states, 
        kv_cache)
    hidden_states = residual + hidden_states
    
    # do the feed forward
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    return hidden_states, kv_cache, draft_attn, true_attn, inputs_record


def get_attn_score_using_angle_lsh(query, key, rot_mat1, rot_mat2, relu, cos, sin, gamma=64):
    query, key = check_and_apply_qk_rope(query, key, cos, sin)

    q_hash = relu(query @ rot_mat1) @ rot_mat2
    k_hash = relu(key @ rot_mat1) @ rot_mat2

    q_hash *= gamma
    k_hash *= gamma

    q_hash = q_hash / (1 + q_hash.abs())
    k_hash = k_hash / (1 + k_hash.abs())

    sim = q_hash @ k_hash.transpose(-1,-2)
    return sim


def self_attn_forward(
    self,
    hidden_states: torch.Tensor,
    kv_cache: Tuple[torch.Tensor, torch.Tensor] = None,
):

    num_heads, embed_dim = self.config.num_attention_heads, self.config.hidden_size
    num_kv_heads = self.config.num_key_value_heads
    num_kv_group = num_heads // num_kv_heads
    head_dim = embed_dim // num_heads

    ques = self.q_proj(hidden_states).unflatten(-1, (num_heads, head_dim)).transpose(1,2)
    keys = self.k_proj(hidden_states).unflatten(-1, (num_kv_heads, head_dim)).transpose(1,2)
    vals = self.v_proj(hidden_states).unflatten(-1, (num_kv_heads, head_dim)).transpose(1,2)

    keys = repeat_kv(keys, num_kv_group)
    vals = repeat_kv(vals, num_kv_group)
    
    cos, sin = self.rotary_emb(vals, seq_len=4096)

    if not self.is_fix_layer:
        draft_score = get_attn_score_using_angle_lsh(
            query=ques, 
            key=keys, 
            rot_mat1=self.rot_mat1,
            rot_mat2=self.rot_mat2, 
            relu=self.relu1, 
            cos=cos, 
            sin=sin, 
            gamma=self.gamma)

        true_score = get_attn_score(query=ques, key=keys, cos=cos, sin=sin)
        ret_attn = (draft_score, true_score)
    else:
        ret_attn = (None, None)

    attn_output = do_sdpa_attn(
        query=ques,
        key=keys,
        value=vals,
        cos=cos,
        sin=sin,
        out_proj=self.o_proj)

    return attn_output, kv_cache, *ret_attn


class Decoder(torch.nn.Module):
    def _init_lora(
            self,
            lora_rank: int, 
            lora_alpha: int, 
            lora_dropout: float):

        target_modules = r".*\.(self_attn|mlp)\.(q|k|v|o|gate|up|down)_proj"
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules)
        self.decoder = get_peft_model(self.decoder, peft_config)


    @property
    def layers(self):
        if self.enable_lora:
            return self.decoder.base_model.model.model.layers
        else:
            return self.decoder.model.layers


    @property
    def model(self):
        if self.enable_lora:
            return self.decoder.base_model.model
        else:
            return self.decoder


    def reset(self):
        for layer in self.layers:
            if hasattr(layer.self_attn, 'k_cache'):
                del layer.self_attn.k_cache
                del layer.self_attn.v_cache


    def __init__(
            self, 
            decoder, 
            enable_lora: bool = False,
            lora_kwargs: dict = None,
            fix_layers: list = [],
            num_rnd_layers: int = 2,
            gamma: int = 64,
            draft_kwargs: dict = {"use_draft": False}):

        super().__init__()
        self.decoder = decoder
        self.enable_lora = False
        self.fix_layers = fix_layers
        self.num_rnd_layers = num_rnd_layers
        self.draft_kwargs = draft_kwargs

        # 修改各种forward函数
        self.model.forward = types.MethodType(model_forward, self.model)
        self.model.model.forward = types.MethodType(model_model_forward, self.model.model)

        for layer_idx, layer in enumerate(self.layers):

            info = {
                "device": layer.self_attn.q_proj.weight.data.device,
                "dtype": layer.self_attn.q_proj.weight.data.dtype}
            
            layer.self_attn.is_fix_layer = layer_idx in fix_layers
            layer.self_attn.gamma = gamma

            # modify the forward function
            layer.self_attn.draft_kwargs = draft_kwargs
            layer.forward = types.MethodType(layer_forward, layer)
            layer.self_attn.forward = types.MethodType(self_attn_forward, layer.self_attn)

            def get_rot_mat():
                rot_mats = []
                for _ in range(32):
                    rot_mats.append(random_rotation_matrix(dim=128, **info))
                return torch.stack(rot_mats, dim=0).unsqueeze(0)

            if not layer.self_attn.is_fix_layer:
                layer.self_attn.rot_mat1 = torch.nn.Parameter(get_rot_mat(), requires_grad=True)
                layer.self_attn.relu1 = torch.nn.SiLU()
                layer.self_attn.rot_mat2 = torch.nn.Parameter(get_rot_mat(), requires_grad=True)


    def is_benchmark_mode(self):
        return self.draft_kwargs['bench_mark']

    
    def enable_benchmark_mode(self):
        self.draft_kwargs['bench_mark'] = True

    
    def disable_benchmark_mode(self):
        self.draft_kwargs['bench_mark'] = False


    def get_ratios(self, reset=False):
        ratios = []
        for idx, layer in enumerate(self.layers):
            if idx in self.fix_layers:
                ratios.append(None)
            else:
                ratios.append(layer.self_attn.ratios)
                del layer.self_attn.ratios
        return ratios
    

    def layer_ft_params(self, layer):
        layer = self.layers[layer]
        if layer.self_attn.is_fix_layer:
            return []
        return [layer.self_attn.rot_mat1, layer.self_attn.rot_mat2]


    def ft_params(self, layer=None):
        params = []

        for layer in self.layers:
            if not layer.self_attn.is_fix_layer:
                params += [
                    layer.self_attn.rot_mat1,
                    layer.self_attn.rot_mat2,
                ]

        return params


    def forward(
            self, 
            input_ids, 
            labels=None,
            return_inputs=False):

        # decoder forward
        outputs = self.decoder(
            input_ids=input_ids, 
            labels=labels,
            return_inputs=return_inputs)

        return outputs


class Model(torch.nn.Module):
    def __init__(
            self, 
            decoder: Decoder
        ):
        super().__init__()
        self.decoder = decoder

    def ft_params(self):
        params = self.decoder.ft_params()
        return params


    def reset(self):
        self.decoder.reset()


    def forward(
            self,
            input_ids,
            labels=None,
            local_rank=None,
            return_inputs=False,
            **kwargs
        ):

        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids, dtype=torch.int64)[None, :]
            labels = torch.tensor(labels, dtype=torch.int64)[None, :]

        label_exist = labels is not None
        rank_exist = local_rank is not None

        if input_ids.ndim == 3:
            input_ids = input_ids.flatten(0,1)
        if label_exist and labels.ndim == 3:
            labels = labels.flatten(0,1)

        if rank_exist:
            device = torch.device(local_rank)
        else:
            device = next(iter(self.decoder.parameters())).device
        input_ids = input_ids.to(device)

        outputs = self.decoder(
            input_ids, 
            labels=labels,
            return_inputs=return_inputs)

        return outputs


class LlamaGenAcc19(Modifier):
    def __init__(self, model, save_ckp, load_ckp, config):
        self.get_conf(config)
        assert isinstance(self.conf, dict)
        enable_lora = self.conf["enable_lora"]
        lora_kwargs = self.conf["lora_kwargs"]

        draft_kwargs = self.conf['draft_kwargs']
        fix_layers = [] if "fix_layers" not in self.conf else self.conf["fix_layers"]
        num_rnd_layers = None if "num_rnd_layers" not in self.conf else self.conf['num_rnd_layers']
        gamma = 64 if "gamma" not in self.conf else self.conf["gamma"]
        
        decoder = Decoder(
            model, 
            enable_lora=enable_lora,
            lora_kwargs=lora_kwargs,
            fix_layers=fix_layers,
            num_rnd_layers=num_rnd_layers,
            gamma=gamma,
            draft_kwargs=draft_kwargs)

        decoder = Model(decoder)

        super().__init__(decoder, save_ckp, load_ckp)


    def ft_params(self):
        return self.model.ft_params()


    def reset(self):
        self.model.reset()


    def is_benchmark_mode(self):
        return self.model.decoder.is_benchmark_mode()

    
    def enable_benchmark_mode(self):
        return self.model.decoder.enable_benchmark_mode()

    
    def disable_benchmark_mode(self):
        return self.model.decoder.disable_benchmark_mode()


    def dump_as_attn_modules(self):
        return [layer.self_attn for layer in self.model.decoder.layers]
    

    def layer_ft_params(self, layer):
        return self.model.decoder.layer_ft_params(layer)


    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=128, eos_token_id=[2]):

        if input_ids.ndim == 3:
            input_ids = input_ids.flatten(0,1)

        # put the tensor on to the model's device
        device = next(iter(self.model.parameters())).device
        input_ids = input_ids.to(device)

        # prefilling
        prefill_ids = input_ids[:, :-1]
        self.model(input_ids=prefill_ids)

        # generation
        new_tok = input_ids[:, -1:]
        new_ids = []
        while len(new_ids) < max_new_tokens:
            logits = self.model(input_ids=new_tok).logits
            new_tok = logits.argmax(dim=-1)
            if new_tok.ravel().item() in eos_token_id: break
            new_ids.append(new_tok.ravel().item())

        self.model.reset()
        new_ids = torch.tensor(new_ids, dtype=input_ids.dtype, device=input_ids.device)[None, :]
        return torch.cat([input_ids, new_ids], dim=-1)