# miqu gguf 모델을 GPTQ int8 weight-only 32g 로 변환하는 스크립트
# [from] https://huggingface.co/miqudev/miqu-1-70b/blob/main/miqu-1-70b.q5_K_M.gguf
# [*ref* to] https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ/tree/gptq-8bit-128g-actorder_True
# [*ref* to] https://huggingface.co/MaziyarPanahi/miqu-1-70b-sf-GPTQ
# (for vLLM inference)

import os
import re
import sys
import json
import struct
import numpy as np
import torch
from pathlib import Path
from tqdm.auto import tqdm
from safetensors.torch import save_file

torch.set_printoptions(precision=8, sci_mode=False)
np.set_printoptions(precision=8, suppress=True)

from safetensors import safe_open

# [gguf tensors]
# tensor: {'name': 'output.weight', 'dims': [8192, 32000], 'type': 14, 'offset': 0}
# tensor: {'name': 'token_embd.weight', 'dims': [8192, 32000], 'type': 13, 'offset': 215040000}
# tensor: {'name': 'output_norm.weight', 'dims': [8192], 'type': 0, 'offset': 395264000}
# tensor: {'name': 'blk.0.attn_norm.weight', 'dims': [8192], 'type': 0, 'offset': 395296768}
# tensor: {'name': 'blk.0.ffn_norm.weight', 'dims': [8192], 'type': 0, 'offset': 395329536}
# tensor: {'name': 'blk.0.ffn_down.weight', 'dims': [28672, 8192], 'type': 14, 'offset': 395362304}
# tensor: {'name': 'blk.0.ffn_gate.weight', 'dims': [8192, 28672], 'type': 13, 'offset': 588038144}
# tensor: {'name': 'blk.0.ffn_up.weight', 'dims': [8192, 28672], 'type': 13, 'offset': 749518848}
# tensor: {'name': 'blk.0.attn_k.weight', 'dims': [8192, 1024], 'type': 13, 'offset': 910999552}
# tensor: {'name': 'blk.0.attn_output.weight', 'dims': [8192, 8192], 'type': 13, 'offset': 916766720}
# tensor: {'name': 'blk.0.attn_q.weight', 'dims': [8192, 8192], 'type': 13, 'offset': 962904064}
# tensor: {'name': 'blk.0.attn_v.weight', 'dims': [8192, 1024], 'type': 14, 'offset': 1009041408}

# [safetensors]
# model.embed_tokens.weight torch.Size([32000, 8192]) 
# model.layers.0.input_layernorm.weight torch.Size([8192]) 
# model.layers.0.mlp.down_proj.weight torch.Size([8192, 28672]) 
# model.layers.0.mlp.gate_proj.weight torch.Size([28672, 8192]) 
# model.layers.0.mlp.up_proj.weight torch.Size([28672, 8192]) 
# model.layers.0.post_attention_layernorm.weight torch.Size([8192]) 
# model.layers.0.self_attn.k_proj.weight torch.Size([1024, 8192]) 
# model.layers.0.self_attn.o_proj.weight torch.Size([8192, 8192]) 
# model.layers.0.self_attn.q_proj.weight torch.Size([8192, 8192]) 
# model.layers.0.self_attn.v_proj.weight torch.Size([1024, 8192]) 

# with safe_open(
#     f"/home/sehee/workspace/hf_models/miqu-from-gguf/model.safetensors",
#     framework="pt",
#     device=0,
# ) as f:
#     for k in f.keys():
#         t: torch.Tensor = f.get_tensor(k)
#         print(k, t.shape, t.dtype)
#         # continue
#         # if k == "model.layers.0.self_attn.k_proj.weight":
#         #     print("0/1024 ==>", t[0, :256])
#         #     print("1/1024 ==>", t[1, :256])
#         #     print("2/1024 ==>", t[2, :256])
#         #     t = t.detach().cpu().numpy()
#         #     b = np.ones((8192, 1), dtype=np.float16)
#         #     c = np.matmul(t, b)
#         #     print("c:", c[:16, 0], c.shape, c.dtype)
# exit()


def read_metadata_value(value_type, fin):
    if value_type == 0:  # uint8
        value = struct.unpack("B", fin.read(1))[0]
    elif value_type == 1:  # int8
        value = struct.unpack("b", fin.read(1))[0]
    elif value_type == 2:  # uint16
        value = struct.unpack("H", fin.read(2))[0]
    elif value_type == 3:  # int16
        value = struct.unpack("h", fin.read(2))[0]
    elif value_type == 4:  # uint32
        value = struct.unpack("I", fin.read(4))[0]
    elif value_type == 5:  # int32
        value = struct.unpack("i", fin.read(4))[0]
    elif value_type == 6:  # float32
        value = struct.unpack("f", fin.read(4))[0]
    elif value_type == 7:  # bool
        value = bool(struct.unpack("B", fin.read(1))[0])
    elif value_type == 8:  # string
        value_len = struct.unpack("Q", fin.read(8))[0]
        value = fin.read(value_len).decode("utf-8")
    elif value_type == 9:  # array
        value = []
        value_type = struct.unpack("I", fin.read(4))[0]  # 0~12
        count = struct.unpack("Q", fin.read(8))[0]
        for _ in range(count):
            value.append(read_metadata_value(value_type, fin))
    elif value_type == 10:  # uint64
        value = struct.unpack("Q", fin.read(8))[0]
    elif value_type == 11:  # int64
        value = struct.unpack("q", fin.read(8))[0]
    elif value_type == 12:  # float64
        value = struct.unpack("d", fin.read(8))[0]
    else:
        raise NotImplementedError
    return value


def calc_tensor_buf_size(type, dims):
    count = int(np.prod(dims))
    if type == 0:  # GGML_TYPE_F32  = 0
        bit_per_weight = 32
        return int(count * bit_per_weight) // 8
    elif type == 13:  # GGML_TYPE_Q5_K = 13
        assert count % 256 == 0
        bit_per_weight = 5.5
        return int(count * bit_per_weight) // 8
    elif type == 14:  # GGML_TYPE_Q6_K = 14
        assert count % 256 == 0
        bit_per_weight = 6.5625
        return int(count * bit_per_weight) // 8
    else:
        raise NotImplementedError


def get_scale_min_k4(j, q):
    if j < 4:
        d = q[j] & 63
        m = q[j + 4] & 63
    else:
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4)
        m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4)
    return d.astype(np.uint8), m.astype(np.uint8)


def load_tensor_weight(name_in_gguf, ggml_type, dims, buf):
    count = int(np.prod(dims))
    if ggml_type == 0:  # GGML_TYPE_FLOAT
        # keep float32 as it was
        # weight = np.zeros((count,), dtype=np.float16)
        # weight[:] = np.frombuffer(buf, dtype=np.float32)
        weight = np.frombuffer(buf, dtype=np.float32)
    elif ggml_type == 13:  # GGML_TYPE_Q5_K = 13
        if name_in_gguf in ["token_embd.weight"]:
            weight = np.zeros((count,), dtype=np.float32)
            y = 0
            buf_idx = 0
            for _ in tqdm(range(count // 256)):
                d = np.frombuffer(buf[buf_idx + 0 : buf_idx + 2], dtype=np.float16).astype(np.float32)[0]
                dmin = np.frombuffer(buf[buf_idx + 2 : buf_idx + 4], dtype=np.float16).astype(np.float32)[0]
                scales = np.frombuffer(buf[buf_idx + 4 : buf_idx + 16], dtype=np.uint8)
                qh = np.frombuffer(buf[buf_idx + 16 : buf_idx + 48], dtype=np.uint8)
                ql = np.frombuffer(buf[buf_idx + 48 : buf_idx + 176], dtype=np.uint8)
                buf_idx += 176

                _is, u1, u2 = 0, 1, 2
                for j in range(0, 256, 64):
                    sc, m = get_scale_min_k4(_is + 0, scales)
                    d1, m1 = d * sc, dmin * m
                    sc, m = get_scale_min_k4(_is + 1, scales)
                    d2, m2 = d * sc, dmin * m
                    weight[y : y + 32] = d1 * ((ql[:32] & 0xF) + np.where(qh & u1, 16, 0)) - m1
                    weight[y + 32 : y + 64] = d2 * ((ql[:32] >> 4) + np.where(qh & u2, 16, 0)) - m2
                    ql = ql[32:]
                    _is += 2
                    u1 <<= 2
                    u2 <<= 2
                    y += 64
            # keep float32 as llama.cpp does
            # weight = weight.astype(np.float16)
        else:
            assert dims[0] % 256 == 0
            weight = np.frombuffer(buf, dtype=np.uint8)
            dims[0] = dims[0] * 176 // 256
            assert weight.size == np.prod(dims)

    elif ggml_type == 14:  # GGML_TYPE_Q6_K = 14
        if name_in_gguf in ["output.weight"]:
            weight = np.zeros((count,), dtype=np.float32)
            y = 0
            buf_idx = 0
            for _ in tqdm(range(count // 256)):
                ql = np.frombuffer(buf[buf_idx + 0 : buf_idx + 128], dtype=np.uint8)
                qh = np.frombuffer(buf[buf_idx + 128 : buf_idx + 192], dtype=np.uint8)
                sc = np.frombuffer(buf[buf_idx + 192 : buf_idx + 208], dtype=np.int8)
                d = np.frombuffer(buf[buf_idx + 208 : buf_idx + 210], dtype=np.float16).astype(np.float32)[0]
                buf_idx += 210

                for j in range(0, 256, 128):
                    _is = np.array([0] * 16 + [1] * 16, dtype=np.int32)
                    q1 = (((ql[:32] & 0xF) | (((qh[:32] >> 0) & 3) << 4)).astype(np.int8) - 32).astype(np.int8)
                    q2 = (((ql[32:64] & 0xF) | (((qh[:32] >> 2) & 3) << 4)).astype(np.int8) - 32).astype(np.int8)
                    q3 = (((ql[:32] >> 4) | (((qh[:32] >> 4) & 3) << 4)).astype(np.int8) - 32).astype(np.int8)
                    q4 = (((ql[32:64] >> 4) | (((qh[:32] >> 6) & 3) << 4)).astype(np.int8) - 32).astype(np.int8)
                    weight[y : y + 32] = d * sc[_is + 0] * q1
                    weight[y + 32 : y + 64] = d * sc[_is + 2] * q2
                    weight[y + 64 : y + 96] = d * sc[_is + 4] * q3
                    weight[y + 96 : y + 128] = d * sc[_is + 6] * q4
                    y += 128
                    ql = ql[64:]
                    qh = qh[32:]
                    sc = sc[8:]
            # keep float32 as llama.cpp does
            # weight = weight.astype(np.float16)
        else:
            assert dims[0] % 256 == 0
            weight = np.frombuffer(buf, dtype=np.uint8)
            dims[0] = dims[0] * 210 // 256
            assert weight.size == np.prod(dims)

    else:
        raise NotImplementedError

    if len(dims) == 2:
        # column-major k*m matrix ==> row-major m*k matrix
        k, m = dims
        return torch.tensor(weight.reshape((m, k)))
    elif len(dims) == 1:
        assert dims[0] == count
        return torch.tensor(weight)
    else:
        raise NotImplementedError


def convert_tensor_name(name_in_gguf):
    if name_in_gguf == "output.weight":
        return "lm_head.weight"  # [8192, 32000] ==> [32000, 8192]
    elif name_in_gguf == "token_embd.weight":
        return "model.embed_tokens.weight"  # [8192, 32000] ==> [32000, 8192]
    elif name_in_gguf == "output_norm.weight":
        return "model.norm.weight"  # [8192]
    elif m := re.match(r"blk\.(\d+)\.attn_norm\.weight", name_in_gguf):
        layer = m.group(1)
        return f"model.layers.{layer}.input_layernorm.weight"  # [8192]
    elif m := re.match(r"blk\.(\d+)\.ffn_norm\.weight", name_in_gguf):
        layer = m.group(1)
        return f"model.layers.{layer}.post_attention_layernorm.weight"  # [8192]
    elif m := re.match(r"blk\.(\d+)\.ffn_down\.weight", name_in_gguf):
        layer = m.group(1)
        return f"model.layers.{layer}.mlp.down_proj.weight"  # [28672, 8192] ==> [8192, 28672]
    elif m := re.match(r"blk\.(\d+)\.ffn_gate\.weight", name_in_gguf):
        layer = m.group(1)
        return f"model.layers.{layer}.mlp.gate_proj.weight"  # [8192, 28672] ==> [28672, 8192]
    elif m := re.match(r"blk\.(\d+)\.ffn_up\.weight", name_in_gguf):
        layer = m.group(1)
        return f"model.layers.{layer}.mlp.up_proj.weight"  # [28672, 8192] ==> [8192, 28672]
    elif m := re.match(r"blk\.(\d+)\.attn_k\.weight", name_in_gguf):
        layer = m.group(1)
        return f"model.layers.{layer}.self_attn.k_proj.weight"  # [8192, 1024] ==> [1024, 8192]
    elif m := re.match(r"blk\.(\d+)\.attn_output\.weight", name_in_gguf):
        layer = m.group(1)
        return f"model.layers.{layer}.self_attn.o_proj.weight"  # [8192, 8192] ==> [8192, 8192]
    elif m := re.match(r"blk\.(\d+)\.attn_q\.weight", name_in_gguf):
        layer = m.group(1)
        return f"model.layers.{layer}.self_attn.q_proj.weight"  # [8192, 8192] ==> [8192, 8192]
    elif m := re.match(r"blk\.(\d+)\.attn_v\.weight", name_in_gguf):
        layer = m.group(1)
        return f"model.layers.{layer}.self_attn.v_proj.weight"  # [8192, 1024] ==> [1024, 8192]
    raise NotImplementedError


GGUF_FILE = Path.home() / "workspace/llama.cpp/models/miqu-1-70b.q5_K_M.gguf"
fin = GGUF_FILE.open("rb")
magic, version, tensor_count, metadata_kv_count = struct.unpack("IIQQ", fin.read(4 + 4 + 8 + 8))
assert magic == 0x46554747 and version == 3
print("magic:", "0x" + format(magic, "08x"))
print("version:", version)
print("tensor_count:", tensor_count)
print("metadata_kv_count:", metadata_kv_count)
for i in range(metadata_kv_count):
    key_len = struct.unpack("Q", fin.read(8))[0]
    key = fin.read(key_len).decode("utf-8")
    value_type = struct.unpack("I", fin.read(4))[0]  # 0~12
    value = read_metadata_value(value_type, fin)
    print("metadata:", key, value)

tensor_data = []
for i in range(tensor_count):
    tensor = {}
    name_len = struct.unpack("Q", fin.read(8))[0]
    tensor["name"] = fin.read(name_len).decode("utf-8")
    n_dims = struct.unpack("I", fin.read(4))[0]
    dims = []
    for _ in range(n_dims):
        dims.append(struct.unpack("Q", fin.read(8))[0])
    tensor["dims"] = dims
    tensor["type"] = struct.unpack("I", fin.read(4))[0]
    tensor["offset"] = struct.unpack("Q", fin.read(8))[0]
    print("tensor:", tensor)
    tensor_data.append(tensor)

# handle padding
tensor_data_offset = fin.tell()
ALIGNMENT = 32
padding = tensor_data_offset % ALIGNMENT
if padding != 0:
    fin.read(ALIGNMENT - padding)
    tensor_data_offset = fin.tell()
print("tensor_data_offset:", tensor_data_offset)

# Note: linear matrix 에서 model.layers.{layer}.mlp.down_proj.weight, model.layers.{layer}.self_attn.v_proj.weight 는 q6_k 이고, 나머지는 q5_k 이다.
out_tensors = {}
for i in range(tensor_count):
    tensor = tensor_data[i]
    fin.seek(tensor_data_offset + tensor["offset"])
    buf = fin.read(calc_tensor_buf_size(tensor["type"], tensor["dims"]))
    weights = load_tensor_weight(tensor["name"], tensor["type"], tensor["dims"], buf)
    out_name = convert_tensor_name(tensor["name"])
    out_tensors[out_name] = weights
    print("weights:", tensor["name"], weights.shape, weights.dtype)
    # if tensor["name"] == "blk.0.attn_k.weight":
    #     fin.seek(tensor_data_offset + tensor["offset"])
    #     buf = fin.read(calc_tensor_buf_size(tensor["type"], tensor["dims"]))
    #     weights = load_tensor_weight(tensor["name"], tensor["type"], tensor["dims"], buf)
    #     print("weights:", tensor["name"], weights.shape, weights.dtype)
    #     # print("0/1024 ==>", weights[0, 0:256])
    #     # print("1/1024 ==>", weights[1, 0:256])
    #     # print("2/1024 ==>", weights[2, 0:256])

    #     # b = np.ones((8192, 1), dtype=np.float16)
    #     # c = np.matmul(weights, b)
    #     # print("c:", c[0:256, 0], c.shape, c.dtype)


from safetensors.torch import save_file

save_file(out_tensors, "model.safetensors")
