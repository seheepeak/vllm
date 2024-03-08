import torch
from pathlib import Path
from safetensors import safe_open
from vllm._C import ops

MODEL_PATH = Path.home() / "workspace/hf_models/miqu-1-70b/model.safetensors"
with safe_open(MODEL_PATH, framework="pt", device=0) as f:
    attn_k0: torch.Tensor = f.get_tensor("model.layers.0.self_attn.k_proj.weight")
    print("attn_k0:", attn_k0.shape, attn_k0.dtype, attn_k0.device)
    attn_v0: torch.Tensor = f.get_tensor("model.layers.0.self_attn.v_proj.weight")
    print("attn_v0:", attn_v0.shape, attn_v0.dtype, attn_v0.device)

x = torch.ones((1, 8192), dtype=torch.float16).to(attn_k0.device)
out = ops.gguf_gemm(x, attn_k0)
print("gguf_gemm(ones, atten_k0):", out.shape, out[0, :64])

out = ops.gguf_gemm(x, attn_v0)
print("gguf_gemm(ones, atten_v0):", out.shape, out[0, :64])


# [blk.0.attn_k.weight] ggml_cuda_op_mul_mat_cublas output from llama.cpp
# -- 0: 1.296875
# -- 1: 5.812500
# -- 2: -3.984375
# -- 3: -2.876953
# -- 4: -13.867188
# -- 5: -9.664062
# -- 6: 5.988281
# -- 7: -5.554688
# -- 8: -16.484375
# -- 9: -12.937500
# -- 10: -21.625000
# -- 11: -18.187500
# -- 12: -23.437500
# -- 13: 9.937500
# -- 14: 7.164062
# -- 15: -4.242188
# -- 16: -22.328125
# -- 17: -10.882812
# -- 18: 0.199219
# -- 19: -11.031250
# -- 20: -23.437500
# -- 21: 0.324219
# -- 22: -10.601562
# -- 23: -1.404297
# -- 24: 11.054688
# -- 25: -2.613281
# -- 26: -15.781250
# -- 27: -7.511719
# -- 28: -7.531250
# -- 29: -18.218750
# -- 30: 2.742188
# -- 31: -0.841797
# -- 32: -3.375000
# -- 33: -20.062500
# -- 34: 20.031250
# -- 35: -9.273438
# -- 36: 2.267578
# -- 37: -9.500000
# -- 38: 7.183594
# -- 39: 7.558594
# -- 40: 4.175781
# -- 41: 6.902344
# -- 42: -8.843750
# -- 43: -1.841797
# -- 44: 10.078125
# -- 45: 4.570312
# -- 46: -18.312500
# -- 47: 12.781250
# -- 48: 7.109375
# -- 49: -21.625000
# -- 50: 2.283203
# -- 51: -5.359375
# -- 52: -9.890625
# -- 53: 0.640625
# -- 54: -16.015625
# -- 55: 15.226562
# -- 56: -11.101562
# -- 57: -14.593750
# -- 58: -3.296875
# -- 59: -9.867188
# -- 60: -9.804688
# -- 61: -19.812500
# -- 62: -19.046875
# -- 63: -5.828125

# [blk.0.attn_v.weight] ggml_cuda_op_mul_mat_cublas output from llama.cpp
# -- 0: 1.522461
# -- 1: -1.415039
# -- 2: -1.782227
# -- 3: -0.137451
# -- 4: 2.640625
# -- 5: -2.017578
# -- 6: 0.245361
# -- 7: -2.003906
# -- 8: -1.380859
# -- 9: 1.466797
# -- 10: -0.830566
# -- 11: -0.748535
# -- 12: -2.539062
# -- 13: 1.221680
# -- 14: -1.071289
# -- 15: -1.083984
# -- 16: 0.241699
# -- 17: -2.568359
# -- 18: 0.878418
# -- 19: -0.207153
# -- 20: -1.995117
# -- 21: 2.056641
# -- 22: 1.414062
# -- 23: -0.075684
# -- 24: -2.761719
# -- 25: 0.201172
# -- 26: -0.522949
# -- 27: 2.283203
# -- 28: 2.341797
# -- 29: 6.406250
# -- 30: 1.957031
# -- 31: -0.695312
# -- 32: -1.346680
# -- 33: 3.875000
# -- 34: -0.484131
# -- 35: -3.480469
# -- 36: 0.256348
# -- 37: 1.165039
# -- 38: -0.023926
# -- 39: -0.566406
# -- 40: 0.443359
# -- 41: 0.234619
# -- 42: 0.717773
# -- 43: -1.398438
# -- 44: 2.513672
# -- 45: 1.276367
# -- 46: 1.111328
# -- 47: -0.151855
# -- 48: -0.070557
# -- 49: -1.718750
# -- 50: 0.061890
# -- 51: -1.007812
# -- 52: 0.824219
# -- 53: 0.010742
# -- 54: -0.767090
# -- 55: -0.492676
# -- 56: 2.582031
# -- 57: -0.989258
# -- 58: 0.816406
# -- 59: 1.038086
# -- 60: -0.604980
# -- 61: 0.534668
# -- 62: -0.400391
# -- 63: 2.699219
