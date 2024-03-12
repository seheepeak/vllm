import sys
import enum
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from fractions import Fraction
import re

import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F

from vllm._C import ops
from vllm.model_executor.layers.linear import LinearMethodBase, set_weight_attrs
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

class GGUFConfig(QuantizationConfig):
    def __init__(self, ggml_types_per_layer: Dict[str, dict]) -> None:
        # GGML_TYPE_FLOAT32(0), GGML_TYPE_F16(1), GGML_TYPE_Q5_K(13), GGML_TYPE_Q6_K(14)
        self.ggml_types_per_layer = ggml_types_per_layer
        self.num_layers = max(ggml_types_per_layer.keys()) + 1
        for i in range(self.num_layers):
            assert self.is_qkv_packed(i) or self.is_qk_packed(i), "QKV or QK should have same GGML quant type."
            assert self.is_gate_up_packed(i), "mlp.gate and mlp.up should have same GGML quant type."

    def get_ggml_types(self, layer: int) -> Dict[str, int]:
        assert 0 <= layer < self.num_layers
        return self.ggml_types_per_layer[layer]

    def is_qkv_packed(self, layer: int) -> bool:
        ggml_types = self.get_ggml_types(layer)
        return ggml_types['q_proj'] == ggml_types['k_proj'] == ggml_types['v_proj']
    
    def is_qk_packed(self, layer: int) -> bool:
        ggml_types = self.get_ggml_types(layer)
        return ggml_types['q_proj'] == ggml_types['k_proj'] and ggml_types['v_proj'] != ggml_types['q_proj']
    
    def is_gate_up_packed(self, layer: int) -> bool:
        ggml_types = self.get_ggml_types(layer)
        return ggml_types['up_proj'] == ggml_types['gate_proj']

    def __repr__(self) -> str:
        return (f"GGUFConfig(ggml_types_per_layer={self.ggml_types_per_layer})")

    @classmethod
    def get_name(cls) -> str:
        return "gguf"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.half, torch.bfloat16, torch.float32]

    @classmethod
    # Need to figure it out
    def get_min_capability(cls) -> int:
        return 60

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return ["quantize_config.json"]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "GGUFConfig":
        ggml_types_per_layer = {}
        patterns = [(r"model\.layers\.(\d+)\.self_attn\.q_proj\.weight", "q_proj"),
            (r"model\.layers\.(\d+)\.self_attn\.k_proj\.weight", "k_proj"),
            (r"model\.layers\.(\d+)\.self_attn\.v_proj\.weight", "v_proj"),
            (r"model\.layers\.(\d+)\.self_attn\.o_proj\.weight", "o_proj"),
            (r"model\.layers\.(\d+)\.mlp\.gate_proj\.weight", "gate_proj"),
            (r"model\.layers\.(\d+)\.mlp\.up_proj\.weight", "up_proj"),
            (r"model\.layers\.(\d+)\.mlp\.down_proj\.weight", "down_proj")]
        for k, v in config.items():
            for p, name in patterns:
                if m := re.match(p, k):
                    layer = int(m.group(1))
                    ggml_types_per_layer.setdefault(layer, {})[name] = v
                    break
        return cls(ggml_types_per_layer)

    def get_linear_method(self) -> "GGUFLinearMethod":
        return GGUFLinearMethod(self)

    def get_scaled_act_names(self) -> List[str]:
        return []

class GGUFLinearMethod(LinearMethodBase):
    """Linear method for GPTQ.

    Args:
        quant_config: The GPTQ quantization config.
    """

    def __init__(self, quant_config: GGUFConfig):
        self.quant_config = quant_config
        self.weight_type_list = []

    def configure_weights(self, layer: int):
        assert not self.weight_type_list
        ggml_types = self.quant_config.get_ggml_types(layer)
        self.weight_type_list = [ggml_types['q_proj'], ggml_types['v_proj'], ggml_types['o_proj'], ggml_types['gate_proj'], ggml_types['down_proj']]

    def create_weights(
        self,
        input_size_per_partition: int,
        output_size_per_partition: int,
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
    ) -> Dict[str, Any]:
        ggml_type = self.weight_type_list.pop(0)
        BYTES_PER_QBLOCK = {13 : 176, 14 : 210}
        if block_bytes := BYTES_PER_QBLOCK.get(ggml_type, 0):
            assert input_size_per_partition % 256 == 0
            input_bytes = input_size_per_partition // 256 * block_bytes
            weight = Parameter(torch.empty(output_size_per_partition, input_bytes, dtype=torch.uint8), requires_grad=False)
        elif ggml_type in [0, 1]:
            dtype = torch.float32 if ggml_type == 0 else torch.half
            weight = Parameter(torch.empty(output_size_per_partition, input_size_per_partition, dtype=dtype), requires_grad=False)
        else:
            raise ValueError(f"Unsupported weight dtype: {ggml_type}")
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        return {"weight": weight}

    def apply_weights(
        self, weights: Dict[str, Any], x: torch.Tensor, bias: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        weight = weights["weight"]
        weight_dtype = weight.dtype
    
        if weight_dtype == torch.uint8:
            out_shape = x.shape[:-1] + (weight.shape[0],)
            reshaped_x = x.reshape(-1, x.shape[-1])
            output: torch.Tensor = ops.gguf_gemm(reshaped_x, weight)
            if bias:
                output = output + bias
            return output.reshape(out_shape)
        else:
            if bias:
                return F.linear(x, weight) + bias
            return F.linear(x, weight)
