from typing import Any, Dict, List, Optional
import re
from dataclasses import dataclass
from collections import defaultdict

import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F

from vllm._C import ops
from vllm.model_executor.layers.linear import LinearMethodBase, set_weight_attrs
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

@dataclass
class LayerGGMLTypes:
    q_proj: int = 0
    k_proj: int = 0
    v_proj: int = 0
    o_proj: int = 0
    gate_proj: int = 0
    up_proj: int = 0
    down_proj: int = 0

    def is_qk_same(self) -> bool:
        return self.q_proj == self.k_proj
    
    def is_qkv_same(self) -> bool:
        return self.q_proj == self.k_proj == self.v_proj
    
    def is_gate_up_same(self) -> bool:
        return self.gate_proj == self.up_proj

# GGML_TYPE_FLOAT32(0), GGML_TYPE_F16(1), GGML_TYPE_Q5_K(13), GGML_TYPE_Q6_K(14)

class GGUFConfig(QuantizationConfig):
    def __init__(self, rope_style: str, ggml_types_per_layer: Dict[int, LayerGGMLTypes]) -> None:
        assert rope_style in ["neox", "gptj"]
        self.rope_style = rope_style
        self.ggml_types_per_layer = ggml_types_per_layer
        self.num_layers = max(ggml_types_per_layer.keys()) + 1

        # NOTE(sehee): resolve qkv or qk stacking
        if all(ggml_types.is_qkv_same() for ggml_types in self.ggml_types_per_layer.values()):
            self.self_attn_stacking = "qkv"
        elif all(ggml_types.is_qk_same() for ggml_types in self.ggml_types_per_layer.values()):
            self.self_attn_stacking = "qk"
        else:
            raise ValueError(f"Unsupported self-attention QKV stacking")
        
        if not all(ggml_types.is_gate_up_same() for ggml_types in self.ggml_types_per_layer.values()):
            raise ValueError(f"Unsupported gate and up stacking")

    def get_linear_weights_config(self, layer) -> List[int]:
        assert 0 <= layer < self.num_layers
        ggml_types = self.ggml_types_per_layer[layer]
        # NOTE(sehee): create_weights 가 호출되는 순서와 list item 의 순서가 일치해야한다
        if self.self_attn_stacking == "qkv":
            return [ggml_types.q_proj, ggml_types.o_proj, ggml_types.gate_proj, ggml_types.down_proj]
        elif self.self_attn_stacking == "qk":
            return [ggml_types.q_proj, ggml_types.v_proj, ggml_types.o_proj, ggml_types.gate_proj, ggml_types.down_proj]
        else:
            raise ValueError(f"Unsupported self-attention QKV stacking mode: {self.self_attn_stacking}")

    def __repr__(self) -> str:
        return (f"GGUFConfig(rope_style={self.rope_style}, ggml_types_per_layer={self.ggml_types_per_layer})")

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
        rope_style = config.get("rope_style", "neox")
        ggml_types_per_layer = defaultdict(LayerGGMLTypes)
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
                    setattr(ggml_types_per_layer[layer], name, v)
        return cls(rope_style, ggml_types_per_layer)

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
        self.weights_config = []

    def configure_weights_for_layer(self, layer: int):
        assert not self.weights_config
        self.weights_config = self.quant_config.get_linear_weights_config(layer)

    def create_weights(
        self,
        input_size_per_partition: int,
        output_size_per_partition: int,
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
    ) -> Dict[str, Any]:
        ggml_type = self.weights_config.pop(0)
        BYTES_PER_QBLOCK = {13 : 176, 14 : 210}
        if block_bytes := BYTES_PER_QBLOCK.get(ggml_type, 0):
            assert input_size_per_partition % 256 == 0
            input_bytes = input_size_per_partition // 256 * block_bytes
            weight = Parameter(torch.empty(output_size_per_partition, input_bytes, dtype=torch.uint8), requires_grad=False)
        elif ggml_type in [0, 1]:
            weight = Parameter(torch.empty(output_size_per_partition, input_size_per_partition, dtype=params_dtype), requires_grad=False)
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
