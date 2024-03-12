import enum
from enum import Enum
from typing import Any, Dict, List, Optional
from fractions import Fraction

import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F

from vllm._C import ops
from vllm.model_executor.layers.linear import LinearMethodBase, set_weight_attrs
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig


class GGUFConfig(QuantizationConfig):
    def __init__(self) -> None:
        pass

    def __repr__(self) -> str:
        return "GGUFConfig()"

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
        return cls()

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

    def create_weights(
        self,
        input_size_per_partition: int,
        output_size_per_partition: int,
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
    ) -> Dict[str, Any]:
        # set column dimension to 0 (unresolved dimension)
        # weight.resize_ required
        weight = Parameter(torch.empty(0, 0, dtype=params_dtype), requires_grad=False)
        set_weight_attrs(
            weight,
            {
                "input_dim": 1,
                "output_dim": 0,
                "dequant_shape": (output_size_per_partition, input_size_per_partition),
            },
        )
        return {"weight": weight}

    def apply_weights(
        self, weights: Dict[str, Any], x: torch.Tensor, bias: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        weight = weights["weight"]
        weight_dtype = weight.dtype
    
        if weight_dtype in [torch.half, torch.bfloat16, torch.float32]:
            if bias:
                return F.linear(x, weight) + bias
            return F.linear(x, weight)
        elif weight_dtype == torch.uint8:
            out_shape = x.shape[:-1] + (weight.shape[0],)
            reshaped_x = x.reshape(-1, x.shape[-1])
            output: torch.Tensor = ops.gguf_gemm(reshaped_x, weight)
            if bias:
                output = output + bias
            return output.reshape(out_shape)

        raise ValueError(f"Unsupported weight dtype: {weight_dtype}")
