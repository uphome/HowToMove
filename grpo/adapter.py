"""参数化方式：LoRA（4GB 主用）/ 全参（换卡用），对外接口完全一致。

两种模式都必须能给出**精确的冻结参考分布**：

    LoRA   → 关掉 adapter，用的就是基座权重（零额外显存）
    全参   → 一份 bf16 冻结快照（+1.2GB）

训练代码只依赖 `TrainAdapter` 的四个方法，看不到 LoRA 字样。
"""

from __future__ import annotations

import contextlib
import math
from abc import ABC, abstractmethod
from typing import Callable, Iterable

import torch
import torch.nn as nn

DEFAULT_TARGETS = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)


# ---------------------------------------------------------------- LoRA 层


class LoRALinear(nn.Module):
    """y = base(x) + scale * B(A(x))

    B 零初始化 → 训练开始时输出与基座完全一致（ratio 第 0 步严格为 1）。
    """

    def __init__(self, base: nn.Linear, r: int = 16, alpha: int = 32):
        super().__init__()
        self.base = base
        base.requires_grad_(False)
        self.r = r
        self.scale = alpha / r
        # LoRA 参数统一 fp32 + 跟随基座设备：bf16 下小梯度会被直接吃掉
        w = base.weight
        self.lora_A = nn.Parameter(torch.empty(r, w.shape[1], device=w.device,
                                               dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(w.shape[0], r, device=w.device,
                                               dtype=torch.float32))
        # A 必须非零：A=B=0 时 dL/dA ∝ B = 0 且 dL/dB ∝ A = 0，梯度恒为零。
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.enabled = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        if self.enabled:
            delta = (x.float() @ self.lora_A.t() @ self.lora_B.t()) * self.scale
            y = y + delta.to(y.dtype)
        return y

    def disable(self) -> None:
        self.enabled = False

    def enable(self) -> None:
        self.enabled = True


def _iter_named_linear(module: nn.Module, prefix: str = ""):
    for name, child in module.named_children():
        full = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear):
            yield module, name, full
        else:
            yield from _iter_named_linear(child, full)


def apply_lora(model: nn.Module, r: int = 16, alpha: int = 32,
               targets: Iterable[str] = DEFAULT_TARGETS) -> list[nn.Parameter]:
    """把名字以 targets 结尾的 Linear 替换成 LoRALinear，返回可训练参数。"""
    targets = tuple(targets)
    model.requires_grad_(False)
    replaced = []
    for parent, name, full in list(_iter_named_linear(model)):
        if not full.endswith(targets):
            continue
        setattr(parent, name, LoRALinear(getattr(parent, name), r, alpha))
        replaced.append(full)
    if not replaced:
        raise RuntimeError(f"no module matched {targets}")
    return [p for p in model.parameters() if p.requires_grad]


# ---------------------------------------------------------------- 接口


class TrainAdapter(ABC):
    model: nn.Module

    @abstractmethod
    def trainable_parameters(self) -> list[nn.Parameter]: ...

    @abstractmethod
    def param_groups(self, base_lr: float) -> list[dict]: ...

    @abstractmethod
    def ref_context(self) -> contextlib.AbstractContextManager: ...

    @abstractmethod
    def state_dict(self) -> dict: ...

    @abstractmethod
    def load_state_dict(self, state: dict) -> None: ...

    @property
    @abstractmethod
    def extra_memory_mb(self) -> float: ...


class LoRAAdapter(TrainAdapter):
    def __init__(self, model: nn.Module, r: int = 16, alpha: int = 32,
                 lr_scale: float = 1.0,
                 targets: Iterable[str] = DEFAULT_TARGETS):
        self.model = model
        self.params = apply_lora(model, r, alpha, targets)
        self.lr_scale = lr_scale
        self._saved: list[tuple[LoRALinear, bool]] = []

    def trainable_parameters(self) -> list[nn.Parameter]:
        return self.params

    def param_groups(self, base_lr: float) -> list[dict]:
        return [{"params": self.params, "lr": base_lr * self.lr_scale,
                 "weight_decay": 0.0}]

    def _lora_layers(self) -> list[LoRALinear]:
        return [m for m in self.model.modules() if isinstance(m, LoRALinear)]

    def set_enabled(self, flag: bool) -> None:
        for layer in self._lora_layers():
            layer.enabled = flag

    @contextlib.contextmanager
    def ref_context(self):
        """临时关掉 adapter → 得到精确的冻结基座分布。"""
        self.set_enabled(False)
        try:
            with torch.no_grad():
                yield self.model
        finally:
            self.set_enabled(True)

    def state_dict(self) -> dict:
        return {
            name: {"A": m.lora_A.detach().clone(),
                   "B": m.lora_B.detach().clone()}
            for name, m in self.model.named_modules()
            if isinstance(m, LoRALinear)
        }

    def load_state_dict(self, state: dict) -> None:
        for name, m in self.model.named_modules():
            if isinstance(m, LoRALinear) and name in state:
                m.lora_A.data.copy_(state[name]["A"])
                m.lora_B.data.copy_(state[name]["B"])

    @property
    def extra_memory_mb(self) -> float:
        return sum(p.numel() for p in self.params) * 4 / 2**20


class FullFTAdapter(TrainAdapter):
    """全参微调。需要额外一份 bf16 快照做 ref，4GB 卡上不可用。"""

    def __init__(self, model: nn.Module, weight_decay: float = 0.01):
        self.model = model
        self.weight_decay = weight_decay
        tie_embeddings(model)
        self.params = [p for p in model.parameters() if p.requires_grad]
        self.ref_model = self._snapshot(model)

    @staticmethod
    def _snapshot(model: nn.Module) -> nn.Module:
        import copy

        ref = copy.deepcopy(model)
        for p in ref.parameters():
            p.requires_grad_(False)
        return ref

    def trainable_parameters(self) -> list[nn.Parameter]:
        return self.params

    def param_groups(self, base_lr: float) -> list[dict]:
        return [{"params": self.params, "lr": base_lr,
                 "weight_decay": self.weight_decay}]

    @contextlib.contextmanager
    def ref_context(self):
        with torch.no_grad():
            yield self.ref_model

    def state_dict(self) -> dict:
        return {k: v.detach().clone() for k, v in self.model.state_dict().items()}

    def load_state_dict(self, state: dict) -> None:
        self.model.load_state_dict(state)

    @property
    def extra_memory_mb(self) -> float:
        return sum(p.numel() for p in self.params) * 2 / 2**20


def tie_embeddings(model: nn.Module) -> None:
    """把 lm_head 与 embed_tokens 绑成同一个 Parameter（全参训练必须）。"""
    core = getattr(model, "model", model)
    embed = getattr(core, "embed_tokens", None)
    lm_head = getattr(model, "lm_head", None)
    if embed is not None and lm_head is not None:
        lm_head.weight = embed.weight
