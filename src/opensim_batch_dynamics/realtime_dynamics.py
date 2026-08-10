from __future__ import annotations

from dataclasses import dataclass, field
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from RT.rt_library import RealtimeWorkspace, get_model_dof_names, initialize_rt_state, qpid


@dataclass
class RealtimeState:
    q: object
    dq: object
    ddq: object
    tau_full: object
    root_residual: object
    foot_wrenches: dict = field(default_factory=lambda: {"left": None, "right": None})
    contact_state: dict = field(default_factory=lambda: {"left": False, "right": False})
    contact_prob: dict = field(default_factory=lambda: {"left": 0.0, "right": 0.0})
    floor_height: float = float("nan")
    step_index: int = 0


@dataclass
class RealtimeConfig:
    dt: float = 0.033
    mu: float = 0.8
    steps: int = 1
    use_stage1_kin_filter: bool = True
    excluded_DOFs: tuple[int, ...] = ()
    init_policy: str = "offline_first_frame"


@dataclass
class RealtimeFrameResult:
    q: object
    dq: object
    ddq: object
    tau: object
    tau_full: object
    root_residual: object
    foot_forces: dict = field(default_factory=dict)
    foot_wrenches: dict = field(default_factory=dict)
    contact_state: dict = field(default_factory=dict)
    contact_prob: dict = field(default_factory=dict)
    floor_height: float = float("nan")
    dynamics_residual_norm: float = 0.0

__all__ = [
    "RealtimeConfig",
    "RealtimeFrameResult",
    "RealtimeState",
    "RealtimeWorkspace",
    "get_model_dof_names",
    "initialize_rt_state",
    "qpid",
]
