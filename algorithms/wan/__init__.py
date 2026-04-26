from .depth_backends import DepthBackend, RepeatCurrentDepthBackend
from .idm_backends import CustomIDMBackend, VidarBackendConfig, VidarIDMBackend, build_idm_backend
from .state_action_reward import (
    RewardBreakdown as StepRewardBreakdown,
    StepExecutabilityReward,
)
from .wan_i2v import WanImageToVideo
from .wan_t2v import WanTextToVideo

# Backward-compatible default export for callers that used `algorithms.wan.RewardBreakdown`
# in the step-level GRPO pipeline.
RewardBreakdown = StepRewardBreakdown

__all__ = [
    "CustomIDMBackend",
    "DepthBackend",
    "RepeatCurrentDepthBackend",
    "RewardBreakdown",
    "StepExecutabilityReward",
    "StepRewardBreakdown",
    "VidarBackendConfig",
    "VidarIDMBackend",
    "WanImageToVideo",
    "WanTextToVideo",
    "build_idm_backend",
]
