# emotion/compute_acceptance_ratio.py

import math

def compute_acceptance_ratio(valence: float, impact_strength: float, inertia_factor: float = 1.5, k: float = math.e) -> float:
    """
    计算情绪接受度。
    衡量当前情绪状态对新情绪冲击的“接受”或“抵抗”程度。
    """
    resistance = abs(valence) * inertia_factor
    x = impact_strength - resistance
    return 1 / (1 + math.exp(-k * x))
