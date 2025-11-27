# emotion/compute_acceptance_ratio.py

import math

def compute_acceptance_ratio(valence: float, impact_strength: float, inertia_factor: float) -> float:
    """
    计算情绪接受度。
    衡量当前情绪状态对新情绪冲击的“接受”或“抵抗”程度。
    :param valence: 当前的情绪效价。
    :param impact_strength: 新情绪冲击的强度。
    :param inertia_factor: 情绪惯性因子，值越高代表情绪越稳定，越能抵抗变化。
    """

    k = math.e

    resistance = abs(valence) * inertia_factor
    x = impact_strength - resistance
    return 1 / (1 + math.exp(-k * x))