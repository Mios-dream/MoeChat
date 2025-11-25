# emotion/compute_arousal_permission_factor.py

def compute_arousal_permission_factor(arousal: float, k: float = 1.5) -> float:
    """
    计算唤醒度（Arousal）的“许可因子”。
    当唤醒度接近0.5时允许较大变化，在两端（0或1）则抑制变化。
    """
    permission = (1 - abs(arousal - 0.5)) ** k
    return max(0, permission)
