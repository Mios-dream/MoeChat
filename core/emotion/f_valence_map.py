# emotion/f_valence_map.py

def f_valence_map(valence: float) -> float:
    """
    计算负向情绪的绝对值映射。
    如果 valence 为负，返回其绝对值；否则返回0。
    """
    if valence < 0:
        return abs(valence)
    return 0.0
