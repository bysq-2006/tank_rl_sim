from __future__ import annotations


def segment_intersects_rect(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    rect: tuple[float, float, float, float],
) -> bool:
    """使用参数区间裁剪判断有限线段是否穿过轴对齐矩形。"""
    lower_t, upper_t = 0.0, 1.0
    for origin, delta, lower, upper in (
        (x0, x1 - x0, rect[0], rect[2]),
        (y0, y1 - y0, rect[1], rect[3]),
    ):
        if abs(delta) < 1e-12:
            if origin < lower or origin > upper:
                return False
            continue
        first, second = (lower - origin) / delta, (upper - origin) / delta
        lower_t = max(lower_t, min(first, second))
        upper_t = min(upper_t, max(first, second))
        if lower_t > upper_t:
            return False
    return True
