import numpy as np


def check_full_coverage(paths, S0):
    """
    检查所有 UAV 路径的并集是否完全覆盖了 ROI。

    参数:
        paths : List[List[(i,j)]] — 每架 UAV 的完整路径（包含所有步骤的格子坐标）
        S0    : np.ndarray        — 初始地图矩阵（S0 != -1 的格子为 ROI）

    返回:
        ok             : bool       — 是否完全覆盖
        coverage_ratio : float      — 覆盖率 = 已覆盖ROI格子数 / ROI总格子数
        visited_sets   : List[set]  — 每架 UAV 在 ROI 内访问过的唯一格子集合
    """
    roi_cells = set(zip(*np.where(S0 != -1)))  # ROI 内所有格子的 (i,j) 集合
    visited_sets = []
    for path in paths:
        vset = set()
        for (i, j) in path:
            if S0[i, j] != -1:  # 只统计 ROI 内的格子
                vset.add((i, j))
        visited_sets.append(vset)

    union_roi = set().union(*visited_sets) if visited_sets else set()
    coverage_ratio = len(union_roi) / len(roi_cells) if roi_cells else 0.0
    ok = (union_roi == roi_cells)
    return ok, coverage_ratio, visited_sets


def evaluate_episode(paths, S0, total_steps):
    """
    评估一个 episode 的结果。
    主要指标：total_steps（越少越好，表示覆盖效率越高）。
    同时计算每架 UAV 的覆盖格子数/占比和冗余率。

    冗余率定义：
        该 UAV 在路径中重复访问 ROI 内已走过格子的步数 / 总 steps。
        因为所有 UAV 同步行动，每个 step 每台 UAV 都走一步，
        所以用总 steps 作为分母是合理的。

    参数:
        paths       : List[List[(i,j)]] — 每架 UAV 的完整路径
        S0          : np.ndarray        — 初始地图矩阵
        total_steps : int               — 该 episode 的总步数

    返回:
        valid : bool — 是否完全覆盖了 ROI
        info  : dict — 包含以下字段：
            valid               : bool        — 同上
            coverage_ratio      : float       — 覆盖率
            total_steps         : int         — 总步数
            roi_total_cells     : int         — ROI 总格子数
            cover_cells_each    : List[int]   — 每架 UAV 覆盖的 ROI 格子数
            cover_ratio_each    : List[float] — 每架 UAV 覆盖占 ROI 总数的比例
            redundant_steps_each: List[int]   — 每架 UAV 的冗余步数
            redundant_ratio_each: List[float] — 每架 UAV 的冗余率
            reason              : str         — 仅在 invalid 时存在，说明失败原因
    """
    ok, coverage_ratio, visited_sets = check_full_coverage(paths, S0)
    roi_total = int(np.sum(S0 != -1))

    # 每架 UAV 覆盖的唯一 ROI 格子数
    cover_cells_each = [len(vs) for vs in visited_sets]
    cover_ratio_each = [c / roi_total if roi_total > 0 else 0.0 for c in cover_cells_each]

    # 冗余统计：遍历每架 UAV 的路径，统计在 ROI 内重复访问已走过格子的次数
    redundant_steps_each = []
    for uav_id, path in enumerate(paths):
        seen = set()   # 该 UAV 已经访问过的 ROI 格子
        red = 0        # 冗余步数计数
        for (i, j) in path:
            if S0[i, j] != -1:  # 只看 ROI 内的格子
                if (i, j) in seen:
                    red += 1    # 重复访问 → 冗余
                else:
                    seen.add((i, j))
        redundant_steps_each.append(red)

    # 冗余率 = 冗余步数 / 总步数
    redundant_ratio_each = [r / total_steps if total_steps > 0 else 0.0
                            for r in redundant_steps_each]

    info = {
        "valid": ok,
        "coverage_ratio": coverage_ratio,
        "total_steps": total_steps,
        "roi_total_cells": roi_total,
        "cover_cells_each": cover_cells_each,
        "cover_ratio_each": cover_ratio_each,
        "redundant_steps_each": redundant_steps_each,
        "redundant_ratio_each": redundant_ratio_each,
    }
    if not ok:
        info["reason"] = "not_full_coverage"

    return ok, info
