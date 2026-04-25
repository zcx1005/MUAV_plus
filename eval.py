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
    同时计算每架 UAV 的有效覆盖格子数/占比和冗余率。

    有效覆盖定义：
        按时间顺序遍历所有 UAV 的路径，每个 ROI 格子只归属于第一个到达的 UAV。
        走到别人已覆盖的格子或自己已覆盖的格子都算冗余。

    冗余率定义：
        该 UAV 的冗余步数（重复走自己的 + 走别人已覆盖的）/ 该 UAV 在 ROI 内的总步数。

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
            cover_cells_each    : List[int]   — 每架 UAV 有效覆盖的 ROI 格子数（不重叠）
            cover_ratio_each    : List[float] — 每架 UAV 有效覆盖占 ROI 总数的比例
            redundant_steps_each: List[int]   — 每架 UAV 的冗余步数
            redundant_ratio_each: List[float] — 每架 UAV 的冗余率
            reason              : str         — 仅在 invalid 时存在，说明失败原因
    """
    ok, coverage_ratio, visited_sets = check_full_coverage(paths, S0)
    roi_total = int(np.sum(S0 != -1))
    n_uav = len(paths)

    # 按时间顺序确定每个格子的归属（第一个到达的 UAV 拥有该格子）
    # global_owner[cell] = 第一个到达该 cell 的 uav_id
    global_owner = {}
    # 同时统计每架 UAV 的冗余步数
    effective_cells_each = [set() for _ in range(n_uav)]
    redundant_steps_each = [0] * n_uav
    roi_steps_each = [0] * n_uav  # 每架 UAV 在 ROI 内的总步数

    # 按时间步遍历（所有 UAV 同步行动，step 0 是起点）
    max_path_len = max(len(p) for p in paths) if paths else 0
    for step in range(max_path_len):
        for uav_id in range(n_uav):
            if step >= len(paths[uav_id]):
                continue
            i, j = paths[uav_id][step]
            if S0[i, j] == -1:
                continue  # ROI 外不统计
            roi_steps_each[uav_id] += 1
            cell = (i, j)
            if cell not in global_owner:
                # 首次被任何 UAV 到达 → 归属该 UAV
                global_owner[cell] = uav_id
                effective_cells_each[uav_id].add(cell)
            else:
                # 已被覆盖（自己或别人）→ 冗余
                redundant_steps_each[uav_id] += 1

    cover_cells_each = [len(s) for s in effective_cells_each]
    cover_ratio_each = [c / roi_total if roi_total > 0 else 0.0
                        for c in cover_cells_each]

    # 冗余率 = 冗余步数 / 该 UAV 在 ROI 内的总步数
    redundant_ratio_each = [
        redundant_steps_each[u] / roi_steps_each[u]
        if roi_steps_each[u] > 0 else 0.0
        for u in range(n_uav)
    ]

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
