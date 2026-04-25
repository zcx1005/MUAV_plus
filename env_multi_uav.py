import numpy as np

# 8 个动作：在 S 矩阵上的 (行偏移dr, 列偏移dc)
# 索引 0~7 分别对应 East, SE, South, SW, West, NW, North, NE
ACTIONS = [
    (0, 1),    # 0: East  — 向右
    (-1, 1),   # 1: SE    — 右下
    (-1, 0),   # 2: South — 向下
    (-1, -1),  # 3: SW    — 左下
    (0, -1),   # 4: West  — 向左
    (1, -1),   # 5: NW    — 左上
    (1, 0),    # 6: North — 向上
    (1, 1),    # 7: NE    — 右上
]

# 每个动作对应的航向角度（度），与 ACTIONS 索引一一对应
HEADINGS_DEG = [0, 45, 90, 135, 180, -135, -90, -45]


class MultiUAVCoverageEnv:
    """
    多 UAV 覆盖路径规划环境。

    核心状态矩阵 S:
        -1 : ROI 外（非任务区，可以飞越但不计覆盖）
         0 : ROI 内、未覆盖 (Mn)
         1 : ROI 内、已覆盖 (Ms)

    属性:
        S0          : 初始地图矩阵（每个 episode 开始时复制一份）
        n_rows      : 地图行数
        n_cols      : 地图列数
        n_uav       : UAV 数量
        inside_cells: ROI 内所有格子的 (i,j) 列表
        uav_cells   : 每架 UAV 当前所在格子 [(row, col), ...]
        uav_heading : 每架 UAV 当前航向索引 [0~7, ...]
        owner       : 格子归属矩阵，-1 表示无人占有，否则为 UAV 编号
        visited     : 形状 (n_uav, n_rows, n_cols) 的布尔数组，记录每架 UAV 访问过的格子
        in_roi      : 每架 UAV 是否已经进入过 ROI（一旦进入就不允许再飞出）
    """

    def __init__(self, S0, n_uav, inside_cells, rng=None):
        """
        参数:
            S0           : np.ndarray — 初始覆盖状态矩阵
            n_uav        : int        — UAV 数量
            inside_cells : list       — ROI 内格子坐标列表
            rng          : RandomState — 随机数生成器（可选）
        """
        self.S0 = S0
        self.n_rows, self.n_cols = S0.shape
        self.n_uav = n_uav
        self.inside_cells = inside_cells
        self.rng = np.random.RandomState(0) if rng is None else rng
        self.owner = None
        self.in_roi = None
        self.uav_cells = None
        self.uav_heading = None
        self.visited = None
        self.S = None
        self.straight_streak = np.zeros(self.n_uav, dtype=int)

    def reset(self):
        """
        重置环境到初始状态（每个 episode 开始时调用）。
        所有 UAV 从左下角 (0,0) 起飞（通常在 ROI 外）。

        返回:
            obs : dict — 初始观测
        """
        self.S = self.S0.copy()
        self.owner = -np.ones_like(self.S, dtype=int)
        self.visited = np.zeros((self.n_uav, self.n_rows, self.n_cols), dtype=bool)
        self.in_roi = np.zeros(self.n_uav, dtype=bool)
        self.straight_streak = np.zeros(self.n_uav, dtype=int)

        # 所有 UAV 从 (0,0) 起飞，初始航向为 East
        start_i, start_j = 0, 0
        self.uav_cells = [(start_i, start_j) for _ in range(self.n_uav)]
        self.uav_heading = [0 for _ in range(self.n_uav)]

        # 如果起点恰好在 ROI 内，标记为已覆盖
        if self.S[start_i, start_j] == 0:
            self.S[start_i, start_j] = 1

        return self._get_obs()

    def get_legal_actions_local(self, uav_id):
        """
        获取指定 UAV 在当前状态下的合法动作列表。
        采用三级兜底策略：
          1) 严格模式：不出界、不出ROI（已进入后）、不踩别人的格子、不重复走自己的格子
          2) 兜底一：允许重复走自己的格子（但仍不踩别人的）
          3) 兜底二：只要不出界就行

        参数:
            uav_id : int — UAV 编号
        返回:
            legal  : list[int] — 合法动作索引列表
        """
        i, j = self.uav_cells[uav_id]
        legal = []

        # 第一级：严格约束
        for a, (dr, dc) in enumerate(ACTIONS):
            ni, nj = i + dr, j + dc
            # 不出界
            if not (0 <= ni < self.n_rows and 0 <= nj < self.n_cols):
                continue
            # 已进入 ROI 后不允许飞到 ROI 外
            if self.in_roi[uav_id] and self.S[ni, nj] == -1:
                continue
            # ROI 内的格子：检查归属和重复访问
            if self.S[ni, nj] != -1:
                owner = self.owner[ni, nj]
                if owner != -1 and owner != uav_id:
                    continue  # 别人的格子不能踩
                if self.visited[uav_id, ni, nj]:
                    continue  # 自己走过的格子不重复
            legal.append(a)

        if legal:
            return legal

        # 第二级兜底：允许走自己访问过的格子
        for a, (dr, dc) in enumerate(ACTIONS):
            ni, nj = i + dr, j + dc
            if not (0 <= ni < self.n_rows and 0 <= nj < self.n_cols):
                continue
            if self.in_roi[uav_id] and self.S[ni, nj] == -1:
                continue
            # if self.S[ni, nj] != -1:   允许穿越
            #     owner = self.owner[ni, nj]
            #     if owner != -1 and owner != uav_id:
            #         continue  # 仍然不踩别人的格子
            legal.append(a)

        if legal:
            return legal

        # 第三级兜底：只要不出界
        for a, (dr, dc) in enumerate(ACTIONS):
            ni, nj = i + dr, j + dc
            if 0 <= ni < self.n_rows and 0 <= nj < self.n_cols:
                legal.append(a)

        return legal if legal else [0]

    def _ang_diff_deg(self, a_deg, b_deg):
        """计算两个角度之间的最小无符号差值（0~180度）"""
        d = abs(a_deg - b_deg) % 360
        return d if d <= 180 else 360 - d

    def _move_to_heading_idx(self, di, dj):
        """
        将位移向量 (di, dj) 映射回动作/航向索引。
        如果没有移动（di==0 且 dj==0），返回 None。
        """
        if di == 0 and dj == 0:
            return None
        for idx, (dr, dc) in enumerate(ACTIONS):
            if dr == di and dc == dj:
                return idx
        return None

    def step_joint(self, actions):
        """
        所有 UAV 同时执行一步动作。

        处理流程：
          1) 边界检查 → 得到候选位置
          2) 节点碰撞检测（多架 UAV 想去同一格子）
          3) 路径交叉检测（两架 UAV 的移动线段相交）
          4) 根据最终位置更新覆盖状态、归属、奖励
          5) 判断是否全覆盖完成

        参数:
            actions : list[int] — 每架 UAV 的动作索引，长度 = n_uav

        返回:
            obs     : dict        — 新的观测
            rewards : list[float] — 每架 UAV 的即时奖励
            done    : bool        — 是否全覆盖完成
            info    : dict        — 预留信息（当前为空字典）

        奖励设计:
            +10.0  : 首次进入 ROI
            +1.0   : 覆盖一个新的 ROI 格子
            +50.0  : 全覆盖完成（终止奖励）
            +0.03  : 转弯但覆盖了新格子（小补偿）
            -0.01  : 在 ROI 外飞行
            -0.02  : 每步的移动代价（对角线 ×√2）
            -0.1   : 越界/原地不动
            -0.10  : 转弯惩罚（按角度比例）
            -0.2   : 重复走自己已访问的 ROI 格子
            -0.3   : 已进入 ROI 后试图飞出
            -0.5   : 节点碰撞 / 路径交叉 / 踩别人的格子
        """
        assert len(actions) == self.n_uav
        old_cells = list(self.uav_cells)       # 本步开始时各 UAV 的位置
        prev_headings = list(self.uav_heading)  # 本步开始时各 UAV 的航向
        cand_cells = []                         # 候选下一位置
        rewards = [0.0] * self.n_uav

        # ===== 1) 边界检查，得到候选位置 =====
        for uav_id, act in enumerate(actions):
            dr, dc = ACTIONS[act]
            i, j = old_cells[uav_id]
            ni, nj = i + dr, j + dc
            if not (0 <= ni < self.n_rows and 0 <= nj < self.n_cols):
                ni, nj = i, j       # 越界则原地不动
                rewards[uav_id] -= 0.1
            cand_cells.append((ni, nj))

        # ===== 2) 节点碰撞检测 =====
        # 只对"与 ROI 有关"的 UAV 做碰撞检测（都在 ROI 外时不检测）
        from collections import defaultdict
        pos2ids = defaultdict(list)
        for uav_id, (ni, nj) in enumerate(cand_cells):
            if self.in_roi[uav_id] or self.S[ni, nj] != -1:
                pos2ids[(ni, nj)].append(uav_id)

        collided_ids = set()
        for pos, ids in pos2ids.items():
            if len(ids) > 1:
                # 多架 UAV 想去同一格子 → 全部回到原位 + 碰撞惩罚
                for uav_id in ids:
                    cand_cells[uav_id] = old_cells[uav_id]
                    rewards[uav_id] -= 0.5
                    collided_ids.add(uav_id)

        # ===== 3) 路径交叉检测 =====
        def segments_intersect_strict(p1, p2, p3, p4):
            """检测线段 p1-p2 与 p3-p4 是否严格相交（不含端点重合）"""
            p1, p2 = np.array(p1, float), np.array(p2, float)
            p3, p4 = np.array(p3, float), np.array(p4, float)
            def orient(a, b, c): return np.cross(b - a, c - a)
            o1, o2 = orient(p1, p2, p3), orient(p1, p2, p4)
            o3, o4 = orient(p3, p4, p1), orient(p3, p4, p2)
            return (o1 * o2 < 0) and (o3 * o4 < 0)

        for u1 in range(self.n_uav):
            for u2 in range(u1 + 1, self.n_uav):
                if u1 in collided_ids or u2 in collided_ids:
                    continue  # 已在节点碰撞中处理过
                A, B = old_cells[u1], cand_cells[u1]  # UAV u1 的移动线段
                C, D = old_cells[u2], cand_cells[u2]  # UAV u2 的移动线段
                if A == B and C == D:
                    continue  # 两边都原地不动，无需检测
                # 如果两架 UAV 都还在 ROI 外且下一步也在 ROI 外，跳过
                both_outside = (
                    (not self.in_roi[u1]) and (self.S[B[0], B[1]] == -1) and
                    (not self.in_roi[u2]) and (self.S[D[0], D[1]] == -1)
                )
                if both_outside:
                    continue
                if segments_intersect_strict(A, B, C, D):
                    # 路径交叉 → 两架都回到原位 + 惩罚
                    cand_cells[u1] = A
                    cand_cells[u2] = C
                    rewards[u1] -= 0.5
                    rewards[u2] -= 0.5
                    collided_ids.add(u1)
                    collided_ids.add(u2)

        # ===== 4) 根据最终位置更新状态和奖励 =====
        self.uav_cells = list(cand_cells)

        for uav_id, (ni, nj) in enumerate(self.uav_cells):
            i_old, j_old = old_cells[uav_id]

            # --- ROI 外的格子 ---
            if self.S[ni, nj] == -1:
                if self.in_roi[uav_id]:
                    # 已进入 ROI 后不允许飞出 → 回到原位 + 惩罚
                    self.uav_cells[uav_id] = (i_old, j_old)
                    rewards[uav_id] -= 0.3
                else:
                    # 还没进 ROI，在外面飞 → 小惩罚促使尽快进入
                    rewards[uav_id] -= 0.01
                continue  # ROI 外不更新 owner/visited

            # --- 首次进入 ROI → 奖励 ---
            if not self.in_roi[uav_id]:
                rewards[uav_id] += 5.0
            self.in_roi[uav_id] = True

            # --- 检查格子归属冲突 ---
            cell_owner = self.owner[ni, nj]
            new_cover = False
            if cell_owner != -1 and cell_owner != uav_id:
                # 别人的格子 → 回到原位 + 惩罚 软约束版本，跨区惩罚
                # self.uav_cells[uav_id] = (i_old, j_old)
                rewards[uav_id] -= 0.3

                # continue

            # --- 覆盖逻辑 ---
            # new_cover = False
            elif self.visited[uav_id, ni, nj]:
                # 重复走自己已访问的格子 → 惩罚
                rewards[uav_id] -= 0.3
            else:
                # 首次访问 → 标记归属
                self.visited[uav_id, ni, nj] = True
                self.owner[ni, nj] = uav_id
                if self.S[ni, nj] == 0:
                    # 覆盖新的 ROI 格子 → 奖励
                    self.S[ni, nj] = 1
                    rewards[uav_id] += 1.0
                    new_cover = True

            # --- 航向更新和移动代价 ---
            fi, fj = self.uav_cells[uav_id]
            di, dj = fi - i_old, fj - j_old
            moved = (di != 0 or dj != 0)
            new_h = self._move_to_heading_idx(di, dj)
            prev_h = prev_headings[uav_id]

            if moved:
                # 移动代价：对角线步长 ×√2
                is_diag = (abs(di) == 1 and abs(dj) == 1)
                rewards[uav_id] -= 0.1 * (np.sqrt(2) if is_diag else 1.0)

                # 转弯惩罚：按角度差与90度的比例
                if new_h is not None:
                    ddeg = self._ang_diff_deg(HEADINGS_DEG[prev_h], HEADINGS_DEG[new_h])
                    rewards[uav_id] -= 0.10 * (ddeg / 90.0)
                    # 如果转弯带来了新覆盖，给一点小补偿
                    if new_cover and ddeg > 0:
                        rewards[uav_id] += 0.03
                    self.uav_heading[uav_id] = new_h

                    # 直飞奖励
                    if new_cover and new_h == prev_h:
                        self.straight_streak[uav_id] += 1
                        streak = min(self.straight_streak[uav_id], 4)  # 上限4
                        rewards[uav_id] += 0.1 * streak
                    else:
                        self.straight_streak[uav_id] = 0
            else:
                # 原地不动（被碰撞/越界回退导致）→ 轻微惩罚
                rewards[uav_id] -= 0.1
                self.uav_heading[uav_id] = prev_h

        # ===== 5) 判断是否全覆盖完成 =====
        done = not np.any(self.S == 0)
        if done:
            # 全覆盖完成 → 每架 UAV 获得终止奖励
            for uav_id in range(self.n_uav):
                rewards[uav_id] += 50.0

        return self._get_obs(), rewards, done, {}

    def _get_obs(self):
        """
        构造当前观测信息。

        返回:
            obs : dict — 包含：
                uav_cells     : 每架 UAV 的当前位置列表
                uav_heading   : 每架 UAV 的当前航向索引列表
                coverage_ratio: 当前 ROI 覆盖率
        """
        covered = np.sum(self.S == 1)
        total_mission = np.sum(self.S != -1)
        cov_ratio = covered / total_mission if total_mission > 0 else 1.0
        return {
            "uav_cells": list(self.uav_cells),
            "uav_heading": list(self.uav_heading),
            "coverage_ratio": cov_ratio,
        }
