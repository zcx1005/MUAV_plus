import numpy as np
from collections import defaultdict
from env_multi_uav import ACTIONS, HEADINGS_DEG


class QLearner:
    """
    独立 Q-learning 智能体（每架 UAV 一个实例）。

    状态编码 = (3x3局部patch哈希, 航向索引)
    - 3x3 patch：以 UAV 当前位置为中心的 3×3 局部覆盖状态
    - 航向索引：当前飞行方向 (0~7)

    使用字典型 Q 表，按需创建条目，避免预分配巨大数组。
    状态空间大小 = 3^9 × 8 = 157,464（use_heading=True 时）
                 = 3^9 = 19,683（use_heading=False 时）

    参数:
        n_actions   : int   — 动作空间大小（8个方向）
        alpha       : float — 学习率，控制 Q 值更新步长，越大学得越快但越不稳定
        gamma       : float — 折扣因子，越接近1越重视未来奖励
        epsilon     : float — ε-greedy 探索率，训练时会从外部动态设置
        use_heading : bool  — 是否将航向纳入状态编码
    """

    def __init__(self, n_actions, alpha=0.1, gamma=0.95, epsilon=0.2,
                 use_heading=True):
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.use_heading = use_heading

        # 字典型 Q 表：键 = 状态元组，值 = 长度为 n_actions 的数组
        # 首次访问某状态时自动初始化为全零
        self.Q = defaultdict(lambda: np.zeros(n_actions, dtype=np.float32))

    def _patch_hash(self, i, j, S):
        """
        将以 (i,j) 为中心的 3×3 局部覆盖状态编码为一个整数。
        每个格子的 S 值映射：-1(ROI外)→0, 0(未覆盖)→1, 1(已覆盖)→2
        采用三进制编码，共 3^9 = 19683 种可能的 patch 模式。
        出界的格子视为 ROI 外（-1）。

        参数:
            i, j : 中心格子的行列坐标
            S    : 当前覆盖状态矩阵
        返回:
            idx  : 整数哈希值 [0, 19682]
        """
        n_rows, n_cols = S.shape
        idx = 0
        factor = 1
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                ni, nj = i + di, j + dj
                if 0 <= ni < n_rows and 0 <= nj < n_cols:
                    v = S[ni, nj]
                else:
                    v = -1  # 出界视为 ROI 外
                code = 0 if v == -1 else (1 if v == 0 else 2)
                idx += code * factor
                factor *= 3
        return idx

    def encode_state(self, i, j, heading_idx, S, nearest_dist=None):
        """
        综合局部 patch + 航向 + 最近 UAV 距离，生成完整的离散状态标识。

        参数:
            i, j         : UAV 当前行列坐标
            heading_idx  : 当前航向索引 (0~7)
            S            : 覆盖状态矩阵
            nearest_dist : 到最近其他 UAV 的曼哈顿距离（可选）
        返回:
            state_tuple : 用作 Q 表键的元组
        """
        ph = self._patch_hash(i, j, S)
        # 距离分桶：0=紧邻(1~2格), 1=近(3~5格), 2=远(>5格)
        if nearest_dist is not None:
            if nearest_dist <= 2:
                dist_bin = 0
            elif nearest_dist <= 5:
                dist_bin = 1
            else:
                dist_bin = 2
        else:
            dist_bin = None

        if self.use_heading:
            if dist_bin is not None:
                return (ph, heading_idx, dist_bin)
            return (ph, heading_idx)
        if dist_bin is not None:
            return (ph, dist_bin)
        return (ph,)

    def select_action(self, s, i, j, allowed_actions, S):
        """
        ε-greedy 动作选择。
        探索时优先选择通往未覆盖格子的动作（加速覆盖）。
        利用时在允许动作中选 Q 值最大的。

        参数:
            s               : 当前状态（encode_state 的返回值）
            i, j            : UAV 当前行列坐标
            allowed_actions : 当前合法动作列表
            S               : 覆盖状态矩阵
        返回:
            action_id : 选中的动作索引
        """
        if np.random.rand() < self.epsilon:
            # 探索：优先选择能到达未覆盖格子的动作
            uncovered = []
            for a in allowed_actions:
                dr, dc = ACTIONS[a]
                ni, nj = i + dr, j + dc
                if 0 <= ni < S.shape[0] and 0 <= nj < S.shape[1]:
                    if S[ni, nj] == 0:  # 未覆盖的 ROI 格子
                        uncovered.append(a)
            if uncovered:
                return int(np.random.choice(uncovered))
            return int(np.random.choice(allowed_actions))

        # 利用：在允许动作中选 Q 值最大的
        q_vals = self.Q[s]
        best_val = -np.inf
        best_a = allowed_actions[0]
        for a in allowed_actions:
            if q_vals[a] > best_val:
                best_val = q_vals[a]
                best_a = a
        return best_a

    def update(self, s, a, r, s_next, done):
        """
        Q-learning 更新规则：
        Q(s,a) ← Q(s,a) + α * [r + γ * max_a' Q(s',a') - Q(s,a)]

        参数:
            s      : 当前状态
            a      : 执行的动作
            r      : 获得的即时奖励
            s_next : 转移到的下一状态
            done   : 是否为终止状态（全覆盖完成）
        """
        q_old = self.Q[s][a]
        if done:
            target = r  # 终止状态没有未来奖励
        else:
            target = r + self.gamma * np.max(self.Q[s_next])
        self.Q[s][a] = q_old + self.alpha * (target - q_old)
