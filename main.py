# main.py
# 多 UAV 覆盖路径规划 — 独立 Q-learning 训练主程序

import json
import os
import glob
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")  # 保存图片不弹窗
import matplotlib.pyplot as plt
# 解决中文显示问题
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示异常
from map_builder import build_map_from_poly
from env_multi_uav import MultiUAVCoverageEnv, ACTIONS
from QLearner import QLearner
from eval import evaluate_episode

# ===== 基本参数 =====
POLY_FILE = "park.poly"   # 多边形区域文件路径
CELL_SIZE = 50.0           # 栅格大小（米）
N_UAV = 2                  # UAV 数量
NUM_EPISODES = 10000        # 训练总轮数
MAX_STEPS_PER_EP = 1500    # 每轮最大步数（防止死循环）


def make_run_dir(n_episodes, n_uav):
    """
    创建本次训练的输出目录。
    命名格式：runs/YYYYMMDD_序号_episodes数_uav数量
    序号从 1 开始递增，用于区分同一天的多次训练。

    参数:
        n_episodes : int — 训练轮数（用于目录名）
        n_uav      : int — UAV 数量（用于目录名）
    返回:
        out_dir    : str — 创建好的目录路径
    """
    today = datetime.now().strftime("%Y%m%d")
    pattern = os.path.join("runs", f"{today}_*")
    existing = glob.glob(pattern)
    seq = len(existing) + 1
    name = f"{today}_{seq}_{n_episodes}ep_{n_uav}uav"
    out_dir = os.path.join("runs", name)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def plot_best_paths(polygon_rot, rows_list, cols_list, best_paths, cell_size,
                    save_path=None, show=False):
    """
    绘制最优覆盖路径图。

    参数:
        polygon_rot : Polygon    — 旋转后的 ROI 多边形（用于画边界）
        rows_list   : np.ndarray — 每行对应的 y 坐标
        cols_list   : np.ndarray — 每列对应的 x 坐标
        best_paths  : list       — 每架 UAV 的最优路径 [(i,j), ...]
        cell_size   : float      — 栅格大小
        save_path   : str        — 保存路径（None 则不保存）
        show        : bool       — 是否弹窗显示
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    minx, maxx = cols_list[0], cols_list[-1] + cell_size
    miny, maxy = rows_list[0], rows_list[-1] + cell_size

    # 画网格线
    for x in cols_list:
        ax.axvline(x, linestyle="--", linewidth=0.3, color="gray")
    ax.axvline(maxx, linestyle="--", linewidth=0.3, color="gray")
    for y in rows_list:
        ax.axhline(y, linestyle="--", linewidth=0.3, color="gray")
    ax.axhline(maxy, linestyle="--", linewidth=0.3, color="gray")

    # 画 ROI 边界
    x_poly, y_poly = polygon_rot.exterior.xy
    ax.plot(x_poly, y_poly, "k-", linewidth=1.5, label="ROI")

    # 画每架 UAV 的路径
    colors = ["r", "g", "b", "m", "c", "y"]
    for uav_id, path in enumerate(best_paths):
        if not path:
            continue
        xs = [cols_list[j] + cell_size / 2.0 for (i, j) in path]
        ys = [rows_list[i] + cell_size / 2.0 for (i, j) in path]
        c = colors[uav_id % len(colors)]
        ax.plot(xs, ys, marker="o", linewidth=1.0, markersize=3, color=c,
                label=f"UAV {uav_id}")
        # 标记终点
        ax.scatter(xs[-1], ys[-1], s=120, marker="X", color=c, zorder=5)
        ax.text(xs[-1] + cell_size * 0.1, ys[-1] + cell_size * 0.1,
                f"End{uav_id}", color=c, fontsize=10, weight="bold")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Best multi-UAV coverage paths")
    ax.legend()
    ax.grid(False)
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_reward_convergence(episode_rewards, save_path=None, win=50):
    """
    绘制奖励收敛曲线图。

    参数:
        episode_rewards : list[float] — 每轮的总奖励
        save_path       : str         — 保存路径
        win             : int         — 移动平均窗口大小
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    eps = np.arange(len(episode_rewards))
    ax.plot(eps, episode_rewards, alpha=0.25, label="Reward")
    ma = _moving_avg(episode_rewards, win)
    ax.plot(eps, ma, label=f"MA({win})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("Reward Convergence")
    ax.legend()
    ax.grid(True)
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_steps_history(episode_steps, save_path=None, win=50):
    """
    绘制每轮步数变化图（含移动平均和 best-so-far 线）。

    参数:
        episode_steps : list[int] — 每轮的步数（valid=实际步数，invalid=MAX_STEPS）
        save_path     : str       — 保存路径
        win           : int       — 移动平均窗口大小
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    eps = np.arange(len(episode_steps))
    ax.plot(eps, episode_steps, alpha=0.25, label="Steps")
    ma = _moving_avg(episode_steps, win)
    ax.plot(eps, ma, label=f"MA({win})")

    # best-so-far 线：记录到当前为止的最小步数
    best_sf = np.full(len(episode_steps), np.nan)
    cur_best = np.inf
    for t in range(len(episode_steps)):
        v = episode_steps[t]
        if not np.isnan(v) and v < cur_best:
            cur_best = v
        best_sf[t] = cur_best if cur_best < np.inf else np.nan
    ax.plot(eps, best_sf, label="Best-so-far", color="red")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Steps (valid=实际, invalid=MAX)")
    ax.set_title("Steps per Episode")
    ax.legend()
    ax.grid(True)
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _moving_avg(x, win=50):
    """
    计算移动平均（忽略 NaN 值）。

    参数:
        x   : array-like — 输入序列
        win : int        — 窗口大小
    返回:
        y   : np.ndarray — 移动平均结果
    """
    x = np.asarray(x, dtype=float)
    y = np.full_like(x, np.nan)
    for t in range(len(x)):
        lo = max(0, t - win + 1)
        seg = x[lo:t + 1]
        seg = seg[~np.isnan(seg)]
        if len(seg) > 0:
            y[t] = seg.mean()
    return y


def main():
    # ===== 1) 构建地图 =====
    S0, rows_list, cols_list, inside_cells, out_pos, polygon_rot = \
        build_map_from_poly(POLY_FILE, CELL_SIZE)

    n_rows, n_cols = S0.shape
    roi_total = int(np.sum(S0 != -1))
    print(f"S0 形状: {S0.shape}, ROI 格子数: {roi_total}, ROI 外格子数: {int(np.sum(S0 == -1))}")

    # ===== 2) 初始化环境 =====
    env = MultiUAVCoverageEnv(S0=S0, n_uav=N_UAV, inside_cells=inside_cells)

    # ===== 3) 初始化 Q-learner（每架 UAV 一个独立 Q 表）=====
    n_actions = len(ACTIONS)
    q_learners = [
        QLearner(
            n_actions=n_actions,
            alpha=0.1,          # 学习率
            gamma=0.95,         # 折扣因子
            epsilon=0.2,        # 占位值，每轮开始时会被 epsilon 衰减覆盖
            use_heading=True,   # 状态编码中包含航向
        )
        for _ in range(N_UAV)
    ]

    # epsilon 衰减参数：从 eps_start 指数衰减到 eps_end
    eps_start = 0.30   # 初始探索率
    eps_end = 0.02     # 最终探索率
    eps_decay = 0.9995  # 每轮衰减系数

    # ===== 训练过程追踪变量 =====
    best_steps = None   # 最优（最少）步数
    best_paths = None   # 最优路径
    best_info = None    # 最优结果的评估信息

    episode_rewards = []      # 每轮总奖励
    episode_steps_list = []   # 每轮步数（valid=实际步数，invalid=MAX_STEPS_PER_EP）
    train_lines = []          # 训练日志行

    # 创建输出目录
    out_dir = make_run_dir(NUM_EPISODES, N_UAV)
    print(f"输出目录: {out_dir}")

    # ===== 训练循环 =====
    for ep in range(NUM_EPISODES):
        # 设置当前轮的 epsilon（所有 UAV 统一）
        eps = max(eps_end, eps_start * (eps_decay ** ep))
        for u in range(N_UAV):
            q_learners[u].epsilon = eps

        # 重置环境
        obs = env.reset()
        done = False

        # 记录每架 UAV 的完整路径（包含所有步骤，含重复访问）
        paths = [[(env.uav_cells[u][0], env.uav_cells[u][1])] for u in range(N_UAV)]

        step_count = 0
        total_reward_ep = 0.0

        while (not done) and (step_count < MAX_STEPS_PER_EP):
            actions = []
            states = []

            # 每架 UAV 用自己的 Q 表选动作
            for uav_id in range(N_UAV):
                learner = q_learners[uav_id]
                i, j = env.uav_cells[uav_id]
                heading_idx = env.uav_heading[uav_id]
                s = learner.encode_state(i, j, heading_idx, env.S)
                states.append(s)

                allowed = env.get_legal_actions_local(uav_id)
                a = learner.select_action(s, i, j, allowed, env.S)
                actions.append(a)

            # 所有 UAV 同时执行动作
            obs, rewards, done, info = env.step_joint(actions)

            # 各自更新自己的 Q 表
            for uav_id in range(N_UAV):
                learner = q_learners[uav_id]
                s = states[uav_id]
                a = actions[uav_id]
                r = rewards[uav_id]

                i2, j2 = env.uav_cells[uav_id]
                heading_idx2 = env.uav_heading[uav_id]
                s_next = learner.encode_state(i2, j2, heading_idx2, env.S)
                learner.update(s, a, r, s_next, done)

                # 记录路径（包含所有位置，即使重复）
                paths[uav_id].append((i2, j2))
                total_reward_ep += r

            step_count += 1

        # ===== 本轮评估 =====
        valid, einfo = evaluate_episode(paths, S0, step_count)

        episode_rewards.append(total_reward_ep)
        # valid 的轮次记录实际步数，invalid 的记录 MAX（方便画图看趋势）
        episode_steps_list.append(step_count if valid else MAX_STEPS_PER_EP)

        # 构造日志行
        if valid:
            cov_parts = " | ".join(
                f"UAV{u}:{einfo['cover_cells_each'][u]}({einfo['cover_ratio_each'][u]*100:.1f}%) "
                f"冗余={einfo['redundant_ratio_each'][u]*100:.1f}%"
                for u in range(N_UAV)
            )
            line = (f"[EP {ep:04d}] VALID steps={step_count:4d} reward={total_reward_ep:8.2f} "
                    f"| {cov_parts}")

            # 更新最优记录（以最少步数为标准）
            if best_steps is None or step_count < best_steps:
                best_steps = step_count
                best_paths = [list(p) for p in paths]
                best_info = einfo
                line += " ★BEST"
        else:
            line = (f"[EP {ep:04d}] INVALID steps={step_count:4d} reward={total_reward_ep:8.2f} "
                    f"覆盖率={einfo['coverage_ratio']*100:.1f}%")

        print(line)
        train_lines.append(line)

    # ===== 保存结果 =====

    # 1) 训练过程日志 → train_record.txt
    with open(os.path.join(out_dir, "train_record.txt"), "w", encoding="utf-8") as f:
        f.write(f"S0 形状: {S0.shape}, ROI 格子数: {roi_total}\n")
        f.write(f"N_UAV={N_UAV}, NUM_EPISODES={NUM_EPISODES}, MAX_STEPS={MAX_STEPS_PER_EP}\n")
        f.write(f"eps_start={eps_start}, eps_end={eps_end}, eps_decay={eps_decay}\n\n")
        for line in train_lines:
            f.write(line + "\n")

    # 2) 最优结果 → best_result.json
    if best_info is not None:
        payload = {
            "best_steps": best_steps,
            "roi_total_cells": roi_total,
            "n_uav": N_UAV,
            "cover_cells_each": best_info["cover_cells_each"],
            "cover_ratio_each": best_info["cover_ratio_each"],
            "redundant_steps_each": best_info["redundant_steps_each"],
            "redundant_ratio_each": best_info["redundant_ratio_each"],
        }
        with open(os.path.join(out_dir, "best_result.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    # 3) 收敛数据 → convergence_result.json
    conv = {
        "episode_rewards": [float(x) for x in episode_rewards],
        "episode_steps": [int(x) for x in episode_steps_list],
    }
    with open(os.path.join(out_dir, "convergence_result.json"), "w", encoding="utf-8") as f:
        json.dump(conv, f, ensure_ascii=False)

    # 4) 最优路径 → best_paths.csv
    if best_paths is not None:
        csv_path = os.path.join(out_dir, "best_paths.csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("uav_id,step,i,j,x,y\n")
            for uav_id, path in enumerate(best_paths):
                for step, (i, j) in enumerate(path):
                    x = cols_list[j] + CELL_SIZE / 2.0
                    y = rows_list[i] + CELL_SIZE / 2.0
                    f.write(f"{uav_id},{step},{i},{j},{x:.2f},{y:.2f}\n")

    # 5) 三张图
    if best_paths is not None:
        plot_best_paths(polygon_rot, rows_list, cols_list, best_paths, CELL_SIZE,
                        save_path=os.path.join(out_dir, "best_paths.png"))

    plot_reward_convergence(episode_rewards,
                            save_path=os.path.join(out_dir, "reward_convergence.png"))
    plot_steps_history(episode_steps_list,
                       save_path=os.path.join(out_dir, "steps_history.png"))

    # ===== 打印训练总结 =====
    print(f"\n====== 训练结束 ======")
    if best_info is not None:
        print(f"最优步数: {best_steps}")
        for u in range(N_UAV):
            print(f"  UAV{u}: 覆盖={best_info['cover_cells_each'][u]}/{roi_total} "
                  f"({best_info['cover_ratio_each'][u]*100:.1f}%), "
                  f"冗余率={best_info['redundant_ratio_each'][u]*100:.1f}%")
    else:
        print("未找到任何完全覆盖的轨迹。")
    print(f"结果已保存到: {out_dir}")


if __name__ == "__main__":
    main()
