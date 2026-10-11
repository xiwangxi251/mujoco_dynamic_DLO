"""Summarize the two-goal feedback/servo/gravity causal probes."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/vertical_workspace/scripted_136_retest_20261004/pose_diagnosis"

if __name__ == "__main__":
    rows = sum([json.loads((OUT / f"{s}_results.json").read_text(encoding="utf-8"))
                for s in ("hierarchy", "full")], [])
    assert len(rows) == 24
    assert len({(r["solver"], r["intervention"], tuple(r["goal_xy_m"])) for r in rows}) == 24
    configs = ("original", "angular_only", "servo_only", "gravity_only", "gravity_servo", "all_three")
    labels = ("Original", "Angular gain = 10 only", "Kp x4 / Kv x2 only", "Gravity feedforward only",
              "Gravity + servo", "Gravity + servo + angular gain 10")
    matrix = np.zeros((6, 4))
    for col, (solver, goal) in enumerate((("hierarchy", [0.7, 0.0]), ("full", [0.7, 0.0]),
                                         ("hierarchy", [0.65, 0.3]), ("full", [0.65, 0.3]))):
        for index, config in enumerate(configs):
            row = next(r for r in rows if r["solver"] == solver and r["goal_xy_m"] == goal and r["intervention"] == config)
            assert abs(row["duration_s"] - 16) < 1e-8
            matrix[index, col] = row["final_pose_deg"]
    fig, ax = plt.subplots(figsize=(12, 7), layout="constrained")
    im = ax.imshow(matrix, cmap="magma", norm=LogNorm(vmin=0.4, vmax=50), aspect="auto")
    ax.set_xticks(range(4), ["Position-first JIK\nCenter", "Full 6D JIK\nCenter",
                            "Position-first JIK\nPositive Y", "Full 6D JIK\nPositive Y"])
    ax.set_yticks(range(6), labels)
    ax.tick_params(length=0)
    for y in range(6):
        for x in range(4):
            ax.text(x, y, f"{matrix[y, x]:.2f} deg", ha="center", va="center",
                    color="white" if matrix[y, x] < 12 else "#19283b", fontsize=13)
    ax.set_title("Measured final pose error after identical 16 s physical trials", pad=16, fontsize=15)
    fig.colorbar(im, ax=ax, label="Full pose error (deg, log scale)", shrink=0.85)
    fig.savefig(OUT / "pose_ablation.png", dpi=175, bbox_inches="tight")
    plt.close(fig)
    lines = ["# 版本1、3、6姿态误差因果诊断", "",
        "在录像中的两个共同目标，分别对位置优先JIK和完整6D JIK做6种控制配置，共24次独立实际物理试跑。每次16秒，合格后仍继续控制到相同时刻；保持原home、参考路径、摩擦、质量、碰撞和限速配置。", "",
        "原始版本1、6的全部关节轨迹，以及重力前馈＋伺服增强＋姿态增益10与版本3的已存轨迹前缀，逐元素最大差值均为0。", "",
        "| 改动 | 位置优先中心/° | 完整6D中心/° | 位置优先正Y/° | 完整6D正Y/° |", "|---|---:|---:|---:|---:|"]
    for label, values in zip(labels, matrix):
        lines.append("| " + label + " | " + " | ".join(f"{v:.3f}" for v in values) + " |")
    lines += ["", "主要证据：完整6D解算器保留原姿态增益6，仅添加重力前馈与kp×4/kv×2，中心误差43.896°→0.945°，正Y误差35.162°→0.858°，两点最后2秒均满足终点条件。仅提高姿态增益到10仍分别为19.421°与31.516°。", "",
        "位置执行器在限幅前近似输出kp(q_cmd−q)−kv*qdot_actual；q_cmd=q+0.11*qdot_JIK，因此任务纠正力矩随JIK反馈而变化。无重力前馈时，任务反馈同时承担抵消重力和摩擦的任务；更高kp增加相同纠正命令的驱动力，前馈在误差很小时仍提供重力支撑。姿态增益10进一步压低补偿重力与增强伺服后的残余误差。", "",
        "中心目标中，版本1/6在0.5秒已经有22.61°/14.31°姿态误差，此时机器人最低离地约81 mm；离地间隙首次小于0.2 mm分别在2.26/2.58秒。两版后来靠近joint7限位，故接触与限位是后续维持或放大偏差的因素，不能解释最初的姿态漂移。", "",
        "中心原始版本1的姿态零空间最弱模态传递约99.69%，原始版本6完整J的最弱模态传递约93.31%；原始两点均未触发实际速度守卫。这两个代表目标的数据不支持把数十度误差主要归于强奇异阻尼或速度守卫。", "",
        "结论仅覆盖这两个代表目标的因果对照；没有重新扫描改进后的版本6全工作范围，也未运行抓取状态机。", ""]
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT / "verification.json").write_text(json.dumps({
        "trials": 24, "unique": 24, "all_trials_16s": True,
        "baseline_and_version3_prefix_max_joint_difference": max(r.get("previous_joint_state_prefix_max_difference", 0) for r in rows),
        "max_torque_saturation_substeps": max(r["torque_saturation_substeps"] for r in rows),
        "max_velocity_guard_substeps": max(r["velocity_guard_substeps"] for r in rows),
    }, indent=2), encoding="utf-8")
    print(OUT / "README.md")
