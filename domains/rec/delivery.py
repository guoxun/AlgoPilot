# -*- coding: utf-8 -*-
"""delivery.py:端到端交付仿真(冻结文件,任何角色不许改)。

理念:**离线棘轮负责走得快,交付对齐负责走得对。**

离线 judge 验证的是"随机曝光分布下的排序质量"(AUC),但搜广推的最终交付是
"真实管线里的业务结果"。两者之间的 gap 是行业最贵的教训(离线在线一致性):
selection bias、马太效应、过拟合代理指标,在 AUC 上全部隐形。

本仿真把当前模型放进一个微缩交付管线,让"离线提升"接受真实世界的检验:

    会话流(固定 seed,确定性)
    ──► 每会话:一个用户 + 固定候选池 → 模型打分 top-K 曝光(曝光分布由模型自选)
    ──► 曝光物品按 prepare.true_logit 的【真实因果 pCTR】产生点击(模型不可见的 ground truth)
    ──► 交付指标:session_ctr(主)、coverage_entropy / cold_user_ctr(生态护栏)

关键机制:曝光分布是模型自己选的(selection bias)——一个在离线 AUC 上钻空子、
或制造马太效应的模型,会在这里现形。协议输出:

    ###DELIVERY### {"metrics": {...}, "guards": {...}, "meta": {...}}

模型权重来自 judge 产出的 model_state(driver 落盘为 model_state.json),
保证"离线评测的模型"与"端到端仿真的模型"逐位一致。
"""
from __future__ import annotations

import json
import os

import numpy as np

import model
import prepare

SEED = 2026
N_SESSIONS = 400       # 会话数(每会话一个用户一次请求)
CAND_POOL = 300        # 全场候选池大小(固定,各模型共享同一货架)
TOPK = 8               # 每会话曝光条数
COLD_THRESHOLD = 0.3   # 冷启用户线(activity 低于此)
HOT_LINE = 0.75        # 与 model.py 的倾斜线一致


def build_world(rng: np.random.Generator) -> dict:
    """固定的用户群与候选货架(与 train 同分布,seed 固定,全模型共享)。"""
    users = {
        "activity": rng.beta(2.0, 2.0, N_SESSIONS).astype(np.float32),
        "age": rng.integers(0, prepare.N_AGE, N_SESSIONS).astype(np.int32),
        "gender": rng.integers(0, prepare.N_GENDER, N_SESSIONS).astype(np.int32),
        "hour": rng.integers(0, 24, N_SESSIONS).astype(np.int32),
    }
    cands = {
        "cate": rng.integers(0, prepare.N_CATE, CAND_POOL).astype(np.int32),
        "heat": rng.beta(1.5, 3.0, CAND_POOL).astype(np.float32),
        "quality": rng.beta(2.0, 2.0, CAND_POOL).astype(np.float32),
    }
    return users, cands


def main() -> None:
    state_path = os.environ.get("ALGOPILOT_MODEL_STATE", "model_state.json")
    state = json.load(open(state_path, encoding="utf-8"))
    w = np.array(state["w"], dtype=np.float64)
    mu = np.array(state["norm"][0], dtype=np.float64)
    sd = np.array(state["norm"][1], dtype=np.float64)
    knobs = state["knobs"]

    rng = np.random.default_rng(SEED)
    users, cands = build_world(rng)

    # real 数据下"已知世界真相"不存在,合成世界仿真自动降级(晨报可见,人类可重配)
    data_dir = os.environ.get("ALGOPILOT_DATA_DIR", "data")
    manifest = json.load(open(os.path.join(data_dir, "manifest.json"),
                              encoding="utf-8")) if os.path.exists(
        os.path.join(data_dir, "manifest.json")) else {}
    kind = manifest.get("kind", "synthetic")
    meta = {"cat_card": manifest.get("cat_card", [prepare.N_AGE, prepare.N_GENDER,
                                                  prepare.N_CATE]),
            "tilt": kind != "real"}

    # ---- 1) 会话×候选 全对打分(与 judge 完全相同的特征空间与权重) ----
    S, C = N_SESSIONS, CAND_POOL
    num = np.stack([np.repeat(users["activity"], C),
                    np.tile(cands["heat"], S),
                    np.tile(cands["quality"], S)], axis=1).astype(np.float32)
    pairs, cols, vals, dim, _ = model.build_csr(
        num,
        np.repeat(users["age"], C), np.repeat(users["gender"], C),
        np.tile(cands["cate"], S), np.repeat(users["hour"], C), knobs,
        norm=(mu, sd), meta=meta)
    if dim != len(w):
        raise SystemExit(f"model_state 与当前特征空间不一致: w={len(w)} vs dim={dim}")
    scores = np.bincount(pairs, weights=w[cols] * vals, minlength=S * C)
    top = np.argpartition(-scores.reshape(S, C), TOPK, axis=1)[:, :TOPK]  # (S, K)

    # ---- 2) 真实世界响应:prepare.true_logit 的因果 pCTR(模型不可见) ----
    ex_cand = top % C                                     # 曝光的候选下标
    ex_p = 1.0 / (1.0 + np.exp(-prepare.true_logit(
        users["activity"][:, None], cands["heat"][ex_cand], cands["quality"][ex_cand],
        users["age"][:, None], cands["cate"][ex_cand], users["hour"][:, None])))
    clicks = (rng.random(ex_p.shape) < ex_p).astype(np.float32)

    # ---- 3) 交付指标与生态护栏 ----
    session_ctr = float(clicks.mean())
    expo_cate = cands["cate"][ex_cand].ravel()
    counts = np.bincount(expo_cate, minlength=prepare.N_CATE).astype(np.float64)
    p = counts / counts.sum()
    nz = p[p > 0]
    entropy = float(-(nz * np.log(nz)).sum() / np.log(prepare.N_CATE))  # 归一化曝光熵
    cold = users["activity"] < COLD_THRESHOLD
    cold_ctr = float(clicks[cold].mean()) if cold.any() else 0.0
    hot_share = float((cands["heat"][ex_cand] > HOT_LINE).mean())

    print("###DELIVERY### " + json.dumps({
        "metrics": {"session_ctr": round(session_ctr, 6),
                    "cold_user_ctr": round(cold_ctr, 6),
                    "coverage_entropy": round(entropy, 6),
                    "hot_exposure_share": round(hot_share, 6)},
        "guards": {"n_sessions": S * TOPK, "seed": SEED},
        "meta": {"knobs": knobs, "data_kind": kind,
                 "degraded": kind == "real"},
    }, ensure_ascii=False))
    if kind == "real":
        print("[delivery] 警告:data kind=real,交付仿真运行在合成世界上(真因果缺失)——"
              "请人工重配 delivery.py 或接入会话化真实回放后再信任交付门禁")


if __name__ == "__main__":
    main()
