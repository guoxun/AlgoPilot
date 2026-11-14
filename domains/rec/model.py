# -*- coding: utf-8 -*-
"""model.py:推荐域排序模型 —— **agent 唯一可编辑的文件**(人类写的朴素基线)。

契约(judge.py 依赖,不可破坏):
  1. 必须保留 KNOBS 字典,且键集合不变(只能改值 / 或整体 rewrite 后仍是这些键);
  2. 必须提供 run_pipeline(data_dir: str) -> dict,
     返回 {"metrics": {"auc": float, "val_logloss": float, ...}, "guards": {...}, "meta": {...}};
  3. 只读数据(ALGOPILOT_DATA_DIR 指向的目录),不得做任何文件写/网络/系统操作。

朴素基线的已知局限(留给迭代空间,详见 program.md):
  - 没有二阶特征交叉:合成数据里「年龄×品类」「活跃度×质量」存在真实交互信号;
  - 交叉分桶粒度固定;
  - 无 early stopping、无损失加权、无校准。
"""
from __future__ import annotations

import json
import os

import numpy as np

N_AGE, N_GENDER, N_CATE = 5, 2, 10  # 合成数据默认类别基数(real 数据以 manifest.cat_card 为准)
BUCKET_PER_FIELD = 512              # 类别域哈希桶上限(real 数据基数大时防 onehot 爆炸)


def _load_manifest(data_dir: str) -> dict:
    path = os.path.join(data_dir, "manifest.json")
    return json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}

# ============================ KNOBS(迭代旋钮区,只改值) ============================
KNOBS = {
    "_use_feature_cross": False,  # 二阶交叉特征:age×cate、gender×cate、act×quality 分桶积
    "_n_act_buckets": 4,          # activity/quality 分桶粒度(交叉组合用的桶数)
    "_heat_boost": 0.0,           # 高热段倾斜(heat>0.75 的额外斜率)—— 真实重排规则,
                                  # 离线可能微涨,但会让曝光生态向热门坍缩(马太效应),
                                  # 交付门禁的典型拦截对象
    "_lr": 0.1,                   # 全量 GD 学习率(朴素保守基线)
    "_momentum": 0.0,             # 动量(留给迭代:可加速但需小心稳定性)
    "_epochs": 200,
    "_l2": 1e-4,
    "_patience": 0,               # early stop 轮数(0=关闭),基于内部验证集
    "_pos_weight": 1.0,           # 正例损失权重(类不平衡)
    "_use_calib": False,          # Platt 校准(只改善 logloss,不影响 AUC)
}
# ===================================================================================


# ---------------------------------------------------------------- 特征工程
def build_csr(num: np.ndarray, cat_age: np.ndarray, cat_gender: np.ndarray,
              cat_cate: np.ndarray, hour: np.ndarray, knobs: dict,
              norm: tuple | None = None, meta: dict | None = None):
    """把原始列编码为稀疏特征(CSR)。基线:标准化数值 + 周期编码 + 类别 one-hot。

    norm=(mu, sd) 必须来自训练集:评测集用训练集统计量变换,否则分布不一致。
    meta 装置参数(real 数据接入时由 manifest 提供,合成路径用默认值):
      cat_card: 三域类别基数(域内先哈希到 min(card, BUCKET_PER_FIELD) 桶)
      tilt:     高热倾斜列是否生效(real 数据无 heat 语义,自动关闭)
    返回 (rows, cols, vals, dim, norm)。
    """
    meta = meta or {"cat_card": [N_AGE, N_GENDER, N_CATE], "tilt": True}
    cards = [min(int(c), BUCKET_PER_FIELD) for c in meta["cat_card"]]
    n = num.shape[0]
    mu, sd = norm if norm is not None else (num.mean(axis=0), num.std(axis=0) + 1e-6)
    num_std = (num - mu) / sd
    hr = hour.astype(np.float64)
    period = np.stack([np.sin(2 * np.pi * hr / 24), np.cos(2 * np.pi * hr / 24)], axis=1)

    n_num = num_std.shape[1] + period.shape[1]          # 3 + 2
    n_cat = N_AGE + N_GENDER + N_CATE

    # 块结构:偏置块(列 0,值恒 1)/ 数值块 / 周期块 / 类别 one-hot 块(值恒 1),列号顺序累加
    blocks = []  # (cols_matrix, vals_matrix)
    blocks.append((np.zeros((n, 1), dtype=np.int64), np.ones((n, 1))))  # bias —— 没有它,优化器要靠零均值特征凑截距,必卡死
    dim = 1
    w_num = num_std.shape[1]
    blocks.append((np.tile(np.arange(dim, dim + w_num), (n, 1)), num_std))
    dim += w_num
    w_per = period.shape[1]
    blocks.append((np.tile(np.arange(dim, dim + w_per), (n, 1)), period))
    dim += w_per
    cat_cols = np.stack([cat_age % cards[0],
                         cat_gender % cards[1],
                         cat_cate % cards[2]], axis=1)
    cat_cols[:, 0] += dim
    cat_cols[:, 1] += dim + cards[0]
    cat_cols[:, 2] += dim + cards[0] + cards[1]
    blocks.append((cat_cols.astype(np.int64), np.ones((n, 3))))
    dim += cards[0] + cards[1] + cards[2]

    # 高热段倾斜列(始终存在,boost=0 时值恒 0,保持 dim 恒定):
    # 训练后高热样本获得额外斜率 —— 行业里"给热门加权"重排规则的模型化,
    # 是交付门禁要拦截的马太效应来源。real 数据无 heat 语义,自动置零。
    if meta.get("tilt", True):
        tilt = np.where(num[:, 1] > 0.75,
                        num[:, 1] * float(knobs.get("_heat_boost", 0.0)), 0.0)
    else:
        tilt = np.zeros(n)
    blocks.append((np.full((n, 1), dim, dtype=np.int64), tilt.reshape(n, 1)))
    dim += 1

    if knobs["_use_feature_cross"]:
        B = int(knobs["_n_act_buckets"])
        act_b = np.clip((num[:, 0] * B).astype(np.int32), 0, B - 1)      # activity 桶
        qua_b = np.clip((num[:, 2] * B).astype(np.int32), 0, B - 1)      # quality 桶
        x_age_cate = cat_age * N_CATE + cat_cate
        x_gen_cate = cat_gender * N_CATE + cat_cate
        x_act_qua = act_b * B + qua_b
        base = dim
        n_cross = N_AGE * N_CATE + N_GENDER * N_CATE + B * B
        cross = np.stack([x_age_cate + base,
                          x_gen_cate + base + N_AGE * N_CATE,
                          x_act_qua + base + N_AGE * N_CATE + N_GENDER * N_CATE],
                         axis=1).astype(np.int64)
        blocks.append((cross, np.ones((n, 3))))
        dim += n_cross

    # 行优先拼接:先按 axis=1 拼成 (n, W) 再 ravel —— 每行的 W 个 (列,值) 对相邻。
    # 若按块 ravel 后拼接会变成块优先(某列全行连续),与 rows=repeat(arange(n),W) 错位。
    cols_mat = np.concatenate([c for c, _ in blocks], axis=1)
    vals_mat = np.concatenate([np.asarray(v, dtype=np.float64) for _, v in blocks], axis=1)
    cols = cols_mat.ravel().astype(np.int64)
    vals = vals_mat.ravel()
    rows = np.repeat(np.arange(n, dtype=np.int64), cols_mat.shape[1])
    return rows, cols, vals, dim, (mu, sd)


# ---------------------------------------------------------------- 训练与评估
def sigmoid(z: np.ndarray) -> np.ndarray:
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def rank_average(s: np.ndarray) -> np.ndarray:
    order = np.argsort(s, kind="mergesort")
    s_sorted = s[order]
    boundaries = np.nonzero(np.diff(s_sorted))[0] + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [len(s)]])
    ranks_sorted = np.empty(len(s))
    for st, en in zip(starts, ends):
        ranks_sorted[st:en] = (st + 1 + en) / 2.0
    ranks = np.empty(len(s))
    ranks[order] = ranks_sorted
    return ranks


def auc_score(y: np.ndarray, s: np.ndarray) -> float:
    ranks = rank_average(s)
    pos = y > 0.5
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def logloss(y: np.ndarray, p: np.ndarray, sw: np.ndarray | None = None) -> float:
    p = np.clip(p, 1e-7, 1 - 1e-7)
    loss = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    return float((loss * sw).mean() if sw is not None else loss.mean())


def train_lr(rows, cols, vals, y: np.ndarray, dim: int, knobs: dict,
             intern_rows, intern_cols, intern_vals, intern_y: np.ndarray):
    """稀疏逻辑回归:全量 GD + momentum,可选 early stop(内部验证集,不碰 val)。"""
    n = len(y)
    row_of_nnz = rows
    sw = np.where(y > 0.5, knobs["_pos_weight"], 1.0)
    w = np.zeros(dim)
    v = np.zeros(dim)
    best_w, best_loss, since_best = w.copy(), float("inf"), 0
    for _ in range(int(knobs["_epochs"])):
        z = np.bincount(row_of_nnz, weights=w[cols] * vals, minlength=n)
        p = np.clip(sigmoid(z), 1e-7, 1 - 1e-7)
        resid = (p - y) * sw
        grad = np.bincount(cols, weights=resid[row_of_nnz] * vals,
                           minlength=dim) / n + knobs["_l2"] * w
        v = knobs["_momentum"] * v - knobs["_lr"] * grad
        w = w + v

        if knobs["_patience"]:
            zi = np.bincount(intern_rows, weights=w[intern_cols] * intern_vals,
                             minlength=len(intern_y))
            il = logloss(intern_y, sigmoid(zi))
            if il < best_loss - 1e-5:
                best_loss, best_w, since_best = il, w.copy(), 0
            else:
                since_best += 1
                if since_best >= int(knobs["_patience"]):
                    w = best_w
                    break
    return w


def platt_fit(z: np.ndarray, y: np.ndarray, iters: int = 300, lr: float = 0.1):
    """内部验证集上的两参数 Platt 校准(只影响 logloss,AUC 不变)。"""
    a, b = 1.0, 0.0
    n = len(y)
    for _ in range(iters):
        p = np.clip(sigmoid(a * z + b), 1e-7, 1 - 1e-7)
        da = ((p - y) * z).mean()
        db = (p - y).mean()
        a, b = a - lr * da, b - lr * db
    return a, b


# ---------------------------------------------------------------- 主入口(契约)
def run_pipeline(data_dir: str) -> dict:
    knobs = dict(KNOBS)
    manifest = _load_manifest(data_dir)
    meta = {
        "cat_card": manifest.get("cat_card", [N_AGE, N_GENDER, N_CATE]),
        "tilt": manifest.get("kind", "synthetic") != "real",
    }
    train = np.load(os.path.join(data_dir, "train.npz"))
    val = np.load(os.path.join(data_dir, "val.npz"))

    # 内部 90/10 切分(early stop 与校准用,val 只在最终评测时碰一次 —— 无泄漏)
    rng = np.random.default_rng(7)
    idx = rng.permutation(len(train["label"]))
    cut = int(len(idx) * 0.9)
    tr, iv = idx[:cut], idx[cut:]

    def take(split, ids):
        return {k: v[ids] for k, v in split.items()}

    tr_s, iv_s = take(train, tr), take(train, iv)
    rows, cols, vals, dim, norm = build_csr(tr_s["num"], tr_s["cat_age"], tr_s["cat_gender"],
                                            tr_s["cat_cate"], tr_s["hour"], knobs, meta=meta)
    i_rows, i_cols, i_vals, i_dim, _ = build_csr(iv_s["num"], iv_s["cat_age"],
                                                 iv_s["cat_gender"], iv_s["cat_cate"],
                                                 iv_s["hour"], knobs, norm=norm, meta=meta)
    assert i_dim == dim
    w = train_lr(rows, cols, vals, tr_s["label"], dim, knobs,
                 i_rows, i_cols, i_vals, iv_s["label"])

    v_rows, v_cols, v_vals, v_dim, _ = build_csr(val["num"], val["cat_age"],
                                                 val["cat_gender"], val["cat_cate"],
                                                 val["hour"], knobs, norm=norm, meta=meta)
    assert v_dim == dim
    zv = np.bincount(v_rows, weights=w[v_cols] * v_vals, minlength=len(val["label"]))
    auc = auc_score(val["label"], zv)
    p = sigmoid(zv)

    if knobs["_use_calib"]:
        zi = np.bincount(i_rows, weights=w[i_cols] * i_vals, minlength=len(iv_s["label"]))
        a, b = platt_fit(zi, iv_s["label"])
        p = sigmoid(a * zv + b)

    return {
        "metrics": {"auc": round(auc, 6), "val_logloss": round(logloss(val["label"], p), 6)},
        "guards": {"n_features": int(dim), "data_hash": manifest.get("data_hash", "")},
        "meta": {"knobs": {k: v for k, v in knobs.items()},
                 "data_kind": manifest.get("kind", "synthetic")},
        # model_state:交付仿真(delivery.py)用同一份权重与归一化参数打分,
        # 保证"离线评测的模型"与"端到端仿真的模型"逐位一致
        "model_state": {"w": [round(float(x), 6) for x in w],
                        "norm": [[round(float(x), 6) for x in norm[0]],
                                 [round(float(x), 6) for x in norm[1]]],
                        "knobs": dict(knobs)},
    }


if __name__ == "__main__":
    out = run_pipeline(os.environ.get("ALGOPILOT_DATA_DIR",
                                      os.path.join(os.path.dirname(__file__), "data")))
    print(json.dumps(out, ensure_ascii=False))
