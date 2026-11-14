# -*- coding: utf-8 -*-
"""prepare.py:推荐域实验装置 —— 数据准备(冻结文件,任何角色不许改)。

合成一份**确定性**的曝光/点击数据(固定 seed,可复现):
  - 真实因果结构:点击概率由「主效应 + 年龄×品类交叉 + 活跃度×质量交叉 + 时段偏置」决定
    → 这保证"特征交叉"等算法改动**真实有效**,不是玩具摆设;
  - 正例率约 8%,接近真实 CTR 量级;
  - 数据版本 hash 写入 data/manifest.json —— judge 上报,防"偷偷换数据"。

输出:data/train.npz、data/val.npz、data/manifest.json
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

SEED = 42
N_TRAIN, N_VAL = 20000, 5000
N_AGE, N_GENDER, N_CATE = 5, 2, 10

# 年龄×品类 偏好强度矩阵(强交叉信号:模型不开交叉特征就拿不到这部分 AUC)
AGE_CATE_BIAS = np.array([
    # cate: 0     1     2     3     4     5     6     7     8     9
    [0.00, 1.30, -0.80, 0.20, -0.40, 0.90, -1.10, 0.30, 0.60, -0.50],  # age 0
    [0.80, -0.60, 0.40, 1.10, -0.90, -0.30, 0.50, -1.00, 0.20, 0.70],  # age 1
    [-0.70, 0.30, 1.20, -0.50, 0.80, -0.90, 0.40, 0.90, -0.60, 0.10],  # age 2
    [0.50, -1.00, -0.30, 0.60, 1.10, 0.20, -0.70, -0.40, 1.00, 0.30],  # age 3
    [-0.40, 0.60, 0.90, -1.00, 0.10, 1.20, 0.60, -0.80, -0.30, 1.10],  # age 4
]) * 1.2

# 时段偏置:晚间黄金档高、凌晨低谷(数值 sin/cos 只能抓住部分周期 → 留有特征工程空间)
HOUR_BIAS = np.array([-.5, -.6, -.6, -.5, -.4, -.3, -.2, 0., .1, .2,
                      .2, .2, .2, .2, .1, .1, .2, .3, .5, .7, .8, .8, .7, -.1])


def true_logit(activity, heat, quality, age, cate, hour):
    """真实因果 pCTR 机制 —— 全域唯一事实源。

    gen_split 用它造数据;delivery.py 用它当"真实世界"(模型不可见的 ground truth)
    生成端到端仿真的点击。改这里 = 换世界,judge/delivery 同时换代,历史台账作废。
    """
    return (-3.6                                            # 基础截距 → CTR≈10%
            + 0.9 * activity + 0.7 * quality + 0.35 * heat
            + AGE_CATE_BIAS[age, cate]
            + 1.6 * (activity - 0.5) * (quality - 0.5)      # 活跃度×质量 交叉
            + HOUR_BIAS[hour])


def gen_split(n: int, rng: np.random.Generator) -> dict:
    activity = rng.beta(2.0, 2.0, n).astype(np.float32)          # 用户活跃度
    age = rng.integers(0, N_AGE, n).astype(np.int32)             # 用户年龄段
    gender = rng.integers(0, N_GENDER, n).astype(np.int32)       # 性别
    cate = rng.integers(0, N_CATE, n).astype(np.int32)           # 物品品类
    heat = rng.beta(1.5, 3.0, n).astype(np.float32)              # 物品热度
    quality = rng.beta(2.0, 2.0, n).astype(np.float32)           # 物品质量
    hour = rng.integers(0, 24, n).astype(np.int32)               # 请求时段

    logit = true_logit(activity, heat, quality, age, cate, hour)
    p = 1.0 / (1.0 + np.exp(-logit))
    label = (rng.random(n) < p).astype(np.float32)
    return {
        "num": np.stack([activity, heat, quality], axis=1),      # 数值特征
        "cat_age": age, "cat_gender": gender, "cat_cate": cate,
        "hour": hour, "label": label,
    }


def load_real_data(out_dir: Path) -> dict:
    """真实数据适配器(L3.5):把 data_raw/*.csv 转成装置标准 npz。

    约定(把 Criteo/公司导出转成本格式只需改列名):
      data_raw/train.csv 与 data_raw/val.csv,UTF-8,含表头,列:
        label           0/1 点击标签
        num_0..num_k    数值特征(缺失留空或非数值 → 0)
        cat_a, cat_b, cat_c  三个类别域(任意基数,装置内哈希成 int)
    输出 npz 结构与合成路径完全一致(model.py 无感),manifest 标记 kind=real。
    """
    import csv

    def load_split(path: Path) -> dict | None:
        if not path.exists():
            return None
        num_cols, rows = [], []
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            num_cols = sorted(c for c in reader.fieldnames if c.startswith("num_"))
            cat_cols = sorted(c for c in reader.fieldnames if c.startswith("cat_"))
            if len(cat_cols) != 3:
                raise SystemExit(f"{path.name}: 需要恰好 3 个类别域(cat_*),实际 {cat_cols}")
            num, cats, labels = [], [[] for _ in cat_cols], []
            for row in reader:
                labels.append(float(row["label"]))
                num.append([float(row[c]) if row.get(c, "").strip() else 0.0
                            for c in num_cols])
                for j, c in enumerate(cat_cols):
                    cats[j].append(row[c])
        import numpy as np
        # 类别域:字符串 → 稳定哈希 → int32(基数不限,模型侧再做桶内哈希)
        cat_arrays = [np.array([abs(hash(v)) % (2 ** 31) for v in col],
                               dtype=np.int32) for col in cats]
        return {
            "num": np.array(num, dtype=np.float32),
            "cat_age": cat_arrays[0], "cat_gender": cat_arrays[1], "cat_cate": cat_arrays[2],
            "hour": np.zeros(len(labels), dtype=np.int32),  # 真实数据无时段列时置 0
            "label": np.array(labels, dtype=np.float32),
        }

    train = load_split(out_dir.parent / "data_raw" / "train.csv")
    val = load_split(out_dir.parent / "data_raw" / "val.csv")
    if train is None or val is None:
        return None
    np.savez_compressed(out_dir / "train.npz", **train)
    np.savez_compressed(out_dir / "val.npz", **val)
    h = hashlib.sha256()
    h.update(train["label"].tobytes())
    h.update(val["label"].tobytes())
    manifest = {
        "kind": "real",
        "n_train": int(len(train["label"])), "n_val": int(len(val["label"])),
        "ctr_train": round(float(train["label"].mean()), 5),
        "ctr_val": round(float(val["label"].mean()), 5),
        "n_num": int(train["num"].shape[1]),
        "cat_card": [int(train[c].max()) + 1 for c in ("cat_age", "cat_gender", "cat_cate")],
        "data_hash": h.hexdigest()[:16],
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[prepare] 真实数据: train={manifest['n_train']} val={manifest['n_val']} "
          f"ctr={manifest['ctr_train']} num列={manifest['n_num']} hash={manifest['data_hash']}")
    return manifest


def main() -> None:
    out_dir = Path(os.environ.get("ALGOPILOT_DATA_DIR", Path(__file__).parent / "data"))
    out_dir.mkdir(parents=True, exist_ok=True)
    if load_real_data(out_dir) is not None:
        return  # L3.5:data_raw/ 存在即用真实数据,不再合成

    rng = np.random.default_rng(SEED)

    train, val = gen_split(N_TRAIN, rng), gen_split(N_VAL, rng)
    np.savez_compressed(out_dir / "train.npz", **train)
    np.savez_compressed(out_dir / "val.npz", **val)

    h = hashlib.sha256()
    h.update(train["label"].tobytes())
    h.update(val["label"].tobytes())
    manifest = {
        "kind": "synthetic",
        "seed": SEED, "n_train": N_TRAIN, "n_val": N_VAL,
        "ctr_train": round(float(train["label"].mean()), 5),
        "ctr_val": round(float(val["label"].mean()), 5),
        "data_hash": h.hexdigest()[:16],
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[prepare] 完成: train={N_TRAIN} val={N_VAL} "
          f"ctr={manifest['ctr_train']} hash={manifest['data_hash']}")


if __name__ == "__main__":
    main()
