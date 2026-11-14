# -*- coding: utf-8 -*-
"""judge.py:推荐域评测器(冻结文件,任何角色不许改)。

评测协议(AlgoPilot 的"仪表盘",engine 解析 stdout 最后一行):
    ###JUDGE### {"metrics": {...}, "guards": {...}, "meta": {...}}

职责划分(谁上报什么,是防作弊设计):
  - model.py 只报算法指标(auc/val_logloss)与静态护栏(特征维度、数据指纹);
  - train_seconds(耗时护栏)由 judge 独立测量 —— 模型自己报的耗时不可信;
  - 数据指纹 data_hash 与 data/manifest.json 比对,换数据 = 立刻 failed。
"""
from __future__ import annotations

import json
import os
import time

import model

BUDGET_SECONDS = 60


def main() -> None:
    data_dir = os.environ.get("ALGOPILOT_DATA_DIR",
                              os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
    manifest_path = os.path.join(data_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        raise SystemExit("数据未准备:请先运行 prepare.py")

    t0 = time.time()
    out = model.run_pipeline(data_dir)
    out["guards"]["train_seconds"] = round(time.time() - t0, 2)

    # 数据指纹校验(装置级,模型不可绕过)
    manifest = json.load(open(manifest_path, encoding="utf-8"))
    reported = str(out.get("guards", {}).get("data_hash", ""))
    if reported and reported != manifest.get("data_hash"):
        raise SystemExit(f"数据指纹不符: 报告 {reported} vs 装置 {manifest.get('data_hash')}")
    out["guards"]["data_hash"] = manifest.get("data_hash", "")
    out["guards"]["n_samples"] = manifest.get("n_train", 0)
    out["meta"]["budget_seconds"] = BUDGET_SECONDS

    print("###JUDGE### " + json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
