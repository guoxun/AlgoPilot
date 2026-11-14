# AlgoPilot 架构

## 总览

```
                      人类
              program.md(唯一入口)      morning_report.md(唯一出口)
                    │                          ▲
                    ▼                          │
   ┌────────────────────────────────────────────────────────┐
   │                    RatchetDriver(棘轮主循环)             │
   │                                                        │
   │   ┌──────────┐   ┌──────────┐   ┌──────────┐            │
   │   │ 研究员    │ → │ 工程师    │ → │ 评审员    │  (pilot/org.py)
   │   │ (LLM)    │   │ (LLM)    │   │ (纯代码)  │            │
   │   └────▲─────┘   └────┬─────┘   └────┬─────┘            │
   │        │              ▼              │ 护栏审查          │
   │        │         model.py patch      ▼                  │
   │        │        ┌──────────────────────┐               │
   │        │        │ sandbox 子进程        │  固定预算      │
   │        │        │ judge.py → ###JUDGE###│  超时即死      │
   │        │        └──────────┬───────────┘               │
   │        │                   ▼                           │
   │        │        ┌──────────────────────┐               │
   │        │        │ 第一道门:离线棘轮      │  AUC ≥ elite+Δ │
   │        │        └──────────┬───────────┘               │
   │        │                   ▼ 通过                      │
   │        │        ┌──────────────────────┐               │
   │        │        │ 第二道门:交付门禁       │  model_state  │
   │        │        │ delivery.py 会话仿真   │  逐位同模型    │
   │        │        │ ###DELIVERY###        │  不回退+生态护栏│
   │        │        └──────────┬───────────┘               │
   │        │                   ▼                           │
   │   ┌────┴───────────────────────────┐                    │
   │   │ kept / rejected / failed /      │                    │
   │   │ rejected_by_delivery            │                    │
   │   └────┬───────────────────────────┘                    │
   └────────┼───────────────────────────────────────────────┘
            ▼
   ┌──────────────────────────────────────┐
   │ Ledger 台账(pilot/memory.py)         │
   │  experiments.jsonl  追加式唯一事实源    │
   │  strategies.jsonl   策略库(kept 沉淀) │
   │  failures.jsonl     失败库(避坑经验)  │
   │  elite_state.json   最优投影(可重建)  │
   │  └ 每条 kept 记录带 alignment(兑现率) │
   └──────────────────────────────────────┘
```

## 端到端交付仿真(domains/rec/delivery.py,冻结装置)

把"离线提升"放进微缩交付管线接受真实世界检验:

```
固定 seed 会话流(400 会话 × 300 候选货架)
→ 模型打分 top-8 曝光(曝光分布由模型自选 → selection bias 天然存在)
→ 曝光物品按 prepare.true_logit 的真实因果 pCTR 产生点击(模型不可见的 ground truth)
→ 交付指标:session_ctr(主)、coverage_entropy / cold_user_ctr(生态护栏)
```

三个设计要点:

1. **model_state 契约**:judge 产出模型权重与归一化参数,driver 落盘后交付仿真
   用**逐位相同的模型**打分——离线评测的模型与端到端仿真的模型严格一致;
2. **曝光分布由模型自己决定**:一个钻离线空子或制造马太效应的模型,会在
   自选择的曝光分布下现形(如 `_heat_boost` 试探:离线微涨,曝光熵 0.68→0.53
   已跌破生态护栏,台账可查);
3. **兑现率沉淀**:每次保留记录 `offline Δ / delivery Δ / ratio`,晨报展示,
   研究员观察上下文携带对齐历史——兑现率成为组织的方向感。

## 批式并行与合并语义(L4)

```
批首:elite E0(分叉快照,整批的判定基准)
  ├─ w0:研究员出假设 → 工程师补丁 → 评审员护栏 → sandbox 并行跑 judge ─┐
  ├─ w1:同上(独立 sandbox,批内 idea 互斥)                          ├─ 并行
  └─ ...                                                             ┘
批末:串行结算
  ├─ 每分支相对【分叉点 E0】过棘轮 + 交付门禁(不背结算顺序的锅)
  ├─ kept 中主指标最高者 = winner,合并为新 elite(写快照 + 策略库)
  └─ append-only 的 merge 决策事件:胜者 + 落选分支(merged_out)+ elite 投影
```

三个设计要点:

1. **分叉点结算**:并行分支从同一快照出发互不背锅——判定基准是批首 elite,
   不是"结算时点"的 elite(否则批内前序 kept 会误杀后序分支);
2. **最优者合并**:一批至多产生一次 gen+1;落选 kept 保留在台账(分支经验、
   交付指标与兑现率俱全)但代码不进主线——像 PR 没被 merge,review 记录仍在;
3. **台账即真相的延续**:merge 是追加的决策事件而非改写历史;rebuild_elite
   优先读 merge 事件携带的 elite 投影。单 worker 时分叉基线=结算基线,
   行为退化为普通棘轮,与 L3 完全一致。

实测教训(并行改变了探索动力学):aggressive-lr 在串行模式曾叠加保留
(基于 feat-cross 之后的基座),在并行模式它基于无交叉基座独立成枝,
离线 +0.0024、交付 CTR +0.0103——**但曝光熵 0.68→0.16**,被生态护栏拦下。
单一指标都会放行,门禁体系的护栏维度起了决定作用。

## 与 autoresearch 的对照

| autoresearch | AlgoPilot | 升级点 |
|---|---|---|
| `train.py`(agent 唯一可改) | `domains/*/model.py` | 同构;增加 KNOBS 声明区,让 mock 引擎也能真实改代码 |
| `prepare.py`(冻结) | `prepare.py` + 数据指纹 | 同构;加 data_hash 防"偷偷换数据" |
| val_bpb(单一指标) | `domain.json`:主指标+护栏+方向 | 搜广推需要护栏体系(时延/生态),契约化 |
| **val_bpb ≈ 交付(幸运)** | **delivery.py 端到端交付门禁** | 搜广推离线≠交付:双门禁 + 兑现率对齐 |
| 5 分钟固定预算 | `budget_seconds` 契约字段 | 同一铁律,秒级(CPU 可跑) |
| `program.md`(人类组织代码) | `program.md` | 同构 |
| 外部编码 agent 执行循环 | `pilot/` 内置循环引擎 | 不依赖特定编码 agent;也可用外部 agent 只读 program.md 手动执行 |
| 无台账(靠 git/对话记忆) | `Ledger` 追加式台账+策略库+失败库 | 组织记忆;决策可重建(继承 2.0"日志即真相") |
| 单 agent | 研究员/工程师/评审员三角色 | 角色产出物互相审计 |

## 关键机制

### 1. DomainSpec 契约(pilot/domain.py)

```json
{
  "name": "rec", "metric": "auc", "direction": "max",
  "min_improve": 0.001, "budget_seconds": 90,
  "guard_limits": {"train_seconds": ["le", 60], "val_logloss": ["le", 0.35]},
  "editable_files": ["model.py"], "frozen_files": ["prepare.py", "judge.py"]
}
```

- 引擎启动前 `validate()` fails loud(缺文件/契约矛盾直接拒绝);
- `better()` 封装方向,循环内不出现任何 `if auc > ...` 裸比较;
- `guard_violations()` 的护栏源 = metrics ∪ guards:任何上报指标都可声明为护栏。

### 2. 沙箱与 ###JUDGE### 协议(pilot/driver.py)

每次实验:拷贝域文件到一次性 `sandbox/` → 应用 patch → 子进程跑
`judge.py`(超时 = 预算 + 30s 缓冲)→ 解析 stdout 最后一行
`###JUDGE### {json}`。装置三重防作弊:

- **耗时由 judge 独立测量**(模型自报耗时不可信);
- **数据指纹校验**(manifest.json 的 data_hash,换数据立刻 failed);
- **评审员黑名单**(rewrite 不得出现 subprocess/socket/eval 等)。

### 3. 棘轮与统计诚实

- warmup:同一份 elite 代码跑 2 次,指标方差 → `min_delta = max(契约下限, 3×std)`
  ——评测噪声大时自动提高认定门槛;
- 复验:每 N 次 kept 重跑一次 elite 快照,漂移超 2×min_delta 写告警事件(防泄漏/漂移);
- 装置确定性:固定 seed,同代码两次运行指标逐位一致(测试保证)。

### 4. 双档 LLM(pilot/llm.py)

| 档 | 实现 | 用途 |
|---|---|---|
| mock | 确定性策略遍历器(读 mock_policy.json,跳过已试) | 零依赖演示/回归测试/无 key 环境 |
| llm | OpenAI 兼容 chat/completions(urllib 标准库) | DeepSeek/智谱 GLM/oneapi 网关;研究员+工程师走真模型 |

接口统一为 `complete(task, payload) -> dict`,org 层不感知引擎档位。
真 LLM 输出经容错 JSON 解析(剥围栏/平衡括号),失败重试 2 次,仍失败记 failed。

### 5. 工程师的两档动作与评审员的红线

- `set_knobs`:只改 KNOBS 字典的值(键集合是契约)——可靠档;
- `rewrite`:整文件重写——进阶档,评审员强制:KNOBS 键集合不变、
  危险内容黑名单、规模上限(≤3×原行数)。

## 规模与复杂度

- pilot/ 引擎 ~700 行,零第三方依赖(标准库 only);
- rec 域装置 ~300 行 + numpy;单实验 0.2~0.6s(预算 90s);
- 测试 41 项,含组织全循环端到端(真子进程、真 sandbox、真棘轮)。
