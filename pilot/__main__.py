# -*- coding: utf-8 -*-
"""AlgoPilot —— 自驱动搜广推算法迭代智能组织。

用法:
  python -m pilot run   --domain domains/rec --engine mock --max-experiments 20
  python -m pilot status --domain domains/rec
  python -m pilot report --domain domains/rec
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .domain import DomainSpec
from .driver import RatchetDriver
from .llm import make_llm
from .report import write_report

PKG_ROOT = Path(__file__).resolve().parent.parent


def _workdir_for(domain_root: Path) -> Path:
    """实验产物默认放 <repo>/experiments/<domain>/,与域目录隔离(装置不被污染)。"""
    return PKG_ROOT / "experiments" / DomainSpec.load(domain_root).name


def cmd_run(args: argparse.Namespace) -> int:
    domain_root = Path(args.domain).resolve()
    spec = DomainSpec.load(domain_root)
    level_cfg = _load_level(args.level)
    max_exp = args.max_experiments or level_cfg.get("max_experiments", 20)
    workdir = Path(args.workdir) if args.workdir else _workdir_for(domain_root)

    print(f"[pilot] 域={spec.name} 引擎={args.engine} 等级={args.level} "
          f"max_experiments={max_exp}")
    print(f"[pilot] {spec.describe()}")
    print(f"[pilot] 产物目录={workdir}")

    llm = make_llm(args.engine)
    driver = RatchetDriver(spec, llm, workdir,
                           stop_on_plateau=level_cfg.get("stop_on_plateau", 8),
                           elite_recheck_every=level_cfg.get("elite_recheck_every", 8),
                           workers=level_cfg.get("workers", 1))
    driver.setup()
    summary = driver.run(max_exp)

    report_path = write_report(spec, workdir)
    print("[pilot] 完成:", json.dumps(summary, ensure_ascii=False))
    print(f"[pilot] 晨报: {report_path}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    spec = DomainSpec.load(Path(args.domain).resolve())
    from .memory import Ledger
    s = Ledger(_workdir_for(Path(args.domain).resolve())).summary()
    print(f"域[{spec.name}] {spec.describe()}")
    print(json.dumps(s, ensure_ascii=False, indent=2))
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    spec = DomainSpec.load(Path(args.domain).resolve())
    path = write_report(spec, _workdir_for(Path(args.domain).resolve()))
    print(path)
    return 0


def cmd_propose(args: argparse.Namespace) -> int:
    """组织自迭代第一步:从台账提炼 program.md 修改提案(不生效,等人审)。"""
    from . import governance
    domain_root = Path(args.domain).resolve()
    spec = DomainSpec.load(domain_root)
    workdir = _workdir_for(domain_root)
    from .memory import Ledger
    ledger = Ledger(workdir)
    program_path = domain_root / spec.program_md

    if args.engine == "llm":
        edits = governance.llm_proposal(make_llm("llm"), ledger,
                                        program_path.read_text(encoding="utf-8"))
    else:
        edits = governance.rules_proposal(
            ledger, program_path.read_text(encoding="utf-8"))
    ok_edits, rejected = governance.validate(edits, program_path.read_text(encoding="utf-8"))
    proposal_path = Path(workdir) / governance.PROPOSAL_NAME
    proposal_path.write_text(
        governance.render_proposal(spec.name, ok_edits, rejected, ledger.summary()),
        encoding="utf-8")
    print(f"[pilot] 提案:{len(ok_edits)} 条通过红线校验,"
          f"{len(rejected)} 条被拒(幂等/越权) → {proposal_path}")
    for e in ok_edits:
        print(f"  + [{e['anchor'][:14]}…] {e['line'][:70]}")
    for r in rejected:
        print(f"  × {r[:80]}")
    print("[pilot] 审阅后执行:python -m pilot approve --domain "
          f"{args.domain}  (--yes 跳过确认)")
    return 0


def cmd_approve(args: argparse.Namespace) -> int:
    """组织自迭代第二步:人类审批 → 备份 → 应用 → 台账 program_update 事件。"""
    from . import governance
    domain_root = Path(args.domain).resolve()
    spec = DomainSpec.load(domain_root)
    workdir = _workdir_for(domain_root)
    from .memory import Ledger
    ledger = Ledger(workdir)
    program_path = domain_root / spec.program_md
    proposal_path = Path(workdir) / governance.PROPOSAL_NAME
    if not proposal_path.exists():
        print("[pilot] 没有提案文件,先运行 propose", file=sys.stderr)
        return 1
    import re as _re
    proposal = proposal_path.read_text(encoding="utf-8")
    edits = [{"anchor": m.group(1).strip(), "line": m.group(2).rstrip()}
             for m in _re.finditer(
                 r"## 修改 \d+\(追加到「(.+?)」\)\n- \*\*内容\*\*:(.+)$",
                 proposal, _re.M)]
    ok_edits, rejected = governance.validate(edits, program_path.read_text(encoding="utf-8"))
    if not ok_edits:
        print("[pilot] 提案中没有可应用条目(全部被拒或已存在)")
        return 0
    if not args.yes:
        for e in ok_edits:
            print(f"  + {e['line'][:80]}")
        ans = input(f"[pilot] 应用以上 {len(ok_edits)} 条修改到 {program_path.name}?[y/N] ")
        if ans.strip().lower() != "y":
            print("[pilot] 已取消(提案保留,可继续修改)")
            return 0
    gen_tag = f".gen{ledger.load_elite().get('gen', 0)}"
    _, diff = governance.apply_edits(ok_edits, program_path, backup_tag=gen_tag)
    ledger.append({
        "verdict": "program_update", "gen": ledger.load_elite().get("gen", 0),
        "idea_id": "program_update",
        "idea": f"program.md 受限自迭代:{len(ok_edits)} 条生效(人类审批)",
        "result": {"applied": True, "edits": [e["line"][:60] for e in ok_edits],
                   "proposal": str(proposal_path)},
    })
    print(diff if args.show_diff else f"[pilot] 已生效(备份 {program_path.name}{gen_tag}),"
          f"台账已记录 program_update 事件。(--show-diff 查看 diff)")
    return 0


def _load_level(level: str) -> dict:
    cfg_path = PKG_ROOT / "configs" / "autonomy_levels.json"
    if cfg_path.exists():
        cfgs = json.loads(cfg_path.read_text(encoding="utf-8"))
        if level in cfgs:
            return cfgs[level]
    defaults = {"L2": {"max_experiments": 5},
                "L3": {"max_experiments": 50, "stop_on_plateau": 8},
                "L4": {"max_experiments": 200, "elite_recheck_every": 8}}
    return defaults.get(level, {})


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="pilot", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="启动自驱动棘轮循环")
    p_run.add_argument("--domain", required=True, help="域目录,如 domains/rec")
    p_run.add_argument("--engine", default="mock", choices=["mock", "llm"],
                       help="mock=零依赖策略遍历器 / llm=OpenAI 兼容(LLM_API_KEY)")
    p_run.add_argument("--level", default="L3", help="自动驾驶等级,见 configs/autonomy_levels.json")
    p_run.add_argument("--max-experiments", type=int, default=None)
    p_run.add_argument("--workdir", default=None, help="实验产物目录(默认 experiments/<域>)")
    p_run.set_defaults(fn=cmd_run)

    p_st = sub.add_parser("status", help="查看台账摘要")
    p_st.add_argument("--domain", required=True)
    p_st.set_defaults(fn=cmd_status)

    p_rp = sub.add_parser("report", help="重新生成晨报")
    p_rp.add_argument("--domain", required=True)
    p_rp.set_defaults(fn=cmd_report)

    p_pr = sub.add_parser("propose", help="组织自迭代:从台账提炼 program.md 修改提案")
    p_pr.add_argument("--domain", required=True)
    p_pr.add_argument("--engine", default="mock", choices=["mock", "llm"])
    p_pr.set_defaults(fn=cmd_propose)

    p_ap = sub.add_parser("approve", help="人类审批提案 → 备份 → 生效 → 台账记录")
    p_ap.add_argument("--domain", required=True)
    p_ap.add_argument("--yes", action="store_true", help="跳过交互确认")
    p_ap.add_argument("--show-diff", action="store_true")
    p_ap.set_defaults(fn=cmd_approve)

    args = ap.parse_args(argv)
    try:
        return args.fn(args)
    except Exception as e:  # noqa: BLE001 fails loud,但不裸 traceback 到用户
        print(f"[pilot] 错误: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
