"""阶段一实验：单步 GRPO，能不能自己学会「状态 → 方向」？

不依赖 SFT。每次更新：
    从冻结数据集采样 K 个状态 → 每个状态 G 条单步采样 → 组内标准化 → GRPO

与旧版的区别：
  - 训练状态来自 data/states/train.jsonl（动作均衡，不再现场随机选格）；
  - val / test 分离：val 做早停和 best，test 只做最终报告；
  - 同时保留 data/eval_states.jsonl 作为 natural eval；
  - 指标拆成 optimal / side / farther / blocked / invalid，不再把走远算成好；
  - 逐 update JSONL + 定期 checkpoint + best checkpoint；
  - 可接训练 UI：每个状态的 G 条采样生成独立 episode，多个浏览器窗口
    可以分别 pin 不同样本并排查看。

    python scripts/build_states.py
    python scripts/check_states.py
    python scripts/train_phase1.py --levels L0 --updates 20 --states 16 --g 8 \
        --ui-url http://127.0.0.1:7861
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import paths  # noqa: E402,F401

import torch  # noqa: E402

from dataset import (  # noqa: E402
    MapDataset, MapRecord, StateDataset, StateRecord, action_class,
    optimal_actions,
)
from engine_rl import RolloutEngine, Sampling, load_model  # noqa: E402
from env.maze import (  # noqa: E402
    ACTIONS,
    LEVELS,
    REWARD_COLLISION,
    REWARD_DISTANCE,
    REWARD_FORMAT,
    REWARD_STEP,
    REWARD_TERMINAL,
    Maze,
    manhattan,
)
from env.render import render_state  # noqa: E402
from env.ui import TrainViz, build_frame  # noqa: E402
from grpo.adapter import LoRAAdapter, tie_embeddings  # noqa: E402
from grpo.train import (  # noqa: E402
    GRPOTrainer,
    Sample,
    TrainConfig,
    group_advantages,
)
from prompts import render_prompt  # noqa: E402
from rollout import parse_action  # noqa: E402


# ---------------------------------------------------------------- 数据


@dataclass
class StateItem:
    id: str
    map_id: str
    level: str
    maze: Maze
    pos: tuple[int, int]
    state: str
    optimal: frozenset[str]
    action_class: str
    distance: int
    manhattan: int


def _read_jsonl(path: Path) -> list[dict]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _map_index(data_dir: Path, split: str) -> dict[str, MapRecord]:
    ds = MapDataset.load(data_dir / "maps" / f"{split}.jsonl")
    return {r.id: r for r in ds.records}


def _make_item(rec: StateRecord, maze: Maze, obs_style: str) -> StateItem:
    pos = rec.pos
    acts = optimal_actions(maze, pos)
    return StateItem(
        id=rec.id,
        map_id=rec.map_id,
        level=rec.level,
        maze=maze,
        pos=pos,
        state=render_state(maze, pos, style=obs_style),
        optimal=frozenset(acts),
        action_class=action_class(acts),
        distance=maze.distance(pos),
        manhattan=manhattan(pos, maze.goal),
    )


def load_state_items(state_dir: Path, data_dir: Path, state_file: str,
                     map_split: str, levels: tuple[str, ...],
                     obs_style: str) -> list[StateItem]:
    path = state_dir / state_file
    if not path.exists():
        raise FileNotFoundError(
            f"{path} 不存在；先运行 python scripts/build_states.py")
    ds = StateDataset.load(path).filter(levels=levels)
    maps = _map_index(data_dir, map_split)
    cache: dict[str, Maze] = {}
    out: list[StateItem] = []
    for rec in ds:
        mrec = maps.get(rec.map_id)
        if mrec is None:
            raise RuntimeError(f"{path}: map_id {rec.map_id} 不在 {map_split}")
        maze = cache.get(rec.map_id)
        if maze is None:
            maze = mrec.to_maze()
            cache[rec.map_id] = maze
        out.append(_make_item(rec, maze, obs_style))
    if not out:
        raise RuntimeError(f"{path}: no states for levels={levels}")
    return out


def load_natural_items(data_dir: Path, levels: tuple[str, ...],
                       obs_style: str, limit: int = 0,
                       path: Path | None = None) -> list[StateItem]:
    """原始 natural eval（默认 data/eval_states.jsonl，不做动作均衡）。"""
    path = Path(path) if path is not None else data_dir / "eval_states.jsonl"
    if not path.exists():
        return []
    maps = _map_index(data_dir, "test_id")
    rows = [r for r in _read_jsonl(path) if r.get("level") in levels]
    rows.sort(key=lambda r: str(r.get("id", "")))
    cache: dict[str, Maze] = {}
    out: list[StateItem] = []
    for row in rows:
        rec = maps.get(str(row.get("map_id", "")))
        if rec is None:
            continue
        pos = tuple(int(x) for x in row["pos"])
        maze = cache.get(rec.id)
        if maze is None:
            maze = rec.to_maze()
            cache[rec.id] = maze
        if not maze.is_free(*pos) or pos == maze.goal:
            continue
        acts = optimal_actions(maze, pos)
        out.append(StateItem(
            id=str(row.get("id", f"nat-{len(out):04d}")),
            map_id=rec.id,
            level=rec.level,
            maze=maze,
            pos=pos,
            state=render_state(maze, pos, style=obs_style),
            optimal=frozenset(acts),
            action_class=action_class(acts),
            distance=maze.distance(pos),
            manhattan=manhattan(pos, maze.goal),
        ))
    if limit > 0:
        out = out[:limit]
    return out


def sample_train_items(n: int, items: list[StateItem],
                       rng: random.Random) -> list[StateItem]:
    if not items:
        raise RuntimeError("empty balanced training state pool")
    if n <= len(items):
        return rng.sample(items, n)
    return rng.choices(items, k=n)


def prompt_ids_for(tokenizer, state_text: str) -> list[int]:
    _, ids = render_prompt(tokenizer, state_text)
    return ids


# ---------------------------------------------------------------- 单步


def classify_step(maze: Maze, pos: tuple[int, int], action: str | None):
    """返回 (kind, reward, next_pos)。

    kind:
      optimal  距离减 1
      side     距离不变
      farther  距离变大
      blocked  撞墙/越界
      invalid  工具调用格式非法
    """
    if action is None:
        return "invalid", REWARD_FORMAT + REWARD_STEP, pos
    dx, dy = ACTIONS[action]
    nxt = (pos[0] + dx, pos[1] + dy)
    if not maze.is_free(*nxt):
        return "blocked", REWARD_COLLISION + REWARD_STEP, pos
    delta = maze.distance(pos) - maze.distance(nxt)
    reward = REWARD_DISTANCE * delta + REWARD_STEP
    if nxt == maze.goal:
        return "optimal", reward + REWARD_TERMINAL, nxt
    if delta > 0:
        return "optimal", reward, nxt
    if delta == 0:
        return "side", reward, nxt
    return "farther", reward, nxt


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    counts = Counter(r["kind"] for r in rows)
    valid = n - counts.get("invalid", 0)
    actions = Counter((r.get("action") or "invalid") for r in rows)

    def rate(x: int) -> float:
        return x / n if n else 0.0

    return {
        "n": n,
        "valid": valid,
        "optimal": counts.get("optimal", 0),
        "side": counts.get("side", 0),
        "farther": counts.get("farther", 0),
        "blocked": counts.get("blocked", 0),
        "invalid": counts.get("invalid", 0),
        "valid_rate": rate(valid),
        "optimal_rate": rate(counts.get("optimal", 0)),
        "side_rate": rate(counts.get("side", 0)),
        "farther_rate": rate(counts.get("farther", 0)),
        "blocked_rate": rate(counts.get("blocked", 0)),
        "invalid_rate": rate(counts.get("invalid", 0)),
        "mean_reward": statistics.mean([r["reward"] for r in rows]) if rows else 0.0,
        "action_counts": dict(actions),
    }


def evaluate(engine, tokenizer, states: list[StateItem], *,
             max_tokens: int, g: int) -> tuple[dict, list[dict]]:
    rows: list[dict] = []
    t0 = time.perf_counter()
    sampling = Sampling(max_tokens=max_tokens, temperature=0.0)
    for i in range(0, len(states), g):
        batch = states[i:i + g]
        prompts = [prompt_ids_for(tokenizer, s.state) for s in batch]
        outs = engine.generate_batch(prompts, sampling)
        for st, out in zip(batch, outs):
            text = tokenizer.decode(out.token_ids, skip_special_tokens=True)
            action = parse_action(text)
            kind, reward, _ = classify_step(st.maze, st.pos, action)
            optimal = action is not None and action in st.optimal
            if optimal:
                kind = "optimal"
            rows.append({
                "id": st.id,
                "map_id": st.map_id,
                "level": st.level,
                "action": action,
                "kind": kind,
                "optimal": optimal,
                "reward": reward,
                "distance": st.distance,
            })
    summary = summarize(rows)
    summary["elapsed_s"] = time.perf_counter() - t0
    return summary, rows


# ---------------------------------------------------------------- 训练


def _fmt(v, digits=3):
    return "—" if v is None else f"{v:.{digits}f}"


def _prefix_metrics(ev: dict, prefix: str) -> dict:
    keys = (
        "n", "valid", "optimal", "side", "farther", "blocked", "invalid",
        "valid_rate", "optimal_rate", "side_rate", "farther_rate",
        "blocked_rate", "invalid_rate", "mean_reward", "action_counts",
        "elapsed_s",
    )
    return {f"{prefix}_{k}": ev[k] for k in keys if k in ev}


def _sample_frame(st: StateItem, sample_index: int, text: str, action,
                  reward: float, kind: str, optimal: bool, token_ids: list[int],
                  logprob: float, latency: float, finish: str,
                  group_size: int, unique_actions: int,
                  action_counts: dict, group_std: float, update: int) -> dict:
    return build_frame(
        st.maze, st.pos, path=[st.pos], mode="live",
        info={
            "action": action,
            "reward": round(reward, 4),
            "kind": kind,
            "optimal": optimal,
            "success": optimal,
            "total_reward": round(reward, 4),
            "steps": 1,
            "reason": kind,
            "group_size": group_size,
            "unique_actions": unique_actions,
            "action_counts": action_counts,
            "group_std": round(group_std, 4),
        },
        llm={
            "text": text,
            "action": action,
            "tokens": len(token_ids),
            "logprob": logprob,
            "latency": round(latency, 4),
            "finish": finish,
        },
        prompt=st.state,
        episode=f"u{update}s{sample_index // group_size}g{sample_index % group_size}",
        turn=1,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="single-step GRPO on maze states")
    ap.add_argument("--model", default=str(paths.MODEL_PATH))
    ap.add_argument("--data", default="data")
    ap.add_argument("--state-dir", default="data/states")
    ap.add_argument("--train-states", default="train.jsonl")
    ap.add_argument("--val-states", default="val.jsonl")
    ap.add_argument("--test-states", default="test.jsonl")
    ap.add_argument("--natural-eval", default="",
                    help="natural eval 文件；默认 data/eval_states.jsonl")
    ap.add_argument("--natural-limit", type=int, default=0,
                    help="natural eval 最多用几条；0 = 全部")
    ap.add_argument("--levels", default="L0",
                    help="逗号分隔，如 L0 或 L0,L1")
    ap.add_argument("--updates", type=int, default=20)
    ap.add_argument("--states", type=int, default=16, help="每次更新的状态数")
    ap.add_argument("--g", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.2)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=40)
    ap.add_argument("--max-seq-len", type=int, default=1024)
    ap.add_argument("--kv-ratio", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--beta", type=float, default=0.0)
    ap.add_argument("--max-logp-delta", type=float, default=0.5)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--obs-style", default="minimal")
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--target-optimal", type=float, default=0.90,
                    help="val optimal 达到该值后提前停止；<=0 关闭")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--save-dir", default="out/phase1")
    ap.add_argument("--save", default="out/lora_phase1.pt")
    ap.add_argument("--json", default="out/phase1_log.json")
    ap.add_argument("--jsonl", default="out/phase1.jsonl")
    ap.add_argument("--ui-url", default=None)
    ap.add_argument("--ui-host", default="127.0.0.1")
    ap.add_argument("--ui-port", type=int, default=7862)
    ap.add_argument("--no-ui", action="store_true")
    ap.add_argument("--no-grad-checkpoint", action="store_true")
    args = ap.parse_args()

    levels = tuple(x.strip().upper() for x in args.levels.split(",") if x.strip())
    bad = [x for x in levels if x not in LEVELS]
    if bad:
        ap.error(f"unknown level(s): {bad}; expected subset of {LEVELS}")
    if not levels:
        ap.error("--levels is empty")

    data_dir = Path(args.data)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.jsonl).parent.mkdir(parents=True, exist_ok=True)

    state_dir = Path(args.state_dir)
    train_items = load_state_items(
        state_dir, data_dir, args.train_states, "train", levels, args.obs_style)
    val_items = load_state_items(
        state_dir, data_dir, args.val_states, "val", levels, args.obs_style)
    test_items = load_state_items(
        state_dir, data_dir, args.test_states, "test_id", levels, args.obs_style)

    natural_path = Path(args.natural_eval) if args.natural_eval else (
        data_dir / "eval_states.jsonl")
    natural_items = load_natural_items(
        data_dir, levels, args.obs_style, args.natural_limit,
        path=natural_path) if natural_path.exists() else []

    viz = None
    if not args.no_ui:
        viz = TrainViz(args.ui_url, host=args.ui_host, port=args.ui_port,
                       serve=args.ui_url is None)
        print(f"UI: {viz.url}")

    print(f"levels={levels}  train={len(train_items)}  val={len(val_items)}  "
          f"test={len(test_items)}  natural={len(natural_items)}  "
          f"states={state_dir}")
    print(f"加载模型 {args.model}")
    model, tokenizer, device = load_model(args.model)
    model.eval()
    tie_embeddings(model)

    engine = None
    try:
        engine = RolloutEngine(
            model, tokenizer, max_batch=args.g,
            max_seq_len=args.max_seq_len, kv_ratio=args.kv_ratio,
            device=device,
        )
        adapter = LoRAAdapter(model, r=args.lora_r, alpha=2 * args.lora_r)
        n_train = sum(p.numel() for p in adapter.trainable_parameters())
        print(f"LoRA params: {n_train/1e6:.2f}M  (r={args.lora_r})")
        trainer = GRPOTrainer(adapter, TrainConfig(
            lr=args.lr, clip_eps=args.clip, beta=args.beta,
            grad_checkpoint=not args.no_grad_checkpoint,
            max_logp_delta=args.max_logp_delta,
        ))

        train_sampling = Sampling(
            max_tokens=args.max_tokens, temperature=args.temperature,
            top_k=args.top_k, top_p=args.top_p,
        )

        hist: list[dict] = []
        base_rows = {
            "update": 0,
            "total_updates": args.updates,
            "levels": list(levels),
            "obs_style": args.obs_style,
            "g": args.g,
            "temperature": args.temperature,
            "train_states": len(train_items),
            "val_states": len(val_items),
            "test_states": len(test_items),
            "natural_states": len(natural_items),
        }

        val_ev, _ = evaluate(engine, tokenizer, val_items,
                             max_tokens=args.max_tokens, g=args.g)
        test_ev, _ = evaluate(engine, tokenizer, test_items,
                              max_tokens=args.max_tokens, g=args.g)
        nat_ev = (evaluate(engine, tokenizer, natural_items,
                           max_tokens=args.max_tokens, g=args.g)[0]
                  if natural_items else None)
        base_metrics = {
            **base_rows,
            **_prefix_metrics(val_ev, "eval"),
            **_prefix_metrics(test_ev, "test"),
        }
        if nat_ev is not None:
            base_metrics.update(_prefix_metrics(nat_ev, "natural"))
        hist.append(base_metrics)
        print(f"[val  @0] valid={val_ev['valid_rate']:.2%} "
              f"optimal={val_ev['optimal_rate']:.2%} "
              f"farther={val_ev['farther_rate']:.2%} "
              f"reward={val_ev['mean_reward']:+.3f}")
        print(f"[test @0] valid={test_ev['valid_rate']:.2%} "
              f"optimal={test_ev['optimal_rate']:.2%} "
              f"farther={test_ev['farther_rate']:.2%} "
              f"reward={test_ev['mean_reward']:+.3f}")
        if nat_ev is not None:
            print(f"[nat  @0] valid={nat_ev['valid_rate']:.2%} "
                  f"optimal={nat_ev['optimal_rate']:.2%} "
                  f"farther={nat_ev['farther_rate']:.2%} "
                  f"reward={nat_ev['mean_reward']:+.3f}")
        if viz is not None:
            viz.update(**base_metrics)

        jsonl_fp = Path(args.jsonl).open("w", encoding="utf-8")
        jsonl_fp.write(json.dumps(base_metrics, ensure_ascii=False) + "\n")
        jsonl_fp.flush()

        best_optimal = val_ev["optimal_rate"]
        t_start = time.time()
        skipped_groups = 0
        try:
            for u in range(1, args.updates + 1):
                # 确定性种子：不要用 tuple.__hash__（受 PYTHONHASHSEED 影响）
                rng = random.Random(args.seed * 100003 + u)
                states = sample_train_items(args.states, train_items, rng)
                samples: list[Sample] = []
                all_rows: list[dict] = []
                group_std: list[float] = []
                group_unique: list[int] = []
                group_count = 0
                skipped_this = 0

                for si, st in enumerate(states):
                    ids = prompt_ids_for(tokenizer, st.state)
                    t0 = time.perf_counter()
                    outs = engine.generate_batch(
                        [ids] * args.g, train_sampling)
                    latency = time.perf_counter() - t0
                    rewards: list[float] = []
                    rows: list[dict] = []
                    for gi, out in enumerate(outs):
                        text = tokenizer.decode(out.token_ids,
                                                skip_special_tokens=True)
                        action = parse_action(text)
                        kind, reward, _ = classify_step(st.maze, st.pos, action)
                        optimal = action is not None and action in st.optimal
                        if optimal:
                            kind = "optimal"
                        rewards.append(reward)
                        rows.append({
                            "action": action,
                            "kind": kind,
                            "optimal": optimal,
                            "reward": reward,
                        })

                    advs, degen = group_advantages(rewards)
                    std = statistics.pstdev(rewards)
                    unique = len({a for a in (r["action"] for r in rows)
                                  if a in ACTIONS})
                    action_counts = dict(Counter(
                        (r["action"] or "invalid") for r in rows))
                    group_count += 1
                    group_std.append(std)
                    group_unique.append(unique)
                    all_rows.extend(rows)

                    if viz is not None:
                        for gi, (out, r) in enumerate(zip(outs, rows)):
                            viz.publish(_sample_frame(
                                st, si * args.g + gi,
                                tokenizer.decode(out.token_ids,
                                                 skip_special_tokens=True),
                                r["action"], r["reward"], r["kind"],
                                r["optimal"], out.token_ids,
                                out.mean_logprob, latency, out.finish_reason,
                                args.g, unique, action_counts, std, u,
                            ))

                    if degen:
                        skipped_groups += 1
                        skipped_this += 1
                        continue
                    for out, adv in zip(outs, advs):
                        if not out.token_ids:
                            continue
                        samples.append(Sample(
                            ids, list(out.token_ids),
                            list(out.logprobs), float(adv),
                        ))

                stats = trainer.update(samples) if samples else None
                summary = summarize(all_rows)
                metrics = {
                    **base_rows,
                    "update": u,
                    "states": len(states),
                    "episodes": len(all_rows),
                    "samples": len(samples),
                    "success": summary["optimal"],
                    "optimal": summary["optimal"],
                    "side": summary["side"],
                    "farther": summary["farther"],
                    "blocked": summary["blocked"],
                    "invalid": summary["invalid"],
                    "valid": summary["valid"],
                    "valid_rate": summary["valid_rate"],
                    "optimal_rate": summary["optimal_rate"],
                    "side_rate": summary["side_rate"],
                    "farther_rate": summary["farther_rate"],
                    "blocked_rate": summary["blocked_rate"],
                    "invalid_rate": summary["invalid_rate"],
                    "mean_reward": summary["mean_reward"],
                    "mean_steps": 1.0,
                    "collisions": summary["blocked"],
                    "format_errors": summary["invalid"],
                    "degenerate": skipped_this,
                    "degenerate_rate": (skipped_this / group_count
                                        if group_count else 0.0),
                    "unique_actions": (statistics.mean(group_unique)
                                       if group_unique else 0.0),
                    "group_std": (statistics.mean(group_std)
                                  if group_std else 0.0),
                    "action_counts": summary["action_counts"],
                }
                if stats is not None:
                    metrics.update({
                        "loss": stats.loss,
                        "pg": stats.pg,
                        "kl": stats.kl,
                        "ratio_min": stats.ratio_min,
                        "ratio_max": stats.ratio_max,
                        "clip_frac": stats.clip_frac,
                        "drop_frac": stats.drop_frac,
                        "grad_norm": stats.grad_norm,
                    })

                stop_early = False
                if args.eval_every > 0 and (
                        u % args.eval_every == 0 or u == args.updates):
                    val_ev, _ = evaluate(engine, tokenizer, val_items,
                                         max_tokens=args.max_tokens, g=args.g)
                    test_ev, _ = evaluate(engine, tokenizer, test_items,
                                          max_tokens=args.max_tokens, g=args.g)
                    nat_ev = (evaluate(engine, tokenizer, natural_items,
                                       max_tokens=args.max_tokens,
                                       g=args.g)[0]
                              if natural_items else None)
                    metrics.update(_prefix_metrics(val_ev, "eval"))
                    metrics.update(_prefix_metrics(test_ev, "test"))
                    if nat_ev is not None:
                        metrics.update(_prefix_metrics(nat_ev, "natural"))
                    trainer.save(save_dir / f"ckpt_u{u:04d}.pt")
                    if val_ev["optimal_rate"] > best_optimal:
                        best_optimal = val_ev["optimal_rate"]
                        trainer.save(save_dir / "best.pt")
                    print(f"      [val @{u}] valid={val_ev['valid_rate']:.2%} "
                          f"optimal={val_ev['optimal_rate']:.2%} "
                          f"farther={val_ev['farther_rate']:.2%} "
                          f"reward={val_ev['mean_reward']:+.3f}")
                    print(f"      [test@{u}] valid={test_ev['valid_rate']:.2%} "
                          f"optimal={test_ev['optimal_rate']:.2%} "
                          f"farther={test_ev['farther_rate']:.2%} "
                          f"reward={test_ev['mean_reward']:+.3f}")
                    if nat_ev is not None:
                        print(f"      [nat @{u}] valid={nat_ev['valid_rate']:.2%} "
                              f"optimal={nat_ev['optimal_rate']:.2%} "
                              f"farther={nat_ev['farther_rate']:.2%} "
                              f"reward={nat_ev['mean_reward']:+.3f}")
                    if (args.target_optimal > 0
                            and val_ev["optimal_rate"] >= args.target_optimal):
                        stop_early = True
                        print(f"      target val optimal={args.target_optimal:.0%} "
                              f"已达到，提前停止")

                hist.append(metrics)
                jsonl_fp.write(json.dumps(metrics, ensure_ascii=False) + "\n")
                jsonl_fp.flush()
                if viz is not None:
                    viz.update(**metrics)
                print(f"[{u:3d}] states={len(states)} n={len(samples)} "
                      f"skip={skipped_this} opt={summary['optimal_rate']:.2f} "
                      f"farther={summary['farther_rate']:.2f} "
                      f"deg={metrics['degenerate_rate']:.2f} "
                      f"loss={_fmt(metrics.get('loss'))} "
                      f"ratio=[{_fmt(metrics.get('ratio_min'), 3)},"
                      f"{_fmt(metrics.get('ratio_max'), 3)}] "
                      f"gnorm={_fmt(metrics.get('grad_norm'))}")
                if stop_early:
                    break
        finally:
            jsonl_fp.close()

        dt = time.time() - t_start
        print(f"\n总耗时 {dt/60:.1f} min，退化组 {skipped_groups}，"
              f"best eval optimal={best_optimal:.2%}")
        trainer.save(save_dir / "last.pt")
        if args.save:
            trainer.save(args.save)
        Path(args.json).write_text(
            json.dumps(hist, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"adapter → {save_dir/'last.pt'}")
        print(f"日志 → {args.jsonl} / {args.json}")
        return 0
    finally:
        if engine is not None:
            engine.close()
        if viz is not None:
            viz.close()


if __name__ == "__main__":
    raise SystemExit(main())
