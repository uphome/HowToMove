"""端到端：真模型 → 真环境 → 浏览器实时轨迹。

    # 1) 先起 UI（另开一个终端也行）
    python -m env.ui --port 7861

    # 2) 用训练环境跑真 rollout，把每一帧推到那个 UI
    conda run -n qwen3 python scripts/live_rollout.py --episodes 3 --level L3

    # 或者自带 UI（占 7862，避免和已有实例冲突）
    ... python scripts/live_rollout.py --serve
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import paths  # noqa: E402,F401

from engine_rl import RolloutEngine, Sampling, load_model  # noqa: E402
from engine_policy import EnginePolicy  # noqa: E402
from env.maze import LEVELS, generate  # noqa: E402
from env.ui import UIServer, VizClient, build_frame  # noqa: E402
from rollout import run_episode  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(paths.MODEL_PATH))
    ap.add_argument("--level", default=None, choices=LEVELS)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--delay", type=float, default=0.0, help="每步额外停顿（看动画用）")
    ap.add_argument("--obs-style", default="minimal",
                    choices=["minimal", "rel", "grid", "coords", "rows", "bitmap"])
    ap.add_argument("--memory", type=int, default=3, help="last 动作记忆条数")
    ap.add_argument("--visited", action="store_true", help="把走过的格子标成 ','")
    ap.add_argument("--serve", action="store_true", help="本进程内起 UI")
    ap.add_argument("--ui-url", default="http://127.0.0.1:7861")
    ap.add_argument("--max-batch", type=int, default=8)
    ap.add_argument("--max-seq-len", type=int, default=768)
    args = ap.parse_args()

    ui = None
    if args.serve:
        server = UIServer(port=7862).start()
        ui = server
        print(f"UI: {server.url}")
    else:
        client = VizClient(args.ui_url)
        if client.health():
            ui = client
            print(f"UI: {args.ui_url}（已连接）")
        else:
            print(f"UI: {args.ui_url} 不可用，仅终端输出")

    print(f"加载模型 {args.model}")
    model, tokenizer, device = load_model(args.model)
    engine = RolloutEngine(
        model, tokenizer,
        max_batch=args.max_batch, max_seq_len=args.max_seq_len,
        kv_ratio=0.5, device=device,
    )
    print(f"KV 池: {engine.pool_stats}")
    policy = EnginePolicy(engine, tokenizer, Sampling(
        max_tokens=args.max_tokens, temperature=args.temperature))

    rng = random.Random(args.seed)
    ok = 0
    try:
        for i in range(args.episodes):
            level = args.level or rng.choice(LEVELS)
            seed = args.seed if args.seed is not None else rng.randrange(10**6)
            maze = generate(level=level, seed=seed)
            tag = f"live-{i + 1}"
            t0 = time.time()
            traj = run_episode(
                maze, policy, ui=ui, delay=args.delay, episode=tag,
                obs_style=args.obs_style, memory=args.memory,
                show_visited=args.visited,
            )
            dt = time.time() - t0
            ok += int(traj.success)
            s = traj.summary()
            print(f"[{i + 1}/{args.episodes}] {level} seed={seed:<7} "
                  f"steps={s['steps']:<2} success={s['success']!s:<5} "
                  f"reward={s['total_reward']:7.2f} "
                  f"shortest={s['shortest']:<2} "
                  f"collisions={s['collisions']:<2} "
                  f"fmterr={s['format_errors']} "
                  f"{dt:5.1f}s")
            if ui is not None:
                ui.publish(build_frame(
                    maze,
                    traj.maze.start if not traj.turns else traj.turns[-1].pos,
                    path=traj.path, mode="done", episode=tag,
                    info={"success": traj.success,
                          "total_reward": round(traj.total_reward, 2),
                          "steps": traj.steps},
                    turn=traj.steps,
                ))
    finally:
        print(f"策略统计: {policy.stats}")
        engine.close()
        if args.serve:
            server.stop()

    print(f"success {ok}/{args.episodes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
