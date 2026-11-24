"""Custom-map playground: validation and model-backed rollout runner.

The browser page lives in ``env/playground_page.py``.  This module deliberately
keeps the heavy inference imports lazy so importing ``env.ui`` does not load
torch or allocate GPU memory until the user actually starts the model.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any, Callable

import paths  # noqa: F401  (bootstraps the vendored engine)

from .maze import Maze, bfs_field

MIN_SIZE = 4
MAX_SIZE = 40
MIN_MAX_STEPS = 1
MAX_MAX_STEPS = 300


def _coord(value: Any, name: str, width: int, height: int) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must be [x, y]")
    try:
        x, y = int(value[0]), int(value[1])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain integers") from exc
    if not (0 <= x < width and 0 <= y < height):
        raise ValueError(f"{name} is outside the map: {(x, y)}")
    return x, y


def parse_playground_map(payload: dict[str, Any]) -> Maze:
    """Validate a browser-authored map and rebuild its BFS distance field."""
    try:
        width = int(payload.get("width", 12))
        height = int(payload.get("height", 12))
    except (TypeError, ValueError) as exc:
        raise ValueError("width and height must be integers") from exc
    if not (MIN_SIZE <= width <= MAX_SIZE):
        raise ValueError(f"width must be between {MIN_SIZE} and {MAX_SIZE}")
    if not (MIN_SIZE <= height <= MAX_SIZE):
        raise ValueError(f"height must be between {MIN_SIZE} and {MAX_SIZE}")

    start = _coord(payload.get("start"), "start", width, height)
    goal = _coord(payload.get("goal"), "goal", width, height)
    if start == goal:
        raise ValueError("start and goal must be different")

    raw_walls = payload.get("walls", [])
    if not isinstance(raw_walls, list):
        raise ValueError("walls must be a list of [x, y] coordinates")
    walls: set[tuple[int, int]] = set()
    for entry in raw_walls:
        cell = _coord(entry, "wall", width, height)
        walls.add(cell)
    if start in walls:
        raise ValueError("start cannot be a wall")
    if goal in walls:
        raise ValueError("goal cannot be a wall")

    dist = bfs_field(width, height, walls, goal)
    shortest = dist[start[1]][start[0]]
    if shortest >= 10**9:
        raise ValueError("goal is unreachable from start")

    try:
        max_steps = int(payload.get("max_steps", max(30, shortest + 5)))
    except (TypeError, ValueError) as exc:
        raise ValueError("max_steps must be an integer") from exc
    if not (MIN_MAX_STEPS <= max_steps <= MAX_MAX_STEPS):
        raise ValueError(
            f"max_steps must be between {MIN_MAX_STEPS} and {MAX_MAX_STEPS}")
    if max_steps < shortest:
        raise ValueError(
            f"max_steps={max_steps} is shorter than the shortest path "
            f"({shortest})")
    if shortest < 1:
        raise ValueError("start and goal must not be the same cell")

    return Maze(
        width=width,
        height=height,
        walls=frozenset(walls),
        start=start,
        goal=goal,
        max_steps=max_steps,
        level="CUSTOM",
        seed=None,
        dist=dist,
    )


def _bounded_number(payload: dict[str, Any], key: str, default: float,
                    lo: float, hi: float, *, cast=float) -> Any:
    try:
        value = cast(payload.get(key, default))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be numeric") from exc
    if not (lo <= value <= hi):
        raise ValueError(f"{key} must be between {lo} and {hi}")
    return value


def parse_run_config(payload: dict[str, Any], *,
                     default_model: str,
                     default_adapter: str) -> dict[str, Any]:
    model = str(payload.get("model") or default_model).strip()
    adapter = str(payload.get("adapter") or default_adapter).strip()
    if not model:
        raise ValueError("model path is empty")
    return {
        "model": model,
        "adapter": adapter or None,
        "temperature": _bounded_number(
            payload, "temperature", 0.0, 0.0, 5.0),
        "top_p": _bounded_number(payload, "top_p", 1.0, 0.01, 1.0),
        "max_tokens": _bounded_number(
            payload, "max_tokens", 48, 1, 512, cast=int),
        "history_window": _bounded_number(
            payload, "history_window", 12, 1, 64, cast=int),
        "max_seq_len": _bounded_number(
            payload, "max_seq_len", 4096, 64, 16384, cast=int),
        "seed": _bounded_number(
            payload, "seed", 7, 0, 2**31 - 1, cast=int),
    }


class PlaygroundRunner:
    """Lazy single-model, single-trajectory runner.

    Model loading is intentionally serialized.  The HTTP layer exposes status
    while loading, and only one rollout may be active at a time.
    """

    def __init__(self, *, default_model: str | None = None,
                 default_adapter: str | None = None):
        self.default_model = str(
            default_model or os.environ.get("PLAYGROUND_MODEL", paths.MODEL_PATH))
        self.default_adapter = str(
            default_adapter or os.environ.get("PLAYGROUND_ADAPTER", "")).strip()
        self._lock = threading.RLock()
        self._state = "idle"
        self._last_error: str | None = None
        self._model = None
        self._tokenizer = None
        self._device = None
        self._adapter = None
        self._loaded_model: str | None = None
        self._loaded_adapter: str | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._last_episode: str | None = None
        self._last_summary: dict[str, Any] | None = None
        self._updated = time.time()

    # ---- status ---------------------------------------------------------

    def _touch(self, state: str | None = None,
               error: str | None = None) -> None:
        with self._lock:
            if state is not None:
                self._state = state
            if error is not None:
                self._last_error = error
            self._updated = time.time()

    def status(self) -> dict[str, Any]:
        with self._lock:
            alive = self._thread is not None and self._thread.is_alive()
            return {
                "state": self._state,
                "running": alive,
                "model_loaded": self._model is not None,
                "model": self.default_model,
                "adapter": self.default_adapter or None,
                "loaded_model": self._loaded_model,
                "loaded_adapter": self._loaded_adapter,
                "last_error": self._last_error,
                "last_episode": self._last_episode,
                "last_summary": self._last_summary,
                "updated": self._updated,
            }

    def config(self) -> dict[str, Any]:
        return {
            "model": self.default_model,
            "adapter": self.default_adapter or "",
            "limits": {
                "min_size": MIN_SIZE,
                "max_size": MAX_SIZE,
                "min_steps": MIN_MAX_STEPS,
                "max_steps": MAX_MAX_STEPS,
            },
        }

    def is_running(self) -> bool:
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    # ---- model lifecycle ------------------------------------------------

    def _ensure_model(self, model_path: str, adapter_path: str | None) -> None:
        """Load base model/adapter if the requested pair is not loaded.

        Public lifecycle methods guard against concurrent rollouts; the worker
        calls this after it has already acquired the single run slot.
        """
        with self._lock:
            if (self._model is not None
                    and self._loaded_model == model_path
                    and self._loaded_adapter == adapter_path):
                return
            self._state = "loading"
            self._last_error = None

        # Heavy imports stay inside the method so the UI can start without
        # importing torch / allocating CUDA memory.
        import torch
        from engine_rl import load_model
        from grpo.adapter import LoRAAdapter, tie_embeddings

        try:
            model, tokenizer, device = load_model(model_path)
            tie_embeddings(model)
            adapter = None
            if adapter_path:
                path = Path(adapter_path)
                if not path.exists():
                    raise FileNotFoundError(f"adapter not found: {path}")
                blob = torch.load(path, map_location=device)
                adapter = LoRAAdapter(model, r=16, alpha=32)
                adapter.load_state_dict(blob.get("adapter", blob))
            with self._lock:
                self._model = model
                self._tokenizer = tokenizer
                self._device = device
                self._adapter = adapter
                self._loaded_model = model_path
                self._loaded_adapter = adapter_path
                self._state = "idle"
                self._updated = time.time()
        except Exception as exc:
            with self._lock:
                self._state = "error"
                self._last_error = f"{type(exc).__name__}: {exc}"
                self._updated = time.time()
            raise

    def load(self, model_path: str | None = None,
             adapter_path: str | None = None) -> dict[str, Any]:
        if self.is_running():
            raise RuntimeError("cannot reload while a rollout is running")
        self._ensure_model(
            str(model_path or self.default_model),
            str(adapter_path).strip() if adapter_path else None,
        )
        return self.status()

    def release(self) -> dict[str, Any]:
        if self.is_running():
            raise RuntimeError("cannot release while a rollout is running")
        with self._lock:
            had_model = self._model is not None
            self._model = None
            self._tokenizer = None
            self._device = None
            self._adapter = None
            self._loaded_model = None
            self._loaded_adapter = None
            self._state = "idle"
            self._updated = time.time()
        if had_model:
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
        return self.status()

    # ---- rollout --------------------------------------------------------

    def start(self, maze: Maze, config: dict[str, Any],
              publish: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
        if self.is_running():
            raise RuntimeError("playground is already running")
        with self._lock:
            self._state = "starting"
            self._last_error = None
            self._stop = threading.Event()
            self._last_episode = f"playground-{int(time.time() * 1000)}"
            self._last_summary = None
            self._thread = threading.Thread(
                target=self._worker,
                args=(maze, config, publish, self._last_episode),
                name="playground-run",
                daemon=True,
            )
            self._thread.start()
            self._updated = time.time()
            return self.status()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        with self._lock:
            if self._state not in ("done", "error"):
                self._state = "stopping"
            self._updated = time.time()
        return self.status()

    def close(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        try:
            self.release()
        except Exception:
            pass

    def _worker(self, maze: Maze, config: dict[str, Any],
                publish: Callable[[dict[str, Any]], None],
                episode: str) -> None:
        engine = None
        try:
            from .ui import build_frame
            from agent import ChatPolicy, MazeAgent
            from engine_rl import RolloutEngine, Sampling

            self._ensure_model(config["model"], config["adapter"])
            self._touch("running")
            engine = RolloutEngine(
                self._model, self._tokenizer, max_batch=1,
                max_seq_len=int(config["max_seq_len"]),
                kv_ratio=0.5, device=self._device,
            )
            policy = ChatPolicy(
                engine, self._tokenizer,
                Sampling(
                    max_tokens=int(config["max_tokens"]),
                    temperature=float(config["temperature"]),
                    top_p=float(config["top_p"]),
                ),
            )
            agent = MazeAgent(
                maze, policy,
                history_window=int(config["history_window"]),
                episode_id=episode,
            )
            publish(build_frame(
                maze, maze.start, path=agent.path,
                mode="playground_reset", episode=episode, turn=0,
                info={"status": "running", "shortest": maze.path_length},
            ))

            while not agent.done and not self._stop.is_set():
                step = policy.act(agent.messages())
                turn = agent.apply(step)
                payload = turn["tool_result"]
                info = {
                    "status": "running",
                    "action": turn["action"],
                    "reward": turn["reward"],
                    "tool_result": payload,
                    "step": turn["turn"],
                }
                publish(build_frame(
                    maze, tuple(turn["pos"]), path=agent.path,
                    mode="playground", episode=episode,
                    turn=turn["turn"], info=info,
                    llm={
                        "text": turn["assistant_text"],
                        "action": turn["action"],
                        "logprob": turn["logprob"],
                    },
                    prompt=policy.last_prompt_text,
                ))

            run = agent.to_run()
            summary = run.summary()
            summary["stopped"] = bool(self._stop.is_set())
            summary["success"] = bool(run.success)
            summary["shortest"] = maze.path_length
            with self._lock:
                self._last_summary = summary
                self._state = "stopped" if self._stop.is_set() else "done"
                self._updated = time.time()
            publish(build_frame(
                maze, agent.env.pos, path=agent.path,
                mode="playground_done", episode=episode,
                turn=run.steps,
                info={"status": self._state, **summary},
            ))
        except Exception as exc:  # keep the HTTP server alive
            with self._lock:
                self._state = "error"
                self._last_error = f"{type(exc).__name__}: {exc}"
                self._updated = time.time()
            try:
                publish({
                    "mode": "playground_error",
                    "episode": episode,
                    "error": self._last_error,
                    "ts": time.time(),
                })
            except Exception:
                pass
        finally:
            if engine is not None:
                engine.close()
            with self._lock:
                self._thread = None
                self._updated = time.time()


__all__ = [
    "MAX_SIZE",
    "MAX_MAX_STEPS",
    "MIN_SIZE",
    "PlaygroundRunner",
    "parse_playground_map",
    "parse_run_config",
]
