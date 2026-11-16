"""实时迷宫可视化 UI：零依赖（标准库 HTTP + SSE + Canvas）。

两种用法
    1. 独立浏览
           python -m env.ui --open
       浏览器里选级别 / 种子 / 方向键手动走，或播放最短路。

    2. 训练时实时推送
           ui = UIServer(port=7861); ui.start()
           ui.show_step(maze, pos, info={"reward": 0.9})
       训练循环每步调一次，浏览器会实时刷新（SSE，不轮询）。

       trainer 与 UI 分属不同进程时，用 HTTP 推：
           VizClient("http://127.0.0.1:7861").push(frame)

HTTP 接口
    GET  /                 页面（训练看板 + 推理轨迹）
    GET  /api/state        当前帧
    GET  /api/stream       SSE，订阅实时帧
    GET  /api/train        训练指标历史 + 最新一条
    GET  /api/episodes     最近 episode 列表（含 trace 长度）
    GET  /api/trace?id=    某条 episode 的逐 turn 帧
    GET  /api/episode?id=  某条 episode 的最新帧
    POST /api/maze         {"level","seed","max_steps"} 新建
    POST /api/action       {"action"} 手动走一步
    POST /api/frame        外部推入任意帧（跨进程）
"""

from __future__ import annotations

import argparse
import json
import queue
import threading
import time
import urllib.parse
import urllib.request
import webbrowser
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .maze import ACTIONS, LEVELS, Maze, MazeEnv, generate, manhattan
from .ui_page import PAGE

DEFAULT_PORT = 7861


# ---------------------------------------------------------------- frame


def build_frame(
    maze: Maze,
    pos: tuple[int, int] | None = None,
    *,
    path: list[tuple[int, int]] | None = None,
    info: dict[str, Any] | None = None,
    mode: str = "step",
    llm: dict[str, Any] | None = None,
    prompt: str | None = None,
    episode: str | None = None,
    turn: int | None = None,
) -> dict[str, Any]:
    pos = maze.start if pos is None else pos
    return {
        "width": maze.width,
        "height": maze.height,
        "walls": sorted([int(x), int(y)] for x, y in maze.walls),
        "start": list(maze.start),
        "goal": list(maze.goal),
        "pos": list(pos),
        "path": [[int(x), int(y)] for x, y in (path or [])],
        "level": maze.level,
        "seed": maze.seed,
        "info": dict(info or {}),
        "llm": dict(llm) if llm else None,
        "prompt": prompt,
        "episode": episode,
        "turn": turn,
        "meta": {
            "shortest": maze.path_length,
            "detour": maze.detour,
            "manhattan": manhattan(maze.start, maze.goal),
            "walls": len(maze.walls),
            "max_steps": maze.max_steps,
            "distance_to_goal": maze.distance(pos),
        },
        "mode": mode,
        "ts": time.time(),
    }


# ---------------------------------------------------------------- hub


class VizHub:
    """当前帧 + 最近若干 episode/trace + 训练历史 + 订阅者扇出。

    除了“最新一帧”，这里还保留两份可回看的数据：
      - ``_train_history``：每个 update 一条训练指标（最多 500 条），供 UI 画曲线；
      - ``_traces``：每个 episode 的逐 turn 帧（最多 64 帧），供 UI 回放推理过程。
    两者都有上限，长期训练不会无限吃内存。
    """

    def __init__(self, queue_size: int = 8, max_episodes: int = 256,
                 max_turns: int = 64, max_train: int = 500):
        self._lock = threading.Lock()
        self._subs: set[queue.Queue] = set()
        self._state: dict[str, Any] = {"mode": "idle", "width": 12, "height": 12}
        self._episodes: OrderedDict[str, dict] = OrderedDict()
        self._traces: OrderedDict[str, list[dict]] = OrderedDict()
        self._train: dict[str, Any] | None = None
        self._train_history: list[dict] = []
        self._frames = 0
        self._queue_size = queue_size
        self._max_episodes = max_episodes
        self._max_turns = max_turns
        self._max_train = max_train

    def publish(self, frame: dict[str, Any]) -> None:
        with self._lock:
            if frame.get("mode") == "train":
                train = frame.get("train")
                if isinstance(train, dict):
                    self._train = train
                    self._record_train(train, frame.get("ts"))
            else:
                self._state = frame
                ep = frame.get("episode")
                if ep:
                    self._episodes.pop(ep, None)
                    self._episodes[ep] = frame
                    trace = self._traces.pop(ep, [])
                    trace.append(frame)
                    if len(trace) > self._max_turns:
                        trace = trace[-self._max_turns:]
                    self._traces[ep] = trace
                    while len(self._episodes) > self._max_episodes:
                        old_id, _ = self._episodes.popitem(last=False)
                        self._traces.pop(old_id, None)
            self._frames += 1
            subs = list(self._subs)
        for q in subs:
            try:
                q.put_nowait(frame)
            except queue.Full:
                try:
                    q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    q.put_nowait(frame)
                except queue.Full:
                    pass

    def _record_train(self, stats: dict, ts: float | None) -> None:
        """按 update 去重后追加训练指标（调用方已持有锁）。"""
        record = dict(stats)
        if ts is not None:
            record.setdefault("ts", ts)
        update = record.get("update")
        if (update is not None and self._train_history
                and self._train_history[-1].get("update") == update):
            self._train_history[-1] = record
        else:
            self._train_history.append(record)
        if len(self._train_history) > self._max_train:
            del self._train_history[: len(self._train_history) - self._max_train]

    def episodes(self) -> list[dict[str, Any]]:
        """最近的 episode 摘要，最新的在前。"""
        with self._lock:
            out = []
            for ep_id, f in reversed(self._episodes.items()):
                info = f.get("info") or {}
                out.append({
                    "id": ep_id,
                    "level": f.get("level"),
                    "turn": f.get("turn"),
                    "mode": f.get("mode"),
                    "success": info.get("success"),
                    "reward": info.get("total_reward", info.get("reward")),
                    "turns": len(self._traces.get(ep_id, [])),
                    "ts": f.get("ts"),
                })
            return out

    def episode(self, ep_id: str) -> dict[str, Any] | None:
        with self._lock:
            return self._episodes.get(ep_id)

    def trace(self, ep_id: str) -> list[dict[str, Any]]:
        """某个 episode 的逐 turn 帧（由旧到新）。"""
        with self._lock:
            return list(self._traces.get(ep_id, []))

    def train_history(self) -> list[dict[str, Any]]:
        """训练指标历史（由旧到新，每个 update 一条）。"""
        with self._lock:
            return list(self._train_history)

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=self._queue_size)
        with self._lock:
            self._subs.add(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            self._subs.discard(q)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {**self._state, "train": self._train}

    @property
    def frames(self) -> int:
        with self._lock:
            return self._frames


# ---------------------------------------------------------------- server


class UIServer:
    def __init__(self, host: str = "127.0.0.1", port: int = DEFAULT_PORT):
        self.hub = VizHub()
        self._env: MazeEnv | None = None
        self._lock = threading.Lock()
        self._httpd = ThreadingHTTPServer((host, port), _make_handler(self))
        self._httpd.daemon_threads = True
        self._thread: threading.Thread | None = None
        self.host, self.port = self._httpd.server_address[:2]

    # ---- 生命周期 ----

    def start(self) -> "UIServer":
        if self._thread is None:
            self._thread = threading.Thread(
                target=self._httpd.serve_forever, name="maze-ui", daemon=True
            )
            self._thread.start()
        return self

    def stop(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/"

    def __enter__(self) -> "UIServer":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    # ---- 供训练循环调用 ----

    def set_maze(
        self, level: str = "L3", seed: int | None = None, max_steps: int = 30
    ) -> Maze:
        maze = generate(level=level, seed=seed, max_steps=max_steps)
        with self._lock:
            self._env = MazeEnv(maze)
        self.hub.publish(build_frame(maze, maze.start, mode="reset"))
        return maze

    def publish(self, frame: dict[str, Any]) -> None:
        """发布一帧（UIServer / VizHub / VizClient 都支持这个接口）。"""
        self.hub.publish(frame)

    def publish_train(self, stats: dict[str, Any]) -> None:
        """发布训练指标帧（不覆盖地图状态）。"""
        self.hub.publish({"mode": "train", "train": dict(stats),
                          "ts": time.time()})

    def show_step(
        self,
        maze: Maze,
        pos: tuple[int, int],
        info: dict[str, Any] | None = None,
        *,
        path: list[tuple[int, int]] | None = None,
        llm: dict[str, Any] | None = None,
        prompt: str | None = None,
        episode: str | None = None,
        turn: int | None = None,
    ) -> None:
        self.hub.publish(build_frame(
            maze, pos, path=path, info=info, mode="live",
            llm=llm, prompt=prompt, episode=episode, turn=turn,
        ))

    def show_episode(
        self,
        maze: Maze,
        frames: list[tuple[int, int]],
        info: dict[str, Any] | None = None,
        *,
        path: list[tuple[int, int]] | None = None,
        episode: str | None = None,
    ) -> None:
        self.hub.publish(build_frame(
            maze, frames[-1] if frames else maze.start,
            path=path or frames, info=info, mode="episode", episode=episode,
        ))

    # ---- 交互 ----

    def step(self, action: str) -> dict[str, Any]:
        with self._lock:
            env = self._env
            if env is None:
                raise ValueError("no maze loaded")
            if env.done:
                return {"done": True, "pos": list(env.pos), "info": {"reason": "already_done"}}
            result = env.step(action)
            maze = env.maze
        frame = build_frame(
            maze,
            result.pos,
            path=maze.shortest_path(),
            info={"action": action, "reward": round(result.reward, 3), **result.info},
            mode="manual",
        )
        self.hub.publish(frame)
        return {"done": result.done, "pos": list(result.pos), "info": frame["info"]}


class _Handler(BaseHTTPRequestHandler):
    server_version = "MazeUI/1.0"

    def __init__(self, *args, server_ref: UIServer, **kwargs):
        self.ui = server_ref
        super().__init__(*args, **kwargs)

    def log_message(self, fmt, *args):
        pass

    # ---- helpers ----

    def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj: Any):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def _body(self) -> dict:
        try:
            n = int(self.headers.get("Content-Length", 0))
        except (TypeError, ValueError):
            n = 0
        if n <= 0:
            return {}
        try:
            data = json.loads(self.rfile.read(n).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return {}
        return data if isinstance(data, dict) else {}

    # ---- routes ----

    def do_GET(self):
        route = self.path.split("?")[0]
        if route in ("/", "/index.html"):
            self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
        elif route == "/api/state":
            self._json(200, self.ui.hub.snapshot())
        elif route == "/api/stream":
            self._stream()
        elif route == "/api/episodes":
            self._json(200, {"episodes": self.ui.hub.episodes()})
        elif route == "/api/episode":
            q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            ep_id = (q.get("id") or [""])[0]
            frame = self.ui.hub.episode(ep_id)
            if frame is None:
                self._json(404, {"error": f"unknown episode {ep_id!r}"})
            else:
                self._json(200, frame)
        elif route == "/api/trace":
            q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            ep_id = (q.get("id") or [""])[0]
            frames = self.ui.hub.trace(ep_id)
            if not frames:
                self._json(404, {"error": f"unknown episode {ep_id!r}"})
            else:
                self._json(200, {"episode": ep_id, "frames": frames})
        elif route == "/api/train":
            latest = self.ui.hub.snapshot().get("train")
            self._json(200, {"latest": latest,
                             "history": self.ui.hub.train_history()})
        elif route == "/api/health":
            self._json(200, {"ok": True, "frames": self.ui.hub.frames})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        route = self.path.split("?")[0]
        body = self._body()
        try:
            if route == "/api/maze":
                level = str(body.get("level", "L3")).upper()
                if level not in LEVELS:
                    self._json(400, {"error": f"unknown level {level}"})
                    return
                seed = body.get("seed")
                seed = None if seed in (None, "") else int(seed)
                maze = self.ui.set_maze(level, seed, int(body.get("max_steps", 30)))
                self._json(200, {"level": maze.level, "seed": maze.seed,
                                 "meta": self.ui.hub.snapshot().get("meta", {})})
            elif route == "/api/action":
                action = str(body.get("action", ""))
                self._json(200, self.ui.step(action))
            elif route == "/api/frame":
                self.ui.publish(body)
                self._json(200, {"ok": True})
            else:
                self._json(404, {"error": "not found"})
        except ValueError as exc:
            self._json(400, {"error": str(exc)})
        except Exception as exc:  # pragma: no cover - 防御
            self._json(500, {"error": f"{type(exc).__name__}: {exc}"})

    def _stream(self):
        q = self.ui.hub.subscribe()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        try:
            self._event(self.ui.hub.snapshot())
            while True:
                try:
                    frame = q.get(timeout=15)
                except queue.Empty:
                    self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
                    continue
                self._event(frame)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            self.ui.hub.unsubscribe(q)

    def _event(self, frame: dict):
        payload = json.dumps(frame, ensure_ascii=False)
        self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
        self.wfile.flush()


def _make_handler(ui: UIServer):
    def factory(*args, **kwargs):
        return _Handler(*args, server_ref=ui, **kwargs)

    return factory


# ---------------------------------------------------------------- client


class VizClient:
    """训练进程 → UI 服务的轻量推送（跨进程用）。失败静默，绝不拖慢训练。"""

    def __init__(self, url: str = f"http://127.0.0.1:{DEFAULT_PORT}", timeout: float = 0.05):
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.failures = 0

    def _post(self, route: str, payload: dict) -> bool:
        try:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            req = urllib.request.Request(
                self.url + route, data=data,
                headers={"Content-Type": "application/json"}, method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                resp.read()
            return True
        except Exception:
            self.failures += 1
            return False

    def health(self) -> bool:
        try:
            with urllib.request.urlopen(self.url + "/api/health",
                                        timeout=self.timeout) as resp:
                return resp.status == 200
        except Exception:
            return False

    def push(self, frame: dict) -> bool:
        return self._post("/api/frame", frame)

    # 与 UIServer / VizHub 同名的接口，便于 rollout 直接传
    publish = push

    def maze(self, level: str = "L3", seed: int | None = None) -> bool:
        return self._post("/api/maze", {"level": level, "seed": seed})

    def action(self, action: str) -> bool:
        return self._post("/api/action", {"action": action})


class TrainViz:
    """训练进程用的推送封装：本地起服务或连已有服务；失败静默，不拖慢训练。"""

    def __init__(self, url: str | None = None, *, host: str = "127.0.0.1",
                 port: int = DEFAULT_PORT, serve: bool = False, quiet: bool = False):
        self.server: UIServer | None = None
        self.client: VizClient | None = None
        if serve or url is None:
            self.server = UIServer(host=host, port=port).start()
            self.url = self.server.url
            self._push = self.server.publish
        else:
            self.url = url.rstrip("/")
            self.client = VizClient(self.url)
            if not self.client.health() and not quiet:
                print(f"[viz] {self.url} 不可用，训练继续但不推送")
            self._push = self.client.push
        self.enabled = True

    def update(self, **stats) -> None:
        if self.enabled:
            self._push({"mode": "train", "train": stats, "ts": time.time()})

    def frame(self, frame: dict) -> None:
        if self.enabled:
            self._push(frame)

    # 与 UIServer / VizHub 同名，run_group / run_episode 可直接传
    publish = frame

    def close(self) -> None:
        if self.server is not None:
            self.server.stop()
            self.server = None


# ---------------------------------------------------------------- page




def _demo(
    ui: UIServer,
    level: str | None = None,
    seed: int | None = None,
    delay: float = 0.28,
    pause: float = 2.5,
) -> None:
    """循环跑 mock 策略的真实轨迹（每步推一帧，含模型原始输出）。"""
    import random

    from rollout import MockPolicy, run_episode

    rng = random.Random()
    n = 0
    while True:
        lv = level or rng.choice(LEVELS)
        sd = seed if seed is not None else rng.randrange(10**6)
        maze = generate(level=lv, seed=sd)
        n += 1
        tag = f"demo-{n}"
        traj = run_episode(maze, MockPolicy(seed=sd), ui=ui, delay=delay, episode=tag)
        ui.hub.publish(build_frame(
            maze, traj.maze.start if not traj.turns else traj.turns[-1].pos,
            path=traj.path, mode="done",
            info={"success": traj.success, "total_reward": round(traj.total_reward, 2),
                  "steps": traj.steps, "reason": "done"},
            episode=tag,
            turn=traj.steps,
        ))
        time.sleep(pause)


# ---------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="interactive maze UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--level", default="L3", choices=LEVELS)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--open", action="store_true", help="启动后自动开浏览器")
    parser.add_argument("--demo", action="store_true", help="启动后实时演示一条轨迹")
    args = parser.parse_args(argv)

    ui = UIServer(host=args.host, port=args.port).start()
    ui.set_maze(args.level, args.seed)
    print(f"Maze UI: {ui.url}  (Ctrl+C 退出)")
    if args.demo:
        threading.Thread(target=_demo, args=(ui, args.level, args.seed),
                         daemon=True).start()
    if args.open:
        webbrowser.open(ui.url)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        ui.stop()


if __name__ == "__main__":
    main()
