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
    GET  /                 页面
    GET  /api/state        当前帧
    GET  /api/stream       SSE，订阅实时帧
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
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .maze import ACTIONS, LEVELS, Maze, MazeEnv, generate, manhattan

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
    """当前帧 + 订阅者扇出。所有 publish 都非阻塞。"""

    def __init__(self, queue_size: int = 8):
        self._lock = threading.Lock()
        self._subs: set[queue.Queue] = set()
        self._state: dict[str, Any] = {"mode": "idle", "width": 12, "height": 12}
        self._frames = 0
        self._queue_size = queue_size

    def publish(self, frame: dict[str, Any]) -> None:
        with self._lock:
            self._state = frame
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
            return dict(self._state)

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


# ---------------------------------------------------------------- page

PAGE = r"""<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Maze Live · AgenticRL</title>
<style>
  :root{
    --bg:#0f1218; --panel:#161b23; --line:#242b36; --text:#e6e9f0; --dim:#8b94a3;
    --wall:#3a4252; --free:#f8f9fc; --grid:#e6e9f0;
    --start:#5694f0; --goal:#38be78; --agent:#fa8a28; --path:#a88ce4; --bad:#f06464;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--text);
       font:14px/1.5 ui-monospace,"Cascadia Code",Consolas,monospace;}
  header{display:flex;align-items:center;gap:14px;padding:14px 20px;border-bottom:1px solid var(--line)}
  header h1{font-size:15px;font-weight:600;margin:0}
  .dot{width:8px;height:8px;border-radius:50%;background:var(--bad);transition:.2s}
  .dot.on{background:var(--goal);box-shadow:0 0 8px var(--goal)}
  .dim{color:var(--dim)}
  main{display:flex;gap:20px;padding:20px;align-items:flex-start;flex-wrap:wrap}
  .board{background:var(--panel);padding:14px;border-radius:12px;border:1px solid var(--line)}
  canvas{display:block;border-radius:8px;background:var(--free)}
  .bar{display:flex;gap:8px;margin-top:12px;flex-wrap:wrap}
  button,select,input{background:#1d2430;color:var(--text);border:1px solid var(--line);
        border-radius:8px;padding:7px 11px;font:inherit;outline:none}
  button{cursor:pointer;transition:.15s}
  button:hover{border-color:#3a4a63;background:#232c3a}
  button:active{transform:translateY(1px)}
  button.primary{background:#2a4a7a;border-color:#3a63a0}
  input[type=number]{width:92px}
  aside{flex:1;min-width:280px;max-width:520px}
  .card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px;margin-bottom:14px}
  .card h2{font-size:12px;font-weight:600;letter-spacing:.08em;text-transform:uppercase;
           color:var(--dim);margin:0 0 10px}
  .kv{display:grid;grid-template-columns:1fr 1fr;gap:6px 14px}
  .kv div{display:flex;justify-content:space-between;gap:10px}
  .kv span:last-child{color:#fff}
  #log{height:220px;overflow:auto;margin:0;font-size:12px;line-height:1.7;color:var(--dim);white-space:pre-wrap}
  #log b{color:var(--text);font-weight:500}
  .good{color:var(--goal)} .bad{color:var(--bad)} .warn{color:#f5be3c}
  .hint{font-size:12px;color:var(--dim);margin-top:8px}
  kbd{background:#1d2430;border:1px solid var(--line);border-radius:4px;padding:1px 5px;font-size:11px}
  .chips{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:8px}
  .chip{background:#1d2430;border:1px solid var(--line);border-radius:6px;
        padding:2px 7px;font-size:11px;color:var(--dim)}
  .chip b{color:var(--text);font-weight:500}
  #out{height:96px;overflow:auto;margin:0;padding:9px;background:#11161e;
       border:1px solid var(--line);border-radius:8px;font-size:12px;line-height:1.55;
       color:#cfd6e2;white-space:pre-wrap;word-break:break-all}
  #out .tag{color:#7aa2f7}
  #out .key{color:#9ece6a}
  #out .str{color:#e0af68}
  details{margin-top:8px}
  summary{cursor:pointer;font-size:12px;color:var(--dim);outline:none}
  #prompt{max-height:180px;overflow:auto;margin:6px 0 0;padding:9px;background:#11161e;
          border:1px solid var(--line);border-radius:8px;font-size:11px;line-height:1.5;
          color:#8b94a3;white-space:pre}
</style>
</head>
<body>
<header>
  <h1>Maze Live</h1>
  <span class="dot" id="dot"></span>
  <span class="dim" id="status">连接中…</span>
  <span class="dim" style="margin-left:auto" id="fps"></span>
</header>

<main>
  <section class="board">
    <canvas id="cv" width="432" height="432"></canvas>
    <div class="bar">
      <select id="level"></select>
      <input type="number" id="seed" placeholder="seed" min="0">
      <button class="primary" id="new">新地图</button>
      <button id="play">播放最短路</button>
      <button id="reset">回到起点</button>
    </div>
    <div class="hint"><kbd>W</kbd><kbd>A</kbd><kbd>S</kbd><kbd>D</kbd> 或方向键手动走；
      训练进程推送时会自动切换为实时模式。</div>
  </section>

  <aside>
    <div class="card">
      <h2>状态</h2>
      <div class="kv" id="stats"></div>
    </div>
    <div class="card">
      <h2>模型输出</h2>
      <div class="chips" id="llmchips"></div>
      <pre id="out"><span class="dim">等待模型输出…</span></pre>
      <details>
        <summary>查看 prompt（模型实际看到的）</summary>
        <pre id="prompt"></pre>
      </details>
    </div>
    <div class="card">
      <h2>事件</h2>
      <canvas id="spark" width="440" height="56" style="width:100%;height:56px"></canvas>
      <pre id="log"></pre>
    </div>
  </aside>
</main>

<script>
const COLORS = {
  free:'#f8f9fc', wall:'#3a4252', grid:'#e6e9f0',
  start:'#5694f0', goal:'#38be78', agent:'#fa8a28', path:'#a88ce4'
};
const CELL = 36;
const cv = document.getElementById('cv');
const ctx = cv.getContext('2d');
const dot = document.getElementById('dot');
const statusEl = document.getElementById('status');
const logEl = document.getElementById('log');
const statsEl = document.getElementById('stats');
const fpsEl = document.getElementById('fps');
const outEl = document.getElementById('out');
const promptEl = document.getElementById('prompt');
const llmChipsEl = document.getElementById('llmchips');

let state = null;
let target = null;          // {x,y} 目标位置（用于插值）
let shown = {x:0, y:0};     // 当前显示位置
let playing = null;
let frames = 0, fpsT = performance.now();
let rewards = [];
const spark = document.getElementById('spark');
const sctx = spark.getContext('2d');

function drawSpark(){
  const w = spark.width, h = spark.height;
  sctx.clearRect(0,0,w,h);
  sctx.strokeStyle = '#242b36'; sctx.lineWidth = 1;
  sctx.beginPath(); sctx.moveTo(0,h/2+.5); sctx.lineTo(w,h/2+.5); sctx.stroke();
  if (!rewards.length) return;
  const maxAbs = Math.max(1, ...rewards.map(Math.abs));
  const step = w / Math.max(1, rewards.length - 1);
  sctx.lineWidth = 1.5;
  for (let i=1;i<rewards.length;i++){
    const x0 = (i-1)*step, x1 = i*step;
    const y0 = h/2 - (rewards[i-1]/maxAbs)*(h/2-4);
    const y1 = h/2 - (rewards[i]/maxAbs)*(h/2-4);
    sctx.strokeStyle = rewards[i] >= 0 ? '#38be78' : '#f06464';
    sctx.beginPath(); sctx.moveTo(x0,y0); sctx.lineTo(x1,y1); sctx.stroke();
  }
  const last = rewards[rewards.length-1];
  sctx.fillStyle = last >= 0 ? '#38be78' : '#f06464';
  sctx.beginPath();
  sctx.arc((rewards.length-1)*step, h/2-(last/maxAbs)*(h/2-4), 2.5, 0, Math.PI*2);
  sctx.fill();
}

function setStatus(on, text){
  dot.classList.toggle('on', !!on);
  statusEl.textContent = text;
}

function log(text, cls){
  const line = document.createElement('div');
  const t = new Date().toLocaleTimeString('zh-CN', {hour12:false});
  line.innerHTML = `<span class="dim">${t}</span> <span class="${cls||''}">${text}</span>`;
  logEl.prepend(line);
  while (logEl.childElementCount > 160) logEl.lastChild.remove();
}

function applyState(s){
  const prev = state;
  state = s;
  frames++;
  if (!prev || prev.mode !== s.mode || prev.level !== s.level || prev.seed !== s.seed){
    if (s.level) document.getElementById('level').value = s.level;
    if (s.seed !== null && s.seed !== undefined) document.getElementById('seed').value = s.seed;
  }
  if (s.pos) { target = {x:s.pos[0], y:s.pos[1]}; }
  if (!prev) { shown = {...target}; }
  renderStats();
  renderLLM(s);
  const info = s.info || {};
  const bits = [];
  if (info.action) bits.push(`<b>${info.action}</b>`);
  if (info.reward !== undefined){
    const c = info.reward > 0 ? 'good' : 'bad';
    bits.push(`reward <span class="${c}">${info.reward.toFixed(3)}</span>`);
  }
  if (info.reason && info.reason !== 'move') bits.push(`<span class="warn">${info.reason}</span>`);
  if (info.step !== undefined) bits.push(`step ${info.step}`);
  if (typeof info.reward === 'number'){
    rewards.push(info.reward);
    if (rewards.length > 120) rewards.shift();
    drawSpark();
  }
  if (bits.length) log(bits.join(' · '));
}

function renderStats(){
  if (!state || !state.meta) { statsEl.innerHTML = ''; return; }
  const m = state.meta, info = state.info || {};
  const dist = (state.pos && state.goal)
    ? Math.abs(state.pos[0]-state.goal[0]) + Math.abs(state.pos[1]-state.goal[1]) : '—';
  const rows = [
    ['级别', state.level || '—'],
    ['种子', state.seed ?? '—'],
    ['模式', state.mode || '—'],
    ['episode', state.episode ?? '—'],
    ['turn', state.turn ?? '—'],
    ['最短路', m.shortest ?? '—'],
    ['曼哈顿', m.manhattan ?? '—'],
    ['detour', m.detour ?? '—'],
    ['墙格', m.walls ?? '—'],
    ['当前距离', m.distance_to_goal ?? '—'],
    ['到终点(曼)', dist],
    ['步数', info.steps ?? '—'],
    ['本步奖励', info.reward ?? '—'],
    ['结束原因', info.reason ?? '—'],
  ];
  statsEl.innerHTML = rows.map(([k,v]) =>
    `<div><span class="dim">${k}</span><span>${v}</span></div>`).join('');
}

function esc(s){
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
function hl(text){
  return esc(text)
    .replace(/(&lt;\/?tool_call&gt;)/g, '<span class="tag">$1</span>')
    .replace(/"([^"]+)":/g, '<span class="key">"$1"</span>:')
    .replace(/: "([^"]*)"/g, ': <span class="str">"$1"</span>');
}
function chip(k, v, cls){
  return `<span class="chip">${k} <b class="${cls||''}">${v}</b></span>`;
}

function renderLLM(s){
  const llm = s.llm;
  const info = s.info || {};
  if (s.mode === 'reset'){
    llmChipsEl.innerHTML = '';
    outEl.innerHTML = '<span class="dim">等待模型输出…</span>';
  }
  const chips = [];
  if (llm && llm.action) chips.push(chip('action', llm.action));
  if (typeof info.reward === 'number'){
    chips.push(chip('reward', info.reward.toFixed(3), info.reward >= 0 ? 'good' : 'bad'));
  }
  if (llm && llm.tokens !== undefined) chips.push(chip('tokens', llm.tokens));
  if (llm && llm.logprob !== undefined) chips.push(chip('logp', Number(llm.logprob).toFixed(3)));
  if (llm && llm.latency !== undefined) chips.push(chip('latency', `${Math.round(llm.latency*1000)}ms`));
  if (llm && llm.finish) chips.push(chip('finish', llm.finish));
  llmChipsEl.innerHTML = chips.join('');
  if (llm && llm.text !== undefined && llm.text !== null){
    outEl.innerHTML = hl(llm.text) || '<span class="dim">（空输出）</span>';
  }
  if (s.prompt){ promptEl.textContent = s.prompt; }
}

function roundRect(x,y,w,h,r){
  ctx.beginPath();
  if (ctx.roundRect) ctx.roundRect(x,y,w,h,r);
  else { ctx.moveTo(x+r,y); ctx.arcTo(x+w,y,x+w,y+h,r); ctx.arcTo(x+w,y+h,x,y+h,r);
         ctx.arcTo(x,y+h,x,y,r); ctx.arcTo(x,y,x+w,y,r); ctx.closePath(); }
}

function drawMaze(){
  if (!state || !state.width) return;
  const w = state.width, h = state.height;
  const size = w * CELL;
  if (cv.width !== size || cv.height !== h*CELL){ cv.width = size; cv.height = h*CELL; }
  ctx.fillStyle = COLORS.free;
  ctx.fillRect(0,0,cv.width,cv.height);

  ctx.fillStyle = COLORS.wall;
  for (const [x,y] of (state.walls||[])) ctx.fillRect(x*CELL, y*CELL, CELL, CELL);

  ctx.strokeStyle = COLORS.grid; ctx.lineWidth = 1;
  for (let i=0;i<=w;i++){ ctx.beginPath(); ctx.moveTo(i*CELL+.5,0); ctx.lineTo(i*CELL+.5,h*CELL); ctx.stroke(); }
  for (let j=0;j<=h;j++){ ctx.beginPath(); ctx.moveTo(0,j*CELL+.5); ctx.lineTo(w*CELL,j*CELL+.5); ctx.stroke(); }

  const path = state.path || [];
  if (path.length > 1){
    ctx.strokeStyle = COLORS.path; ctx.lineWidth = CELL*0.18;
    ctx.lineCap = 'round'; ctx.lineJoin = 'round';
    ctx.beginPath();
    path.forEach(([x,y], i) => {
      const px = x*CELL+CELL/2, py = y*CELL+CELL/2;
      i ? ctx.lineTo(px,py) : ctx.moveTo(px,py);
    });
    ctx.stroke();
  }

  const marker = ([x,y], color, inset=0.24) => {
    const p = CELL*inset;
    ctx.fillStyle = color;
    roundRect(x*CELL+p, y*CELL+p, CELL-2*p, CELL-2*p, CELL*0.16);
    ctx.fill();
  };
  if (state.start) marker(state.start, COLORS.start);
  if (state.goal) marker(state.goal, COLORS.goal);

  // agent
  const ax = shown.x*CELL+CELL/2, ay = shown.y*CELL+CELL/2;
  ctx.beginPath();
  ctx.arc(ax, ay, CELL*0.30, 0, Math.PI*2);
  ctx.fillStyle = COLORS.agent; ctx.fill();
  ctx.lineWidth = 2; ctx.strokeStyle = '#ffffff'; ctx.stroke();
}

function tick(){
  if (target){
    shown.x += (target.x - shown.x) * 0.28;
    shown.y += (target.y - shown.y) * 0.28;
    if (Math.abs(target.x-shown.x) + Math.abs(target.y-shown.y) < 0.01){
      shown.x = target.x; shown.y = target.y; target = null;
    }
  }
  drawMaze();
  const now = performance.now();
  if (now - fpsT > 1000){ fpsEl.textContent = `${frames} fps`; frames = 0; fpsT = now; }
  requestAnimationFrame(tick);
}

async function api(path, body){
  const opt = body ? {method:'POST', headers:{'Content-Type':'application/json'},
                      body:JSON.stringify(body)} : {};
  const r = await fetch(path, opt);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

document.getElementById('new').onclick = async () => {
  const level = document.getElementById('level').value;
  const seedRaw = document.getElementById('seed').value;
  const seed = seedRaw === '' ? null : Number(seedRaw);
  try { await api('/api/maze', {level, seed}); log(`新地图 ${level} seed=${seed ?? '随机'}`); }
  catch(e){ log(String(e), 'bad'); }
};

document.getElementById('reset').onclick = () => {
  if (!state) return;
  stopPlay();
  target = {x: state.start[0], y: state.start[1]};
  log('回到起点');
};

function stopPlay(){ if (playing){ clearInterval(playing); playing = null; } }

document.getElementById('play').onclick = () => {
  if (!state || !state.path || state.path.length < 2) { log('没有可播放的路径', 'warn'); return; }
  stopPlay();
  let i = 0;
  const p = state.path;
  target = {x:p[0][0], y:p[0][1]};
  playing = setInterval(() => {
    i++;
    if (i >= p.length){ stopPlay(); log(`到达终点 · ${p.length-1} 步`, 'good'); return; }
    target = {x:p[i][0], y:p[i][1]};
  }, 180);
};

const KEYMAP = {ArrowUp:'up', ArrowDown:'down', ArrowLeft:'left', ArrowRight:'right',
                w:'up', s:'down', a:'left', d:'right', W:'up', S:'down', A:'left', D:'right'};
addEventListener('keydown', async (e) => {
  const action = KEYMAP[e.key];
  if (!action) return;
  e.preventDefault();
  stopPlay();
  try { await api('/api/action', {action}); } catch(err){ log(String(err), 'bad'); }
});

function connect(){
  const es = new EventSource('/api/stream');
  es.onopen = () => { setStatus(true, '已连接'); log('SSE 已连接', 'good'); };
  es.onerror = () => { setStatus(false, '重连中…'); };
  es.onmessage = (e) => {
    try { applyState(JSON.parse(e.data)); } catch(err){ log('parse error: '+err, 'bad'); }
  };
}

(async function init(){
  const sel = document.getElementById('level');
  sel.innerHTML = ['L0','L1','L2','L3','L4']
    .map(v => `<option value="${v}">${v}</option>`).join('');
  try {
    const s = await api('/api/state');
    if (!s.width) await api('/api/maze', {level:'L3', seed:null});
    else applyState(s);
  } catch(e){ log(String(e), 'bad'); }
  connect();
  requestAnimationFrame(tick);
})();
</script>
</body>
</html>
"""


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
