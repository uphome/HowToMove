"""浏览器端页面（纯 HTML/CSS/JS，无外部依赖）。

单独拆出来是因为页面本身已经比 HTTP/SSE 服务长；env/ui.py 只负责
数据模型、路由和推送，页面只消费 /api/state、/api/train、/api/trace。
"""

PAGE = r"""<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AgenticRL · Maze Train / Inference</title>
<style>
  :root{
    --bg:#0f1218; --panel:#161b23; --panel2:#11161e; --line:#242b36;
    --text:#e6e9f0; --dim:#8b94a3;
    --wall:#3a4252; --free:#f8f9fc; --grid:#e6e9f0;
    --start:#5694f0; --goal:#38be78; --agent:#fa8a28; --path:#a88ce4;
    --bad:#f06464; --warn:#f5be3c; --accent:#7aa2f7;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--text);
       font:13px/1.5 ui-monospace,"Cascadia Code",Consolas,monospace;}
  header{display:flex;align-items:center;gap:10px;padding:12px 18px;
         border-bottom:1px solid var(--line);flex-wrap:wrap}
  header h1{font-size:15px;font-weight:600;margin:0;letter-spacing:.02em}
  .dot{width:8px;height:8px;border-radius:50%;background:var(--bad);transition:.2s}
  .dot.on{background:var(--goal);box-shadow:0 0 8px var(--goal)}
  .dim{color:var(--dim)}
  .good{color:var(--goal)} .bad{color:var(--bad)} .warn{color:var(--warn)}
  .spacer{flex:1}
  .badge{font-size:11px;padding:2px 8px;border-radius:999px;
         border:1px solid var(--line);color:var(--dim);white-space:nowrap}
  .badge.live{color:var(--goal);border-color:#2c5a43}
  .badge.train{color:var(--path);border-color:#4a3d6b}
  main{display:flex;gap:18px;padding:16px 18px;align-items:flex-start;flex-wrap:wrap}
  .board{background:var(--panel);padding:12px;border-radius:12px;
         border:1px solid var(--line);flex:0 0 auto}
  canvas{display:block;border-radius:8px;background:var(--free)}
  .bar{display:flex;gap:7px;margin-top:10px;flex-wrap:wrap}
  button,select,input{background:#1d2430;color:var(--text);border:1px solid var(--line);
        border-radius:8px;padding:6px 10px;font:inherit;outline:none}
  button{cursor:pointer;transition:.15s}
  button:hover{border-color:#3a4a63;background:#232c3a}
  button:active{transform:translateY(1px)}
  button.primary{background:#2a4a7a;border-color:#3a63a0}
  input[type=number]{width:88px}
  aside{flex:1;min-width:430px;max-width:700px;max-height:calc(100vh - 90px);
        overflow-y:auto;padding-right:2px}
  .tabs{display:flex;gap:6px;margin-bottom:10px;position:sticky;top:0;
        background:var(--bg);z-index:3;padding:2px 0 8px}
  .tab{border-radius:999px;padding:5px 13px;color:var(--dim);background:#151b24}
  .tab.active{color:#fff;background:#26344d;border-color:#3a63a0}
  .pane{display:none}
  .pane.active{display:block}
  .card{background:var(--panel);border:1px solid var(--line);border-radius:12px;
        padding:12px;margin-bottom:12px}
  .card h2{font-size:11px;font-weight:600;letter-spacing:.08em;text-transform:uppercase;
           color:var(--dim);margin:0 0 10px;display:flex;gap:8px;align-items:baseline}
  .card h2 .dim{text-transform:none;letter-spacing:0;font-weight:400}
  .kpi-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:7px}
  .kpi{background:var(--panel2);border:1px solid var(--line);border-radius:8px;
       padding:7px 9px;min-width:0}
  .kpi .k{font-size:10px;color:var(--dim);white-space:nowrap;overflow:hidden;
          text-overflow:ellipsis}
  .kpi .v{font-size:15px;font-weight:600;white-space:nowrap;overflow:hidden;
          text-overflow:ellipsis}
  .kpi.good .v{color:var(--goal)} .kpi.bad .v{color:var(--bad)}
  .kpi.warn .v{color:var(--warn)}
  #direction-bars{display:flex;flex-direction:column;gap:7px}
  .dir-row{display:grid;grid-template-columns:48px 1fr 96px;gap:8px;align-items:center}
  .dir-name{font-size:11px;color:var(--dim);text-transform:uppercase}
  .dir-track{height:9px;background:#1d2430;border-radius:999px;overflow:hidden}
  .dir-fill{height:100%;border-radius:999px;background:var(--accent);transition:width .3s}
  .dir-val{font-size:11px;text-align:right;color:var(--text)}
  details.card summary{font-size:12px;color:var(--dim);cursor:pointer;outline:none}
  details.card[open] summary{color:var(--text)}
  .chart-grid{display:grid;grid-template-columns:1fr 1fr;gap:9px}
  .chart{background:var(--panel2);border:1px solid var(--line);border-radius:8px;
         padding:7px;min-width:0}
  .chart-head{display:flex;justify-content:space-between;gap:6px;align-items:center;
              font-size:10px;color:var(--dim);margin-bottom:4px;min-height:15px}
  .chart canvas{width:100%;height:125px;background:transparent}
  .legend{display:flex;gap:7px;flex-wrap:wrap;justify-content:flex-end}
  .legend span{white-space:nowrap}
  .legend i{display:inline-block;width:6px;height:6px;border-radius:50%;
            margin-right:3px;vertical-align:1px}
  .table-wrap{max-height:270px;overflow:auto;border:1px solid var(--line);
              border-radius:8px;background:var(--panel2)}
  table{width:100%;border-collapse:collapse;font-size:11px;white-space:nowrap}
  th,td{padding:5px 7px;border-bottom:1px solid var(--line);text-align:right}
  th{position:sticky;top:0;background:#141a23;color:var(--dim);font-weight:500;z-index:1}
  th:first-child,td:first-child{text-align:left}
  tbody tr:hover{background:#1a2230}
  #trace tbody tr{cursor:pointer}
  #trace tbody tr.sel{background:#23304a}
  .kv{display:grid;grid-template-columns:1fr 1fr;gap:5px 14px;font-size:12px}
  .kv div{display:flex;justify-content:space-between;gap:10px;min-width:0}
  .kv span:last-child{color:#fff;text-align:right;overflow:hidden;text-overflow:ellipsis}
  #out{height:110px;overflow:auto;margin:0;padding:8px;background:var(--panel2);
       border:1px solid var(--line);border-radius:8px;font-size:11px;line-height:1.55;
       color:#cfd6e2;white-space:pre-wrap;word-break:break-all}
  #out .tag{color:var(--accent)} #out .key{color:#9ece6a} #out .str{color:#e0af68}
  details{margin-top:7px}
  summary{cursor:pointer;font-size:11px;color:var(--dim);outline:none}
  #prompt{max-height:190px;overflow:auto;margin:6px 0 0;padding:8px;background:var(--panel2);
          border:1px solid var(--line);border-radius:8px;font-size:10px;line-height:1.5;
          color:var(--dim);white-space:pre}
  .chips{display:flex;gap:5px;flex-wrap:wrap;margin-bottom:7px;min-height:22px}
  .chip{background:#1d2430;border:1px solid var(--line);border-radius:6px;
        padding:2px 6px;font-size:10px;color:var(--dim)}
  .chip b{color:var(--text);font-weight:500}
  #log{height:150px;overflow:auto;margin:6px 0 0;font-size:11px;line-height:1.65;
       color:var(--dim);white-space:pre-wrap}
  #log b{color:var(--text);font-weight:500}
  .hint{font-size:11px;color:var(--dim);margin-top:8px}
  kbd{background:#1d2430;border:1px solid var(--line);border-radius:4px;
      padding:0 4px;font-size:10px}
  .row{display:flex;gap:7px;align-items:center}
  .row select{flex:1;min-width:0}
  #spark{width:100%;height:46px;background:transparent}
  @media(max-width:900px){
    aside{max-height:none;max-width:100%}
    .chart-grid{grid-template-columns:1fr}
    .kpi-grid{grid-template-columns:repeat(2,minmax(0,1fr))}
  }
</style>
</head>
<body>
<header>
  <h1>AgenticRL · Maze</h1>
  <span class="dot" id="dot"></span>
  <span class="dim" id="status">连接中…</span>
  <span class="badge" id="modeBadge">IDLE</span>
  <span class="badge" id="updateBadge">update —</span>
  <span class="spacer"></span>
  <a href="/playground" style="color:var(--accent);font-size:12px;text-decoration:none">Playground →</a>
  <span class="dim" id="fps"></span>
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
      训练/推理进程推送时自动切换为实时模式。</div>
  </section>

  <aside>
    <div class="tabs">
      <button class="tab active" data-pane="pane-train">训练看板</button>
      <button class="tab" data-pane="pane-infer">推理轨迹</button>
    </div>

    <section class="pane active" id="pane-train">
      <div class="card">
        <h2>关键指标 <span class="dim" id="train-updated"></span></h2>
        <div class="kpi-grid" id="key-kpis"></div>
      </div>
      <div class="card" id="direction-card" style="display:none">
        <h2>方向 singleton 准确率 <span class="dim" id="direction-meta"></span></h2>
        <div id="direction-bars"></div>
      </div>
      <div class="card" id="main-eval-card" style="display:none">
        <h2>optimal 曲线 <span class="dim">val / balanced / natural</span></h2>
        <div class="chart">
          <div class="chart-head"><span>optimal rate</span><span class="legend" id="legend-eval"></span></div>
          <canvas id="chart-eval" width="700" height="150"></canvas>
        </div>
      </div>
      <details class="card">
        <summary>训练诊断 / 全部指标</summary>
        <div class="kpi-grid" id="train" style="margin-top:10px"></div>
        <div class="chart-grid" style="margin-top:10px">
          <div class="chart">
            <div class="chart-head"><span>loss / pg</span><span class="legend" id="legend-loss"></span></div>
            <canvas id="chart-loss" width="700" height="125"></canvas>
          </div>
          <div class="chart">
            <div class="chart-head"><span>mean reward</span><span class="legend" id="legend-reward"></span></div>
            <canvas id="chart-reward" width="700" height="125"></canvas>
          </div>
          <div class="chart">
            <div class="chart-head"><span>success / episodes</span><span class="legend" id="legend-success"></span></div>
            <canvas id="chart-success" width="700" height="125"></canvas>
          </div>
          <div class="chart">
            <div class="chart-head"><span>collisions / format errors</span><span class="legend" id="legend-errors"></span></div>
            <canvas id="chart-errors" width="700" height="125"></canvas>
          </div>
          <div class="chart">
            <div class="chart-head"><span>ratio min / max</span><span class="legend" id="legend-ratio"></span></div>
            <canvas id="chart-ratio" width="700" height="125"></canvas>
          </div>
          <div class="chart">
            <div class="chart-head"><span>grad norm</span><span class="legend" id="legend-grad"></span></div>
            <canvas id="chart-grad" width="700" height="125"></canvas>
          </div>
          <div class="chart">
            <div class="chart-head"><span>单步 outcome</span><span class="legend" id="legend-outcome"></span></div>
            <canvas id="chart-outcome" width="700" height="125"></canvas>
          </div>
        </div>
      </details>
      <details class="card">
        <summary>更新历史 <span class="dim">最近 30 次</span></summary>
        <div class="table-wrap" style="margin-top:10px">
          <table id="train-history">
            <thead><tr>
              <th>upd</th><th>eps</th><th>succ</th><th>reward</th><th>steps</th>
              <th>coll</th><th>fmt</th><th>deg</th><th>n</th><th>loss</th>
              <th>pg</th><th>kl</th><th>ratio</th><th>clip</th><th>drop</th><th>gnorm</th>
            </tr></thead>
            <tbody></tbody>
          </table>
        </div>
      </details>
    </section>

    <section class="pane" id="pane-infer">
      <div class="card">
        <h2>轨迹</h2>
        <div class="row">
          <select id="eps"><option value="">（跟随最新）</option></select>
          <button id="follow">跟随最新</button>
        </div>
        <div class="row" style="margin-top:7px">
          <input id="eps-filter" placeholder="筛选 episode，如 u5s3（显示 g0~g7）"
                 style="flex:1;min-width:0">
        </div>
        <div class="kv" id="ep-summary" style="margin-top:10px"></div>
      </div>
      <div class="card">
        <h2>当前状态</h2>
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
        <h2>逐步轨迹 <span class="dim" id="trace-count"></span></h2>
        <div class="table-wrap">
          <table id="trace">
            <thead><tr>
              <th>turn</th><th>action</th><th>reward</th><th>pos</th><th>dist</th>
              <th>reason</th><th>tokens</th><th>logp</th><th>latency</th>
            </tr></thead>
            <tbody></tbody>
          </table>
        </div>
        <div class="hint">点击某一行可回看该步；之后用「跟随最新」恢复实时。</div>
      </div>
    </section>

    <div class="card">
      <h2>事件</h2>
      <canvas id="spark" width="600" height="46"></canvas>
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
const spark = document.getElementById('spark');
const sctx = spark.getContext('2d');

let state = null;
let target = null;
let shown = {x:0, y:0};
let playing = null;
let frames = 0, fpsT = performance.now();
let rewards = [];
let latestTrain = null;
let trainHistory = [];
let pinned = false;
let traceKey = '';
let traceFrames = [];
let selectedTraceTurn = null;
let epsSig = '';
let episodeList = [];

function fmtNum(v, d=2){
  if (v === undefined || v === null || v === '') return '—';
  if (typeof v === 'number') return Number.isFinite(v)
    ? (Number.isInteger(v) ? String(v) : v.toFixed(d)) : '—';
  return String(v);
}
function pct(a,b){ return b ? `${Math.round(100*a/b)}%` : '—'; }
function ratePct(v){
  return (typeof v === 'number' && Number.isFinite(v))
    ? `${(v * 100).toFixed(1)}%` : '—';
}
function tone(v){ return typeof v === 'number' ? (v > 0 ? 'good' : v < 0 ? 'bad' : '') : ''; }

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
}

function setStatus(on, text){
  dot.classList.toggle('on', !!on);
  statusEl.textContent = text;
}
function setMode(mode){
  const el = document.getElementById('modeBadge');
  const m = String(mode || 'idle').toLowerCase();
  el.textContent = m === 'train' ? 'TRAIN' : m === 'live' ? 'INFER'
    : m === 'done' ? 'DONE' : m === 'reset' ? 'RESET' : m.toUpperCase();
  el.className = 'badge ' + (m === 'train' ? 'train' : m === 'live' || m === 'done' ? 'live' : '');
}
function setUpdateBadge(u){
  document.getElementById('updateBadge').textContent =
    (u === undefined || u === null) ? 'update —' : `update ${u}`;
}

function log(text, cls){
  const line = document.createElement('div');
  const t = new Date().toLocaleTimeString('zh-CN', {hour12:false});
  line.innerHTML = `<span class="dim">${t}</span> <span class="${cls||''}">${text}</span>`;
  logEl.prepend(line);
  while (logEl.childElementCount > 160) logEl.lastChild.remove();
}

/* ------------------------------------------------------------ training */

function upsertTrain(t, ts){
  if (!t || typeof t !== 'object') return;
  const rec = {...t};
  if (ts !== undefined && rec.ts === undefined) rec.ts = ts;
  latestTrain = rec;
  const u = rec.update;
  if (u !== undefined && u !== null){
    const i = trainHistory.findIndex(x => x.update === u);
    if (i >= 0) trainHistory[i] = rec; else trainHistory.push(rec);
    trainHistory.sort((a,b) => (a.update ?? 0) - (b.update ?? 0));
  } else {
    trainHistory.push(rec);
  }
  if (trainHistory.length > 500) trainHistory = trainHistory.slice(-500);
  renderTraining();
}

function latestEvalRecord(){
  for (let i = trainHistory.length - 1; i >= 0; i--){
    const r = trainHistory[i];
    if (r.test_singleton_accuracy !== undefined
        || r.eval_singleton_accuracy !== undefined
        || r.test_optimal_rate !== undefined
        || r.eval_optimal_rate !== undefined) return r;
  }
  return latestTrain;
}

function renderTraining(){
  const t = latestTrain;
  const hasSingle = (t && (t.optimal !== undefined || t.eval_optimal_rate !== undefined))
    || trainHistory.some(r => r.optimal !== undefined || r.eval_optimal_rate !== undefined);
  const evalRec = latestEvalRecord();

  renderKeyKpis(t, hasSingle, evalRec);
  renderDiagnostics(t);

  const dirAcc = evalRec && (evalRec.test_singleton_accuracy
    || evalRec.eval_singleton_accuracy);
  const dirCard = document.getElementById('direction-card');
  dirCard.style.display = (hasSingle && dirAcc) ? '' : 'none';
  if (hasSingle && dirAcc) renderDirectionBars(evalRec);

  const evalCard = document.getElementById('main-eval-card');
  evalCard.style.display = hasSingle ? '' : 'none';

  const upd = document.getElementById('train-updated');
  upd.textContent = (t && t.ts)
    ? new Date(t.ts * 1000).toLocaleTimeString('zh-CN', {hour12:false}) : '';
  if (t && t.update !== undefined){
    setUpdateBadge(t.total_updates !== undefined
      ? `${t.update}/${t.total_updates}` : t.update);
  }
  renderTrainCharts();
  renderTrainTable();
}

function renderKeyKpis(t, hasSingle, evalRec){
  const el = document.getElementById('key-kpis');
  if (!t){
    el.innerHTML = '<div class="dim">等待训练指标…（运行 scripts/train_phase1.py 或 train_grpo.py）</div>';
    return;
  }
  const cards = [];
  if (hasSingle){
    cards.push(['balanced test optimal', ratePct(evalRec && evalRec.test_optimal_rate),
      (evalRec && evalRec.test_optimal_rate >= 0.9) ? 'good' : '']);
    cards.push(['natural test optimal', ratePct(evalRec && evalRec.natural_optimal_rate), '']);
    cards.push(['val optimal', ratePct(evalRec && evalRec.eval_optimal_rate), '']);
    cards.push(['degenerate rate', ratePct(t.degenerate_rate),
      t.degenerate_rate > 0.7 ? 'warn' : '']);
    cards.push(['valid rate', ratePct(t.valid_rate), t.valid_rate < 0.99 ? 'bad' : 'good']);
    const acc = (evalRec && (evalRec.test_singleton_accuracy
      || evalRec.eval_singleton_accuracy)) || {};
    cards.push(['down singleton', ratePct(acc.down), '']);
  } else {
    cards.push(['success', `${fmtNum(t.success,0)}${t.episodes ? ` · ${pct(t.success,t.episodes)}` : ''}`,
      t.success > 0 ? 'good' : '']);
    cards.push(['mean reward', fmtNum(t.mean_reward,2), tone(t.mean_reward)]);
    cards.push(['mean steps', fmtNum(t.mean_steps,1)]);
    cards.push(['collisions', fmtNum(t.collisions,0), t.collisions > 0 ? 'warn' : '']);
    cards.push(['format errors', fmtNum(t.format_errors,0),
      t.format_errors > 0 ? 'bad' : 'good']);
    cards.push(['degenerate rate', ratePct(t.degenerate_rate),
      t.degenerate_rate > 0.7 ? 'warn' : '']);
  }
  el.innerHTML = cards.map(([k,v,c]) =>
    `<div class="kpi ${c||''}"><div class="k">${k}</div><div class="v">${v}</div></div>`
  ).join('');
}

function renderDiagnostics(t){
  const el = document.getElementById('train');
  if (!t){
    el.innerHTML = '<div class="dim">等待训练指标…</div>';
    return;
  }
  const ratio = (t.ratio_min !== undefined && t.ratio_max !== undefined)
    ? `${fmtNum(t.ratio_min,3)}~${fmtNum(t.ratio_max,3)}` : '—';
  const progress = (t.total_updates !== undefined)
    ? `${fmtNum(t.update,0)}/${t.total_updates}` : fmtNum(t.update,0);
  const ac = t.action_counts || {};
  const actStr = ['up','down','left','right','invalid']
    .filter(k => ac[k]).map(k => `${k}:${ac[k]}`).join(' ');
  const cards = [
    ['update', progress],
    ['episodes', fmtNum(t.episodes,0)],
    ['samples', fmtNum(t.samples,0)],
    ['states', fmtNum(t.states,0)],
    ['optimal', fmtNum(t.optimal,0)],
    ['farther', fmtNum(t.farther,0), t.farther > 0 ? 'bad' : ''],
    ['blocked', fmtNum(t.blocked,0), t.blocked > 0 ? 'warn' : ''],
    ['invalid', fmtNum(t.invalid,0), t.invalid > 0 ? 'bad' : ''],
    ['mean reward', fmtNum(t.mean_reward,2), tone(t.mean_reward)],
    ['unique actions', fmtNum(t.unique_actions,2)],
    ['group std', fmtNum(t.group_std,3)],
    ['mix opt', ratePct(t.mixed_optimal_rate)],
    ['loss', fmtNum(t.loss,4)],
    ['pg', fmtNum(t.pg,4)],
    ['kl', fmtNum(t.kl,4)],
    ['ratio', ratio],
    ['clip', fmtNum(t.clip_frac,3)],
    ['drop', fmtNum(t.drop_frac,3)],
    ['grad norm', fmtNum(t.grad_norm,3)],
    ['actions', actStr || '—'],
  ];
  el.innerHTML = cards.map(([k,v,c]) =>
    `<div class="kpi ${c||''}"><div class="k">${k}</div><div class="v">${v}</div></div>`
  ).join('');
}

function renderDirectionBars(evalRec){
  const acc = evalRec.test_singleton_accuracy || evalRec.eval_singleton_accuracy || {};
  const ns = evalRec.test_singleton_n || evalRec.eval_singleton_n || {};
  const src = evalRec.test_singleton_accuracy ? 'balanced test' : 'val';
  document.getElementById('direction-meta').textContent =
    `@${src} update ${evalRec.update ?? '—'}`;
  const colors = {up:'#7aa2f7', down:'#fa8a28', left:'#a88ce4', right:'#38be78'};
  const html = ['up','down','left','right'].map(d => {
    const v = (typeof acc[d] === 'number') ? acc[d] : null;
    const n = ns[d] !== undefined ? ns[d] : '—';
    const width = v === null ? 0 : Math.round(v * 1000) / 10;
    const text = v === null ? '—' : ratePct(v);
    return `<div class="dir-row">
      <span class="dir-name">${d}</span>
      <div class="dir-track"><div class="dir-fill" style="width:${width}%;background:${colors[d]}"></div></div>
      <span class="dir-val">${text} <span class="dim">(n=${n})</span></span>
    </div>`;
  }).join('');
  document.getElementById('direction-bars').innerHTML = html;
}

function clearChart(id){
  const c = document.getElementById(id);
  if (c) c.getContext('2d').clearRect(0,0,c.width,c.height);
}
function fmtAxis(v){
  const a = Math.abs(v);
  if (a >= 100) return v.toFixed(0);
  if (a >= 10) return v.toFixed(1);
  if (a >= 1) return v.toFixed(2);
  return v.toFixed(3);
}
function setLegend(id, series){
  const el = document.getElementById(id);
  if (!el) return;
  el.innerHTML = series.map(s =>
    `<span><i style="background:${s.color}"></i>${s.name}</span>`).join('');
}
function drawLineChart(id, series, opts={}){
  const c = document.getElementById(id);
  if (!c) return;
  const g = c.getContext('2d');
  const W = c.width, H = c.height;
  g.clearRect(0,0,W,H);
  const pad = {l:40, r:10, t:10, b:20};
  const pw = W - pad.l - pad.r;
  const ph = H - pad.t - pad.b;
  const vals = series.flatMap(s => s.values)
    .filter(v => typeof v === 'number' && Number.isFinite(v));
  if (!vals.length){
    g.fillStyle = '#8b94a3'; g.font = '11px monospace';
    g.textAlign = 'left'; g.textBaseline = 'top';
    g.fillText('等待数据…', pad.l, pad.t + 4);
    return;
  }
  let min = Math.min(...vals), max = Math.max(...vals);
  if (opts.zeroLine){ min = Math.min(min, 0); max = Math.max(max, 0); }
  if (opts.oneLine){ min = Math.min(min, 1); max = Math.max(max, 1); }
  if (max - min < 1e-9) max = min + 1;
  const span = max - min;
  min -= span * 0.06; max += span * 0.06;

  const n = series[0].values.length;
  const X = i => n <= 1 ? pad.l + pw / 2 : pad.l + (i / (n - 1)) * pw;
  const Y = v => pad.t + (1 - (v - min) / (max - min)) * ph;

  g.font = '10px monospace';
  g.textBaseline = 'middle';
  for (let k=0;k<=4;k++){
    const v = min + (max - min) * k / 4;
    const y = Y(v);
    g.strokeStyle = '#242b36'; g.lineWidth = 1;
    g.beginPath(); g.moveTo(pad.l, y); g.lineTo(W - pad.r, y); g.stroke();
    g.fillStyle = '#8b94a3'; g.textAlign = 'right';
    g.fillText(fmtAxis(v), pad.l - 5, y);
  }
  if (opts.zeroLine && min < 0 && max > 0){
    const y = Y(0);
    g.strokeStyle = '#3a4252'; g.setLineDash([3,3]);
    g.beginPath(); g.moveTo(pad.l, y); g.lineTo(W - pad.r, y); g.stroke();
    g.setLineDash([]);
  }
  if (opts.oneLine){
    const y = Y(1);
    g.strokeStyle = '#3a4252'; g.setLineDash([3,3]);
    g.beginPath(); g.moveTo(pad.l, y); g.lineTo(W - pad.r, y); g.stroke();
    g.setLineDash([]);
  }

  for (const s of series){
    g.strokeStyle = s.color; g.lineWidth = 1.6;
    g.beginPath();
    let started = false;
    for (let i=0;i<s.values.length;i++){
      const v = s.values[i];
      if (typeof v !== 'number' || !Number.isFinite(v)){ started = false; continue; }
      const x = X(i), y = Y(v);
      if (!started){ g.moveTo(x, y); started = true; } else g.lineTo(x, y);
    }
    g.stroke();
    for (let i=s.values.length-1;i>=0;i--){
      const v = s.values[i];
      if (typeof v === 'number' && Number.isFinite(v)){
        g.fillStyle = s.color; g.beginPath();
        g.arc(X(i), Y(v), 2.4, 0, Math.PI * 2); g.fill();
        break;
      }
    }
  }
  if (opts.labels && opts.labels.length === 2){
    g.fillStyle = '#8b94a3'; g.font = '10px monospace';
    g.textBaseline = 'top';
    g.textAlign = 'left'; g.fillText(String(opts.labels[0]), pad.l, H - pad.b + 4);
    g.textAlign = 'right'; g.fillText(String(opts.labels[1]), W - pad.r, H - pad.b + 4);
  }
}

function renderTrainCharts(){
  const hist = trainHistory.filter(r => r.update !== undefined)
    .sort((a,b) => a.update - b.update);
  const ids = ['loss','reward','success','errors','ratio','grad','outcome','eval'];
  if (!hist.length){
    ids.forEach(id => clearChart('chart-' + id));
    ids.forEach(id => setLegend('legend-' + id, []));
    return;
  }
  const take = key => hist.map(r => typeof r[key] === 'number' ? r[key] : null);
  const labels = [hist[0].update, hist[hist.length-1].update];
  const hasSingle = hist.some(r => r.optimal !== undefined || r.eval_optimal_rate !== undefined);
  const charts = [
    ['loss', [
      {name:'loss', color:'#f06464', values:take('loss')},
      {name:'pg', color:'#7aa2f7', values:take('pg')},
    ], {zeroLine:true, labels}],
    ['reward', [
      {name:'mean reward', color:'#38be78', values:take('mean_reward')},
    ], {zeroLine:true, labels}],
    ['success', [
      {name:'success', color:'#38be78', values:take('success')},
      {name:'episodes', color:'#8b94a3', values:take('episodes')},
    ], {labels}],
    ['errors', [
      {name:'collisions', color:'#f06464', values:take('collisions')},
      {name:'format errors', color:'#f5be3c', values:take('format_errors')},
    ], {labels}],
    ['ratio', [
      {name:'ratio min', color:'#7aa2f7', values:take('ratio_min')},
      {name:'ratio max', color:'#a88ce4', values:take('ratio_max')},
    ], {oneLine:true, labels}],
    ['grad', [
      {name:'grad norm', color:'#f5be3c', values:take('grad_norm')},
    ], {labels}],
  ];
  if (hasSingle){
    charts.push(['outcome', [
      {name:'optimal', color:'#38be78', values:take('optimal_rate')},
      {name:'side', color:'#8b94a3', values:take('side_rate')},
      {name:'farther', color:'#f06464', values:take('farther_rate')},
      {name:'blocked', color:'#f5be3c', values:take('blocked_rate')},
      {name:'invalid', color:'#a88ce4', values:take('invalid_rate')},
    ], {zeroLine:true, oneLine:true, labels}]);
    charts.push(['eval', [
      {name:'val optimal', color:'#38be78', values:take('eval_optimal_rate')},
      {name:'balanced test', color:'#7aa2f7', values:take('test_optimal_rate')},
      {name:'natural test', color:'#a88ce4', values:take('natural_optimal_rate')},
    ], {zeroLine:true, oneLine:true, labels}]);
  } else {
    clearChart('chart-outcome');
    clearChart('chart-eval');
    setLegend('legend-outcome', []);
    setLegend('legend-eval', []);
  }
  for (const [id, series, opts] of charts){
    drawLineChart('chart-' + id, series, opts);
    setLegend('legend-' + id, series);
  }
}

function renderTrainTable(){
  const hist = trainHistory.filter(r => r.update !== undefined)
    .sort((a,b) => a.update - b.update).slice(-30).reverse();
  const tbody = document.querySelector('#train-history tbody');
  if (!hist.length){
    tbody.innerHTML = '<tr><td colspan="16" class="dim">等待训练数据…</td></tr>';
    return;
  }
  tbody.innerHTML = hist.map(r => {
    const ratio = (r.ratio_min !== undefined && r.ratio_max !== undefined)
      ? `${fmtNum(r.ratio_min,3)}~${fmtNum(r.ratio_max,3)}` : '—';
    return `<tr>
      <td>${fmtNum(r.update,0)}</td><td>${fmtNum(r.episodes,0)}</td>
      <td>${fmtNum(r.success,0)}</td><td>${fmtNum(r.mean_reward,2)}</td>
      <td>${fmtNum(r.mean_steps,1)}</td><td>${fmtNum(r.collisions,0)}</td>
      <td>${fmtNum(r.format_errors,0)}</td><td>${fmtNum(r.degenerate,0)}</td>
      <td>${fmtNum(r.samples,0)}</td><td>${fmtNum(r.loss,4)}</td>
      <td>${fmtNum(r.pg,4)}</td><td>${fmtNum(r.kl,4)}</td>
      <td>${ratio}</td><td>${fmtNum(r.clip_frac,3)}</td>
      <td>${fmtNum(r.drop_frac,3)}</td><td>${fmtNum(r.grad_norm,3)}</td>
    </tr>`;
  }).join('');
}

/* ------------------------------------------------------------ inference */

function recordTrace(s){
  if (!s || !s.episode) return;
  if (traceKey !== s.episode){ traceKey = s.episode; traceFrames = []; selectedTraceTurn = null; }
  const key = `${s.turn ?? ''}:${s.mode ?? ''}`;
  const i = traceFrames.findIndex(f => `${f.turn ?? ''}:${f.mode ?? ''}` === key);
  if (i >= 0) traceFrames[i] = s; else traceFrames.push(s);
  if (traceFrames.length > 200) traceFrames.shift();
}

function renderEpisodeSummary(){
  const el = document.getElementById('ep-summary');
  const frames = traceFrames.length ? traceFrames : (state ? [state] : []);
  const last = [...frames].reverse().find(f => f.episode === traceKey)
    || frames[frames.length-1] || state;
  if (!last){ el.innerHTML = '<div class="dim">等待轨迹…</div>'; return; }
  const info = last.info || {};
  const rows = [
    ['episode', last.episode ?? '—'],
    ['level', last.level ?? '—'],
    ['seed', last.seed ?? '—'],
    ['turn', last.turn ?? '—'],
    ['shortest', last.meta?.shortest ?? '—'],
    ['detour', last.meta?.detour ?? '—'],
    ['success', info.success === undefined ? '—' : (info.success ? '✓' : '✗')],
    ['total reward', info.total_reward === undefined ? '—' : fmtNum(info.total_reward,2)],
    ['steps', info.steps ?? '—'],
    ['reason', info.reason ?? '—'],
  ];
  el.innerHTML = rows.map(([k,v]) =>
    `<div><span class="dim">${k}</span><span>${v}</span></div>`).join('');
}

function renderTraceTable(){
  const tbody = document.querySelector('#trace tbody');
  const byTurn = new Map();
  for (const f of traceFrames){
    if (f.mode === 'done') continue;
    if (f.turn === null || f.turn === undefined) continue;
    if (!(f.llm || (f.info && f.info.action))) continue;
    const prev = byTurn.get(f.turn);
    if (!prev || (f.llm && !prev.llm)) byTurn.set(f.turn, f);
  }
  const rows = [...byTurn.values()].sort((a,b) => a.turn - b.turn);
  document.getElementById('trace-count').textContent = rows.length ? `${rows.length} 步` : '';
  if (!rows.length){
    tbody.innerHTML = '<tr><td colspan="9" class="dim">等待推理轨迹…</td></tr>';
    return;
  }
  tbody.innerHTML = rows.map(f => {
    const llm = f.llm || {}, info = f.info || {}, meta = f.meta || {};
    const act = llm.action || info.action || '—';
    const rew = info.reward;
    const pos = f.pos ? `(${f.pos[0]},${f.pos[1]})` : '—';
    const dist = meta.distance_to_goal ?? '—';
    const reason = info.reason ?? '—';
    const tokens = llm.tokens ?? '—';
    const logp = (llm.logprob !== undefined && llm.logprob !== null)
      ? Number(llm.logprob).toFixed(3) : '—';
    const lat = llm.latency !== undefined ? `${Math.round(llm.latency*1000)}ms` : '—';
    const sel = selectedTraceTurn === f.turn ? ' class="sel"' : '';
    return `<tr data-turn="${f.turn}"${sel}>
      <td>${f.turn}</td><td>${esc(act)}</td>
      <td>${rew === undefined ? '—' : fmtNum(rew,2)}</td>
      <td>${pos}</td><td>${dist}</td><td>${esc(reason)}</td>
      <td>${tokens}</td><td>${logp}</td><td>${lat}</td>
    </tr>`;
  }).join('');
}

function renderInference(){
  renderEpisodeSummary();
  renderStats();
  renderLLM(state || {});
  renderTraceTable();
}

/* ------------------------------------------------------------ state */

function applyState(s, opts={}){
  if (!s) return;
  const prev = state;
  state = s;
  frames++;
  if (!prev || prev.mode !== s.mode || prev.level !== s.level || prev.seed !== s.seed){
    if (s.level) document.getElementById('level').value = s.level;
    if (s.seed !== null && s.seed !== undefined) document.getElementById('seed').value = s.seed;
  }
  if (s.pos){ target = {x:s.pos[0], y:s.pos[1]}; }
  if (!prev || s.mode === 'reset' || !Number.isFinite(shown.x) || !Number.isFinite(shown.y)){
    if (target) shown = {...target};
  }
  if (opts.record !== false) recordTrace(s);
  if (!prev || (s.episode && prev.episode !== s.episode)){
    llmChipsEl.innerHTML = '';
    outEl.innerHTML = '<span class="dim">等待模型输出…</span>';
    promptEl.textContent = '';
  }
  renderStats();
  renderLLM(s);
  renderEpisodeSummary();
  renderTraceTable();
  setMode(s.mode);

  const info = s.info || {};
  const bits = [];
  if (info.action) bits.push(`<b>${info.action}</b>`);
  if (info.reward !== undefined){
    const c = info.reward > 0 ? 'good' : 'bad';
    bits.push(`reward <span class="${c}">${info.reward.toFixed(3)}</span>`);
  }
  if (info.reason && info.reason !== 'move') bits.push(`<span class="warn">${info.reason}</span>`);
  if (info.step !== undefined) bits.push(`step ${info.step}`);
  if (typeof info.reward === 'number' && opts.record !== false){
    rewards.push(info.reward);
    if (rewards.length > 120) rewards.shift();
    drawSpark();
  }
  if (bits.length && opts.log !== false) log(bits.join(' · '));
}

function renderStats(){
  if (!state || !state.meta){ statsEl.innerHTML = ''; return; }
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

/* ------------------------------------------------------------ maze */
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

/* ------------------------------------------------------------ API / controls */
async function api(path, body){
  const opt = body ? {method:'POST', headers:{'Content-Type':'application/json'},
                      body:JSON.stringify(body)} : {};
  const r = await fetch(path, opt);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

document.querySelectorAll('.tab').forEach(btn => {
  btn.onclick = () => {
    document.querySelectorAll('.tab').forEach(b => b.classList.toggle('active', b === btn));
    document.querySelectorAll('.pane').forEach(p => p.classList.toggle('active', p.id === btn.dataset.pane));
    if (btn.dataset.pane === 'pane-train') renderTraining();
    else renderInference();
  };
});

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

document.querySelector('#trace tbody').addEventListener('click', ev => {
  const tr = ev.target.closest('tr[data-turn]');
  if (!tr) return;
  const turn = Number(tr.dataset.turn);
  const f = traceFrames.find(x => x.turn === turn && (x.llm || (x.info && x.info.action)));
  if (!f) return;
  selectedTraceTurn = turn;
  pinned = true;
  applyState(f, {record:false, log:false});
  setStatus(true, `已暂停 · ${f.episode || ''}/t${turn}`);
  renderTraceTable();
});

function renderEpsOptions(){
  const sel = document.getElementById('eps');
  const cur = sel.value;
  const filter = (document.getElementById('eps-filter').value || '').trim().toLowerCase();
  const list = episodeList.filter(e =>
    !filter || String(e.id).toLowerCase().includes(filter));
  sel.innerHTML = '<option value="">（跟随最新）</option>' + list.slice(0, 1000).map(e => {
    const mark = e.success === true ? 'OK' : e.success === false ? 'X' : '-';
    return `<option value="${e.id}">${mark} ${e.id} · ${e.level || ''} · t${e.turn ?? '?'}</option>`;
  }).join('');
  if (cur && list.some(e => e.id === cur)) sel.value = cur;
}

async function refreshEps(){
  try {
    const r = await api('/api/episodes');
    const list = r.episodes || [];
    const sig = list.map(e => `${e.id}:${e.turn}:${e.success}:${e.turns}`).join('|');
    if (sig === epsSig) return;
    epsSig = sig;
    episodeList = list;
    renderEpsOptions();
  } catch (err) { /* UI 服务可能刚重启 */ }
}

async function loadTrace(id){
  try {
    const r = await api('/api/trace?id=' + encodeURIComponent(id));
    traceKey = id;
    traceFrames = r.frames || [];
    selectedTraceTurn = null;
    const last = [...traceFrames].reverse().find(f => f.llm)
      || traceFrames[traceFrames.length-1];
    if (last) applyState(last, {record:false, log:false});
    else { renderEpisodeSummary(); renderTraceTable(); }
    pinned = true;
    setStatus(true, '已暂停 · ' + id);
    renderInference();
    log(`查看历史轨迹 ${id}`, 'warn');
  } catch (err) { log(String(err), 'bad'); }
}

document.getElementById('eps').onchange = async (ev) => {
  const id = ev.target.value;
  if (!id){ followLatest(); return; }
  await loadTrace(id);
};
document.getElementById('follow').onclick = () => followLatest();
document.getElementById('eps-filter').oninput = renderEpsOptions;
function followLatest(){
  pinned = false;
  traceKey = '';
  traceFrames = [];
  selectedTraceTurn = null;
  document.getElementById('eps').value = '';
  setStatus(true, '已连接');
  api('/api/state').then(s => { if (s && s.width) applyState(s); }).catch(() => {});
  renderInference();
}

let es = null;
function connect(){
  if (es) es.close();
  es = new EventSource('/api/stream');
  es.onopen = () => { setStatus(true, '已连接'); log('SSE 已连接', 'good'); };
  es.onerror = () => { setStatus(false, '重连中…'); };
  es.onmessage = (e) => {
    try {
      const s = JSON.parse(e.data);
      if (s.mode === 'train'){
        upsertTrain(s.train, s.ts);
        setMode('train');
        return;
      }
      if (s.train) upsertTrain(s.train, s.ts);   // SSE 首帧快照
      if (pinned) return;
      applyState(s);
    } catch(err){ log('parse error: ' + err, 'bad'); }
  };
}

(async function init(){
  const sel = document.getElementById('level');
  sel.innerHTML = ['L0','L1','L2','L3','L4']
    .map(v => `<option value="${v}">${v}</option>`).join('');
  try {
    const tr = await api('/api/train');
    trainHistory = tr.history || [];
    trainHistory.sort((a,b) => (a.update ?? 0) - (b.update ?? 0));
    latestTrain = trainHistory.length ? trainHistory[trainHistory.length-1] : tr.latest;
    renderTraining();
  } catch(e){}
  try {
    const s = await api('/api/state');
    if (!s.width) await api('/api/maze', {level:'L3', seed:null});
    else applyState(s);
  } catch(e){ log(String(e), 'bad'); }
  connect();
  refreshEps();
  setInterval(refreshEps, 2000);
  requestAnimationFrame(tick);
})();
</script>
</body>
</html>
"""
