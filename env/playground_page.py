"""Standalone browser page for the custom-map Playground.

The page is intentionally dependency-free: canvas editing plus fetch/SSE,
served by the existing UIServer.  It does not load the model itself; the
server-side PlaygroundRunner owns inference and publishes frames.
"""

PAGE = r"""<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Maze Playground</title>
<style>
:root{
  --bg:#0f1218; --panel:#161b23; --panel2:#11161e; --line:#242b36;
  --text:#e6e9f0; --dim:#8b94a3; --wall:#3a4252; --free:#f8f9fc;
  --start:#5694f0; --goal:#38be78; --agent:#fa8a28; --path:#a88ce4;
  --bad:#f06464; --warn:#f5be3c; --accent:#7aa2f7; --good:#38be78;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);
     font:13px/1.5 ui-monospace,"Cascadia Code",Consolas,monospace}
header{display:flex;align-items:center;gap:10px;padding:12px 18px;
       border-bottom:1px solid var(--line);flex-wrap:wrap}
header h1{font-size:15px;font-weight:600;margin:0;letter-spacing:.02em}
header a{color:var(--accent);text-decoration:none;font-size:12px}
.dot{width:8px;height:8px;border-radius:50%;background:var(--bad)}
.dot.on{background:var(--good);box-shadow:0 0 8px var(--good)}
.dim{color:var(--dim)} .good{color:var(--good)} .bad{color:var(--bad)}
.warn{color:var(--warn)} .spacer{flex:1}
.badge{font-size:11px;padding:2px 8px;border-radius:999px;
       border:1px solid var(--line);color:var(--dim);white-space:nowrap}
main{display:flex;gap:18px;padding:16px 18px;align-items:flex-start;
     flex-wrap:wrap}
.board{background:var(--panel);padding:12px;border-radius:12px;
       border:1px solid var(--line);flex:0 0 auto}
canvas{display:block;border-radius:8px;background:var(--free);
       max-width:100%;height:auto;touch-action:none;cursor:crosshair}
.bar{display:flex;gap:7px;margin-top:10px;flex-wrap:wrap;align-items:center}
button,select,input{background:#1d2430;color:var(--text);
  border:1px solid var(--line);border-radius:8px;padding:6px 10px;
  font:inherit;outline:none}
button{cursor:pointer;transition:.15s}
button:hover{border-color:#3a4a63;background:#232c3a}
button:active{transform:translateY(1px)}
button.primary{background:#2a4a7a;border-color:#3a63a0}
button.stop{background:#4a2530;border-color:#7a3a4a}
button.tool.active{background:#26344d;border-color:#3a63a0;color:#fff}
input[type=number]{width:82px}
label{font-size:11px;color:var(--dim);white-space:nowrap}
aside{flex:1;min-width:420px;max-width:760px;max-height:calc(100vh - 90px);
      overflow-y:auto;padding-right:2px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;
      padding:12px;margin-bottom:12px}
.card h2{font-size:11px;font-weight:600;letter-spacing:.08em;
         text-transform:uppercase;color:var(--dim);margin:0 0 10px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:7px 10px}
.field{display:flex;align-items:center;justify-content:space-between;gap:8px}
.field span{font-size:11px;color:var(--dim);white-space:nowrap}
.field input{width:100%;min-width:0}
.field input[type=number]{width:86px}
.full{grid-column:1/-1}
.full input{width:100%}
.hint{font-size:11px;color:var(--dim);margin-top:8px;line-height:1.6}
#mapmsg{min-height:17px;font-size:11px;color:var(--dim);margin-top:7px}
#mapmsg.bad{color:var(--bad)}
.kpi-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));
          gap:7px}
.kpi{background:var(--panel2);border:1px solid var(--line);
     border-radius:8px;padding:7px 9px;min-width:0}
.kpi .k{font-size:10px;color:var(--dim);white-space:nowrap;
        overflow:hidden;text-overflow:ellipsis}
.kpi .v{font-size:15px;font-weight:600;white-space:nowrap;
        overflow:hidden;text-overflow:ellipsis}
.kpi.good .v{color:var(--good)} .kpi.bad .v{color:var(--bad)}
.kpi.warn .v{color:var(--warn)}
.table-wrap{max-height:260px;overflow:auto;border:1px solid var(--line);
            border-radius:8px;background:var(--panel2)}
table{width:100%;border-collapse:collapse;font-size:11px;white-space:nowrap}
th,td{padding:5px 7px;border-bottom:1px solid var(--line);text-align:right}
th{position:sticky;top:0;background:#141a23;color:var(--dim);
   font-weight:500;z-index:1}
th:first-child,td:first-child{text-align:left}
tbody tr:hover{background:#1a2230}
#out{height:120px;overflow:auto;margin:0;padding:8px;
     background:var(--panel2);border:1px solid var(--line);border-radius:8px;
     font-size:11px;line-height:1.55;color:#cfd6e2;white-space:pre-wrap;
     word-break:break-all}
#prompt{max-height:180px;overflow:auto;margin:6px 0 0;padding:8px;
        background:var(--panel2);border:1px solid var(--line);
        border-radius:8px;font-size:10px;line-height:1.5;color:var(--dim);
        white-space:pre}
details{margin-top:8px} summary{cursor:pointer;font-size:11px;color:var(--dim)}
.row{display:flex;gap:7px;align-items:center;flex-wrap:wrap}
@media(max-width:900px){aside{max-height:none;max-width:100%}
  .grid2{grid-template-columns:1fr}.kpi-grid{grid-template-columns:repeat(2,minmax(0,1fr))}}
</style>
</head>
<body>
<header>
  <h1>Maze Playground</h1>
  <span class="dot" id="dot"></span>
  <span class="dim" id="status">连接中…</span>
  <span class="spacer"></span>
  <span class="badge" id="modelBadge">model: idle</span>
  <a href="/">训练看板 →</a>
</header>

<main>
<section class="board">
  <canvas id="cv" width="620" height="620"></canvas>
  <div class="bar">
    <label>宽 <input id="width" type="number" value="12" min="4" max="40"></label>
    <label>高 <input id="height" type="number" value="12" min="4" max="40"></label>
    <label>max steps <input id="maxSteps" type="number" value="30" min="1" max="300"></label>
  </div>
  <div class="bar">
    <button class="tool active" data-tool="wall">画墙</button>
    <button class="tool" data-tool="erase">擦墙</button>
    <button class="tool" data-tool="start">起点</button>
    <button class="tool" data-tool="goal">终点</button>
    <button id="clear">清空墙</button>
    <button id="random">随机墙</button>
  </div>
  <div class="bar">
    <button id="export">导出 JSON</button>
    <button id="importBtn">导入 JSON</button>
    <input id="importFile" type="file" accept="application/json" style="display:none">
  </div>
  <div id="mapmsg"></div>
  <div class="hint">
    左键画；右键快速擦除；起点/终点工具会移除该格墙。
    运行模型时地图会锁定。
  </div>
</section>

<aside>
  <section class="card">
    <h2>模型配置</h2>
    <div class="grid2">
      <div class="field full"><span>model path</span><input id="model" placeholder="Qwen3 model path"></div>
      <div class="field full"><span>adapter path</span><input id="adapter" placeholder="empty = base model"></div>
      <div class="field"><span>temperature</span><input id="temperature" type="number" value="0" min="0" max="5" step="0.1"></div>
      <div class="field"><span>top_p</span><input id="topP" type="number" value="1" min="0.01" max="1" step="0.05"></div>
      <div class="field"><span>max tokens</span><input id="maxTokens" type="number" value="48" min="1" max="512"></div>
      <div class="field"><span>history</span><input id="historyWindow" type="number" value="12" min="1" max="64"></div>
      <div class="field"><span>max seq len</span><input id="maxSeqLen" type="number" value="4096" min="64" max="16384"></div>
      <div class="field"><span>seed</span><input id="seed" type="number" value="7" min="0"></div>
    </div>
    <div class="bar" style="margin-top:10px">
      <button id="load">加载模型</button>
      <button class="primary" id="run">运行</button>
      <button class="stop" id="stop">停止</button>
      <button id="release">释放模型</button>
    </div>
    <div id="runmsg" class="hint"></div>
  </section>

  <section class="card">
    <h2>本次轨迹</h2>
    <div class="kpi-grid">
      <div class="kpi"><div class="k">success</div><div class="v" id="kSuccess">—</div></div>
      <div class="kpi"><div class="k">steps</div><div class="v" id="kSteps">—</div></div>
      <div class="kpi"><div class="k">collisions</div><div class="v" id="kColl">—</div></div>
      <div class="kpi"><div class="k">format errors</div><div class="v" id="kFmt">—</div></div>
      <div class="kpi"><div class="k">shortest</div><div class="v" id="kShortest">—</div></div>
      <div class="kpi"><div class="k">reward</div><div class="v" id="kReward">—</div></div>
      <div class="kpi"><div class="k">turn</div><div class="v" id="kTurn">—</div></div>
      <div class="kpi"><div class="k">status</div><div class="v" id="kStatus">—</div></div>
    </div>
  </section>

  <section class="card">
    <h2>逐步轨迹</h2>
    <div class="table-wrap">
      <table>
        <thead><tr><th>#</th><th>action</th><th>pos</th><th>blocked</th>
          <th>reward</th><th>goal Δ</th></tr></thead>
        <tbody id="trace"></tbody>
      </table>
    </div>
  </section>

  <section class="card">
    <h2>模型输出</h2>
    <pre id="out">还没有运行。</pre>
    <details><summary>查看本轮 prompt</summary><pre id="prompt"></pre></details>
  </section>
</aside>
</main>

<script>
const $ = id => document.getElementById(id);
const cv = $('cv');
const ctx = cv.getContext('2d');
const S = 620;
const PAD = 20;
const COLORS = {
  wall: '#3a4252', free: '#f8f9fc', grid: '#e6e9f0',
  start: '#5694f0', goal: '#38be78', agent: '#fa8a28', path: '#a88ce4'
};
let state = {
  width: 12, height: 12, walls: new Set(), start: [0, 0],
  goal: [11, 11], maxSteps: 30, tool: 'wall'
};
let frame = null;
let running = false;
let drawing = false;
let lastCell = null;
let trace = [];
let es = null;

function key(x, y){ return x + ',' + y; }
function hasWall(x, y){ return state.walls.has(key(x, y)); }
function setMsg(t, bad=false){ const el=$('mapmsg'); el.textContent=t; el.classList.toggle('bad', !!bad); }
function setRunMsg(t, bad=false){ const el=$('runmsg'); el.textContent=t; el.classList.toggle('bad', !!bad); }
function setStatus(t, ok=null){
  $('status').textContent = t;
  const dot=$('dot');
  if (ok === null) return;
  dot.classList.toggle('on', !!ok);
}

function bfs(){
  const w=state.width, h=state.height;
  const dist=Array.from({length:h},()=>Array(w).fill(Infinity));
  const [sx,sy]=state.start, [gx,gy]=state.goal;
  if (sx<0||sy<0||sx>=w||sy>=h||gx<0||gy<0||gx>=w||gy>=h) return {ok:false, shortest:null};
  if (hasWall(...state.start)||hasWall(...state.goal)) return {ok:false, shortest:null};
  dist[sy][sx]=0;
  const q=[[sx,sy]];
  const dirs=[[1,0],[-1,0],[0,1],[0,-1]];
  for(let qi=0; qi<q.length; qi++){
    const [x,y]=q[qi], d=dist[y][x];
    for(const [dx,dy] of dirs){
      const nx=x+dx, ny=y+dy;
      if(nx<0||ny<0||nx>=w||ny>=h||hasWall(nx,ny)||dist[ny][nx]!==Infinity) continue;
      dist[ny][nx]=d+1; q.push([nx,ny]);
    }
  }
  const shortest=dist[gy][gx];
  return {ok:Number.isFinite(shortest), shortest:Number.isFinite(shortest)?shortest:null};
}
function refreshMeta(){
  const m=bfs();
  const walls=state.walls.size;
  if(!m.ok){
    setMsg(`当前地图不可达 · walls=${walls}`, true);
    $('run').disabled=true;
    return;
  }
  if(state.maxSteps < m.shortest){
    setMsg(`最短路 ${m.shortest} > max steps ${state.maxSteps}`, true);
    $('run').disabled=true;
    return;
  }
  setMsg(`最短路 ${m.shortest} · walls ${walls} · 可运行`, false);
  $('run').disabled=running;
}
function cellSize(){
  return Math.max(4, Math.min((S-2*PAD)/state.width, (S-2*PAD)/state.height));
}
function cellRect(x,y){
  const c=cellSize();
  const ox=(S-state.width*c)/2, oy=(S-state.height*c)/2;
  return {x:ox+x*c, y:oy+y*c, c};
}
function cellFromEvent(e){
  const r=cv.getBoundingClientRect();
  const px=(e.clientX-r.left)*S/r.width;
  const py=(e.clientY-r.top)*S/r.height;
  const c=cellSize();
  const ox=(S-state.width*c)/2, oy=(S-state.height*c)/2;
  if(px<ox||py<oy) return null;
  const x=Math.floor((px-ox)/c), y=Math.floor((py-oy)/c);
  if(x<0||y<0||x>=state.width||y>=state.height) return null;
  return [x,y];
}
function draw(){
  ctx.clearRect(0,0,S,S);
  ctx.fillStyle=COLORS.free; ctx.fillRect(0,0,S,S);
  const c=cellSize();
  const ox=(S-state.width*c)/2, oy=(S-state.height*c)/2;
  ctx.strokeStyle=COLORS.grid; ctx.lineWidth=1;
  for(let x=0;x<=state.width;x++){
    ctx.beginPath(); ctx.moveTo(ox+x*c,oy); ctx.lineTo(ox+x*c,oy+state.height*c); ctx.stroke();
  }
  for(let y=0;y<=state.height;y++){
    ctx.beginPath(); ctx.moveTo(ox,oy+y*c); ctx.lineTo(ox+state.width*c,oy+y*c); ctx.stroke();
  }
  for(const k of state.walls){
    const [x,y]=k.split(',').map(Number);
    ctx.fillStyle=COLORS.wall; ctx.fillRect(ox+x*c,oy+y*c,c,c);
  }
  if(frame && frame.path && frame.path.length>1){
    ctx.strokeStyle=COLORS.path; ctx.lineWidth=Math.max(2,c*.14);
    ctx.globalAlpha=.65; ctx.beginPath();
    frame.path.forEach((p,i)=>{
      const q=cellRect(p[0],p[1]);
      if(i===0) ctx.moveTo(q.x+q.c/2,q.y+q.c/2);
      else ctx.lineTo(q.x+q.c/2,q.y+q.c/2);
    });
    ctx.stroke(); ctx.globalAlpha=1;
  }
  const mark=(p,color,label)=>{
    const q=cellRect(p[0],p[1]);
    ctx.fillStyle=color; ctx.fillRect(q.x+q.c*.12,q.y+q.c*.12,q.c*.76,q.c*.76);
    ctx.fillStyle='#fff'; ctx.font=`600 ${Math.max(10,c*.32)}px ui-monospace,monospace`;
    ctx.textAlign='center'; ctx.textBaseline='middle';
    ctx.fillText(label,q.x+q.c/2,q.y+q.c/2);
  };
  mark(state.start,COLORS.start,'S');
  mark(state.goal,COLORS.goal,'G');
  const pos=frame && frame.pos ? frame.pos : state.start;
  const q=cellRect(pos[0],pos[1]);
  ctx.fillStyle=COLORS.agent; ctx.beginPath();
  ctx.arc(q.x+q.c/2,q.y+q.c/2,Math.max(4,c*.28),0,Math.PI*2); ctx.fill();
  ctx.strokeStyle='#fff'; ctx.lineWidth=2; ctx.stroke();
}

function ensureDistinct(){
  if(state.start[0]!==state.goal[0]||state.start[1]!==state.goal[1]) return;
  const candidates=[[state.width-1,state.height-1],[0,state.height-1],
    [state.width-1,0],[0,0],[1,0],[0,1]];
  for(const p of candidates){
    if(p[0]===state.start[0]&&p[1]===state.start[1]) continue;
    if(p[0]>=0&&p[1]>=0&&p[0]<state.width&&p[1]<state.height){
      state.goal=p; state.walls.delete(key(...p)); return;
    }
  }
}
function clearFrame(){
  frame=null; trace=[]; renderTrace();
  $('kSuccess').textContent='—'; $('kSteps').textContent='—';
  $('kColl').textContent='—'; $('kFmt').textContent='—';
  $('kTurn').textContent='—'; $('kStatus').textContent='idle';
}
function resizeFromInputs(){
  if(running) return;
  const w=Math.max(4,Math.min(40,parseInt($('width').value)||12));
  const h=Math.max(4,Math.min(40,parseInt($('height').value)||12));
  const old=state;
  state.width=w; state.height=h;
  state.maxSteps=Math.max(1,Math.min(300,parseInt($('maxSteps').value)||30));
  state.walls=new Set([...old.walls].filter(k=>{
    const [x,y]=k.split(',').map(Number); return x<w&&y<h;
  }));
  state.start=[Math.min(old.start[0],w-1),Math.min(old.start[1],h-1)];
  state.goal=[Math.min(old.goal[0],w-1),Math.min(old.goal[1],h-1)];
  state.walls.delete(key(...state.start)); state.walls.delete(key(...state.goal));
  ensureDistinct();
  clearFrame();
  $('width').value=w; $('height').value=h; $('maxSteps').value=state.maxSteps;
  refreshMeta(); draw();
}
function setTool(t){
  state.tool=t;
  document.querySelectorAll('.tool').forEach(b=>b.classList.toggle('active',b.dataset.tool===t));
  cv.style.cursor = t==='wall'||t==='erase' ? 'crosshair' : 'pointer';
}
function editCell(cell, forceErase=false){
  if(!cell || running) return;
  const [x,y]=cell;
  const k=key(x,y);
  if(forceErase || state.tool==='erase'){
    state.walls.delete(k);
  }else if(state.tool==='wall'){
    if((x===state.start[0]&&y===state.start[1])||(x===state.goal[0]&&y===state.goal[1])) return;
    state.walls.add(k);
  }else if(state.tool==='start'){
    state.walls.delete(k); state.start=[x,y]; ensureDistinct();
  }else if(state.tool==='goal'){
    state.walls.delete(k); state.goal=[x,y]; ensureDistinct();
  }
  clearFrame();
  refreshMeta(); draw();
}
cv.addEventListener('mousedown',e=>{
  if(running) return;
  e.preventDefault();
  drawing=true; lastCell=null;
  const cell=cellFromEvent(e);
  lastCell=cell; editCell(cell,e.button===2);
});
cv.addEventListener('mousemove',e=>{
  if(!drawing||running) return;
  const cell=cellFromEvent(e);
  if(!cell) return;
  const k=key(...cell);
  if(k===lastCell) return;
  lastCell=cell; editCell(cell,false);
});
window.addEventListener('mouseup',()=>{drawing=false; lastCell=null;});
cv.addEventListener('contextmenu',e=>e.preventDefault());

$('width').onchange=resizeFromInputs;
$('height').onchange=resizeFromInputs;
$('maxSteps').onchange=resizeFromInputs;
document.querySelectorAll('.tool').forEach(b=>b.onclick=()=>setTool(b.dataset.tool));
$('clear').onclick=()=>{ if(running) return; state.walls.clear(); clearFrame(); refreshMeta(); draw(); };
$('random').onclick=()=>{
  if(running) return;
  state.walls.clear();
  for(let y=0;y<state.height;y++) for(let x=0;x<state.width;x++){
    if((x===state.start[0]&&y===state.start[1])||(x===state.goal[0]&&y===state.goal[1])) continue;
    if(Math.random()<0.18) state.walls.add(key(x,y));
  }
  clearFrame(); refreshMeta(); draw();
};
$('export').onclick=()=>{
  const data={width:state.width,height:state.height,
    walls:[...state.walls].map(k=>k.split(',').map(Number)),
    start:state.start,goal:state.goal,max_steps:state.maxSteps};
  const a=document.createElement('a');
  a.href=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));
  a.download='maze.json'; a.click(); URL.revokeObjectURL(a.href);
};
$('importBtn').onclick=()=>$('importFile').click();
$('importFile').onchange=async e=>{
  const f=e.target.files[0]; if(!f) return;
  try{
    const d=JSON.parse(await f.text());
    state.width=Math.max(4,Math.min(40,parseInt(d.width)||12));
    state.height=Math.max(4,Math.min(40,parseInt(d.height)||12));
    state.walls=new Set((d.walls||[]).map(p=>key(parseInt(p[0]),parseInt(p[1])))
      .filter(k=>{const [x,y]=k.split(',').map(Number);return x<state.width&&y<state.height;}));
    state.start=[parseInt(d.start?.[0]??0),parseInt(d.start?.[1]??0)];
    state.goal=[parseInt(d.goal?.[0]??state.width-1),parseInt(d.goal?.[1]??state.height-1)];
    state.maxSteps=Math.max(1,Math.min(300,parseInt(d.max_steps??30)));
    $('width').value=state.width; $('height').value=state.height;
    $('maxSteps').value=state.maxSteps;
    state.walls.delete(key(...state.start)); state.walls.delete(key(...state.goal));
    ensureDistinct(); clearFrame();
    refreshMeta(); draw();
  }catch(err){setMsg('导入失败: '+err,true);}
  e.target.value='';
};

function cfg(){
  return {
    model:$('model').value.trim(), adapter:$('adapter').value.trim(),
    temperature:parseFloat($('temperature').value)||0,
    top_p:parseFloat($('topP').value)||1,
    max_tokens:parseInt($('maxTokens').value)||48,
    history_window:parseInt($('historyWindow').value)||12,
    max_seq_len:parseInt($('maxSeqLen').value)||4096,
    seed:parseInt($('seed').value)||7
  };
}
async function api(path,body){
  const opt=body===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)};
  const r=await fetch(path,opt);
  let data={}; try{data=await r.json();}catch(e){}
  if(!r.ok) throw new Error(data.error||`HTTP ${r.status}`);
  return data;
}
function setModelBadge(s){
  const st=s?.state||'idle';
  $('modelBadge').textContent='model: '+st+(s?.model_loaded?' · loaded':'');
  $('modelBadge').style.color=s?.state==='error'?'var(--bad)':'';
}
async function refreshStatus(){
  try{
    const s=await api('/api/playground/status');
    setModelBadge(s);
    if(s.last_error) setRunMsg(s.last_error,true);
    running=!!s.running;
    if(running){ setStatus('运行中…',true); }
    else if(s.state==='loading') setStatus('加载模型中…',false);
    else if(s.state==='done') setStatus('完成',true);
    else if(s.state==='error') setStatus('错误',false);
    else setStatus('空闲',true);
    $('run').disabled=running;
    $('stop').disabled=!running;
    if(s.last_summary) renderSummary(s.last_summary);
  }catch(e){ setStatus('连接断开',false); }
}
async function loadModel(){
  setRunMsg('加载模型中…');
  try{
    const d=await api('/api/playground/load',{model:$('model').value.trim(),adapter:$('adapter').value.trim()});
    setModelBadge(d.status);
    setRunMsg('模型已加载');
  }catch(e){setRunMsg('加载失败: '+e.message,true);}
}
async function runModel(){
  setRunMsg('校验地图…');
  try{
    await api('/api/playground/maze',{
      width:state.width,height:state.height,
      walls:[...state.walls].map(k=>k.split(',').map(Number)),
      start:state.start,goal:state.goal,max_steps:state.maxSteps
    });
    clearFrame();
    setRunMsg('启动模型…');
    const d=await api('/api/playground/run',cfg());
    setModelBadge(d.status); running=true; $('run').disabled=true;
    setStatus('运行中…',true);
  }catch(e){setRunMsg('运行失败: '+e.message,true); setStatus('空闲',false);}
}
async function stopModel(){
  try{ await api('/api/playground/stop',{}); setRunMsg('已发送停止请求'); }
  catch(e){setRunMsg('停止失败: '+e.message,true);}
}
async function releaseModel(){
  if(running) return;
  try{ const d=await api('/api/playground/release',{}); setModelBadge(d.status); setRunMsg('模型已释放'); }
  catch(e){setRunMsg('释放失败: '+e.message,true);}
}
function renderTrace(){
  $('trace').innerHTML=trace.map(r=>`<tr>
    <td>${r.turn??''}</td><td>${esc(r.action??'—')}</td>
    <td>${r.pos??''}</td><td>${r.blocked??'—'}</td>
    <td>${fmt(r.reward)}</td><td>${r.delta??'—'}</td></tr>`).join('');
}
function renderSummary(s){
  if(!s) return;
  $('kSuccess').textContent=s.success?'yes':'no';
  $('kSuccess').className='v '+(s.success?'good':'bad');
  $('kSteps').textContent=s.steps??'—';
  $('kColl').textContent=s.collisions??'—';
  $('kFmt').textContent=s.format_errors??'—';
  $('kShortest').textContent=s.shortest??'—';
  $('kReward').textContent=fmt(s.total_reward);
  $('kTurn').textContent=s.steps??'—';
  $('kStatus').textContent=s.stopped?'stopped':(s.success?'success':'done');
}
function fmt(v){ return typeof v==='number'?v.toFixed(2):(v??'—'); }
function esc(s){ return String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
function applyFrame(s){
  if(s.mode==='playground_error'){
    running=false; setStatus('错误',false); setRunMsg(s.error||'模型运行出错',true); $('run').disabled=false; return;
  }
  if(s.mode!=='playground'&&s.mode!=='playground_reset'&&s.mode!=='playground_done') return;
  frame=s;
  if(s.width&&s.height){ state.width=s.width; state.height=s.height; }
  if(Array.isArray(s.walls)) state.walls=new Set(s.walls.map(p=>key(p[0],p[1])));
  if(Array.isArray(s.start)) state.start=s.start;
  if(Array.isArray(s.goal)) state.goal=s.goal;
  if(s.meta?.max_steps){ state.maxSteps=s.meta.max_steps; $('maxSteps').value=s.meta.max_steps; }
  $('kShortest').textContent=s.meta?.shortest??'—';
  $('kTurn').textContent=s.turn??'—';
  const info=s.info||{};
  if(s.mode==='playground'){
    const tr=s.turn;
    if(tr && !trace.some(x=>x.turn===tr)){
      trace.push({turn:tr,action:info.action,pos:(s.pos||[]).join(','),
        blocked:info.tool_result?.blocked_by??(info.tool_result?.blocked?'yes':'no'),
        reward:info.reward,delta:(info.tool_result?.goal_delta||[]).join(',')});
      renderTrace();
    }
    $('kColl').textContent=trace.reduce((n,x)=>n+(x.blocked&&x.blocked!=='no'?1:0),0);
    $('kStatus').textContent='running';
  }
  if(s.mode==='playground_done'){
    running=false; $('run').disabled=false; $('stop').disabled=true;
    if(info.status) $('kStatus').textContent=info.status;
    if(info.success!==undefined) renderSummary(info);
    setStatus(info.success?'成功':'结束', !!info.success);
  }
  if(s.llm){
    $('out').textContent=(s.llm.text||'')+'\n\n'+JSON.stringify(s.llm,null,2);
  }
  if(s.prompt) $('prompt').textContent=s.prompt;
  refreshMeta(); draw();
}
function connect(){
  if(es) es.close();
  es=new EventSource('/api/stream');
  es.onopen=()=>{setStatus('已连接',true);};
  es.onerror=()=>{setStatus('SSE 重连中…',false);};
  es.onmessage=e=>{
    try{
      const s=JSON.parse(e.data);
      if(s.mode&&String(s.mode).startsWith('playground')) applyFrame(s);
    }catch(err){}
  };
}
(async function init(){
  setTool('wall');
  try{
    const c=await api('/api/playground/config');
    $('model').value=c.model||'';
    $('adapter').value=c.adapter||'';
  }catch(e){}
  $('run').disabled=false;
  $('stop').disabled=true;
  $('load').onclick=loadModel;
  $('run').onclick=runModel;
  $('stop').onclick=stopModel;
  $('release').onclick=releaseModel;
  refreshMeta(); draw(); connect(); refreshStatus();
  setInterval(refreshStatus,1500);
})();
</script>
</body>
</html>
"""
