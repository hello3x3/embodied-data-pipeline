"""Static review page: task → subtask → phase, with continuous frame playback.

The phase is a **variable-length dense trajectory** (6.25 Hz). There is no
chunking in the data; the train-time loader slides its own 7-point window.
The JSON is inlined, so the page also works from ``file://``.
"""
from __future__ import annotations

import json
from pathlib import Path

__all__ = ["write_index"]

_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>RoboDojo · 任务 / 子任务 / 阶段</title>
<style>
  :root { --bg:#0b0f14; --panel:#131a22; --line:#22303d; --fg:#d7e0ea; --dim:#8aa0b4;
          --accent:#e53e3e; --ok:#38a169; --cur:#f6c90e; }
  * { box-sizing:border-box; }
  body { margin:0; font:13px/1.5 -apple-system,"Segoe UI",Roboto,sans-serif;
         background:var(--bg); color:var(--fg); }
  header { padding:10px 16px; border-bottom:1px solid var(--line); display:flex; gap:16px;
           align-items:baseline; flex-wrap:wrap; }
  header h1 { font-size:15px; margin:0; font-weight:600; }
  header .meta { color:var(--dim); font-size:12px; }
  .layout { display:grid; grid-template-columns:300px 1fr; height:calc(100vh - 49px); }
  aside { border-right:1px solid var(--line); overflow:auto; padding:10px; background:var(--panel); }
  main { overflow:auto; padding:12px 16px; }
  h2 { font-size:11px; text-transform:uppercase; letter-spacing:.08em; color:var(--dim);
       margin:14px 4px 6px; }
  button { background:#1b2530; color:var(--fg); border:1px solid var(--line); border-radius:5px;
           padding:4px 9px; cursor:pointer; font-size:12px; }
  button:hover { border-color:#3b5064; }
  button.on { border-color:var(--accent); background:#2a1518; color:#fff; }
  .list { display:flex; flex-direction:column; gap:4px; }
  .list button { text-align:left; }
  .group { color:var(--dim); font-size:11px; margin:8px 0 2px; letter-spacing:.06em; }
  .toolbar { display:flex; gap:8px; align-items:center; margin-bottom:8px; flex-wrap:wrap; }
  .toolbar .spacer { flex:1; }
  #shot { width:100%; max-height:68vh; object-fit:contain; display:block;
          background:#05080b; border:1px solid var(--line); border-radius:6px; padding:4px; }
  .panels { display:grid; grid-template-columns:1.1fr 1fr; gap:12px; margin-top:10px; }
  .card { border:1px solid var(--line); border-radius:6px; background:var(--panel); padding:10px 12px; }
  .card h3 { margin:0 0 6px; font-size:12px; color:#fff; font-weight:600; }
  .kv { display:grid; grid-template-columns:110px 1fr; gap:3px 10px; font-size:12px; }
  .kv div:nth-child(odd) { color:var(--dim); }
  .hint { color:var(--dim); font-size:11.5px; }
  code { color:#ffd38a; }
  #toast { position:fixed; right:16px; bottom:16px; background:#2a1518; border:1px solid var(--accent);
           color:#ffd7d7; padding:6px 10px; border-radius:6px; font-size:12px; display:none; }
</style>
</head>
<body>
<header>
  <h1>RoboDojo · 任务 / 子任务 / 阶段</h1>
  <span class="meta" id="contract"></span>
</header>
<div class="layout">
  <aside>
    <h2>任务</h2><div class="list" id="tasks"></div>
    <h2>子任务（一次抓取）</h2><div class="list" id="subs"></div>
    <h2>阶段（不定长轨迹）</h2><div class="list" id="phases"></div>
  </aside>
  <main>
    <div class="toolbar">
      <button id="overview">阶段总览图</button>
      <button id="prevph">← 上一阶段</button>
      <button id="play">▶ 播放</button>
      <button id="nextph">下一阶段 →</button>
      <input type="range" id="slider" min="0" max="0" value="0" style="flex:1;max-width:420px">
      <span id="stepinfo" class="meta"></span>
      <span class="spacer"></span>
      <span id="where" class="meta"></span>
    </div>
    <div class="hint" style="margin-bottom:6px">
      快捷键：<b>↓</b> 下一个阶段（到底后进入下一子任务/任务）、<b>↑</b> 上一个、
      <b>←/→</b> 单步帧、<b>空格</b> 播放。
    </div>
    <img id="shot" alt="phase frame">
    <div class="panels">
      <div class="card">
        <h3>这一页在看什么</h3>
        <div class="hint">
          一个阶段 = 一条<b>不定长的稠密点轨迹</b>（6.25 Hz：每 4 个动作步一个点）。
          阶段分 <b>reach / pick / move / place / stop</b>：<b>stop</b> 表示这只手保持不动
          （等待另一只手，或拿着东西等），另一只手可能仍在动。
          数据侧<b>不切 chunk</b>；训练时从这条轨迹上滑一个 7 点窗口（对应 25 步 action）。
          画面上：<b>红/绿</b> = 抓取臂，<b>蓝灰</b> = 另一只手，<b>橙</b> = 物体关键点，
          细线是整段轨迹，粗线是最近 1 秒的轨迹。
        </div>
      </div>
      <div class="card" id="meta"></div>
    </div>
  </main>
</div>
<div id="toast"></div>
<script>
const DATA = __DATA__;
const S = { t:0, s:0, p:0, f:0, playing:false, timer:null, overview:false };
const IMG = new Map();          // src -> preloaded Image
const PRELOAD = 24;             // frames ahead

const el = id => document.getElementById(id);
const task = () => DATA.tasks[S.t];
const sub = () => task().subtasks[S.s];
const phase = () => sub().phases[S.p];
const nframes = () => (phase() ? phase().frames.length : 0);

function toast(msg){
  const t = el("toast"); t.textContent = msg; t.style.display = "block";
  clearTimeout(t._h); t._h = setTimeout(()=>{ t.style.display="none"; }, 1600);
}
function init(){
  const m = DATA.meta;
  el("contract").textContent =
    `${m.tracker} · 轨迹 ${m.traj_hz} Hz（每 ${m.stride} 个动作步一个点）· ` +
    `阶段 = 不定长轨迹；chunk（7 点 / ${m.chunk} 步）由训练侧滑窗生成`;
  DATA.tasks.forEach((t,i)=>{
    const b = document.createElement("button");
    b.textContent = `${t.task}  (${t.subtasks.length} 子任务)`;
    b.onclick = ()=>{ S.t=i; S.s=0; S.p=0; S.f=0; S.overview=false; stop(); render(); };
    el("tasks").appendChild(b);
  });
  el("overview").onclick = ()=>{ S.overview=!S.overview; stop(); render(); };
  el("prevph").onclick = ()=>nextPhase(-1);
  el("nextph").onclick = ()=>nextPhase(+1);
  el("play").onclick = togglePlay;
  el("slider").oninput = e => setFrame(parseInt(e.target.value,10));
  document.addEventListener("keydown", e=>{
    if(e.key==="ArrowLeft"){ e.preventDefault(); setFrame(S.f-1); }
    if(e.key==="ArrowRight"){ e.preventDefault(); setFrame(S.f+1); }
    if(e.key==="ArrowDown"){ e.preventDefault(); nextPhase(+1); }
    if(e.key==="ArrowUp"){ e.preventDefault(); nextPhase(-1); }
    if(e.key===" "){ e.preventDefault(); togglePlay(); }
  });
  render();
}
function preload(from, n = PRELOAD){
  const p = phase(); if(!p) return;
  const end = Math.min(p.frames.length - 1, from + n);
  for(let i = Math.max(0, from); i <= end; i++){
    const src = p.frames[i];
    if(IMG.has(src)) continue;
    const im = new Image(); im.src = src;
    IMG.set(src, im);
  }
  while(IMG.size > 400){ IMG.delete(IMG.keys().next().value); }
}
function setFrame(k){
  if(S.overview) return;
  S.f = Math.max(0, Math.min(nframes()-1, k));
  renderStage();
  preload(S.f, PRELOAD);
}
function stop(){
  if(S.timer){ clearTimeout(S.timer); S.timer=null; }
  S.playing=false; el("play").textContent="▶ 播放";
}
function playLoop(){
  if(!S.playing) return;
  const n = nframes();
  if(S.f >= n - 1){ stop(); return; }
  const next = S.f + 1;
  const src = phase().frames[next];
  let im = IMG.get(src);
  if(!im){ im = new Image(); im.src = src; IMG.set(src, im); }
  const go = () => {
    if(!S.playing) return;
    setFrame(next);
    S.timer = setTimeout(playLoop, 50);
  };
  if(im.complete && im.naturalWidth > 0) go();
  else { im.onload = go; im.onerror = go; }
}
function togglePlay(){
  if(S.overview){ toast("先关掉总览图"); return; }
  if(S.playing){ stop(); return; }
  if(S.f >= nframes()-1) S.f = 0;
  S.playing = true; el("play").textContent = "⏸ 暂停";
  preload(S.f, PRELOAD);
  playLoop();
}
function nextPhase(dir){
  S.overview = false;
  const T = DATA.tasks, t = T[S.t], s = t.subtasks[S.s];
  if(dir > 0){
    if(S.p < s.phases.length - 1) S.p++;
    else if(S.s < t.subtasks.length - 1){ S.s++; S.p = 0; }
    else if(S.t < T.length - 1){ S.t++; S.s = 0; S.p = 0; }
  } else {
    if(S.p > 0) S.p--;
    else if(S.s > 0){ S.s--; S.p = t.subtasks[S.s].phases.length - 1; }
    else if(S.t > 0){ S.t--; S.s = T[S.t].subtasks.length - 1;
                      S.p = T[S.t].subtasks[S.s].phases.length - 1; }
  }
  S.f = 0; stop(); render();
}
function button(container, text, on, fn){
  const b = document.createElement("button");
  b.textContent = text; b.className = on ? "on" : ""; b.onclick = fn;
  container.appendChild(b);
}
function render(){
  [...el("tasks").children].forEach((b,i)=>b.className = i===S.t ? "on" : "");
  const subs = el("subs"); subs.innerHTML = "";
  let lastArm = null;
  task().subtasks.forEach((s,i)=>{
    if(s.arm !== lastArm){
      const h = document.createElement("div");
      h.className = "group";
      h.textContent = s.arm === "L" ? "左手" : (s.arm === "R" ? "右手" : s.arm);
      subs.appendChild(h);
      lastArm = s.arm;
    }
    button(subs, `#${s.index+1}  [${s.t_close}-${s.t_open}]`, i===S.s,
           ()=>{ S.s=i; S.p=0; S.f=0; stop(); render(); });
  });
  const phases = el("phases"); phases.innerHTML = "";
  sub().phases.forEach((p,i)=>{
    button(phases, `${p.phase}  [${p.t_start}-${p.t_end}]  ${p.n_points}pts`, i===S.p,
           ()=>{ S.p=i; S.f=0; stop(); render(); });
  });
  el("where").textContent = `${task().task} · #${sub().index+1}${sub().arm} · ` +
    `${phase().phase} · ${phase().n_points} 点 / ${phase().duration_steps} 步`;
  el("slider").max = Math.max(0, nframes()-1);
  preload(S.f, PRELOAD);
  renderStage();
}
function renderStage(){
  const p = phase();
  if(S.overview){
    el("shot").src = p.overview;
    el("stepinfo").textContent = "";
    el("meta").innerHTML = metaHtml(null);
    return;
  }
  el("shot").src = p.frames[S.f];
  el("slider").value = S.f;
  el("stepinfo").innerHTML = `帧 <b>${S.f+1}</b>/${p.frames.length} · ` +
    `t=${p.t_start + S.f} / ${task().T - 1}`;
  el("meta").innerHTML = metaHtml(p);
}
function qcText(qc){
  if(!qc || !qc.seed_uv) return "无（只有 anchor）";
  const f = x => (typeof x === "number" ? x.toFixed(1) : x);
  return `seed (${qc.seed_uv[0]}, ${qc.seed_uv[1]}) @ ${qc.t_seed} · ` +
         `rigid ${f(qc.rigidity)} · move ${f(qc.movement)} · vis ${f(qc.visibility)}`;
}
function metaHtml(p){
  const s = sub();
  const head = `<h3>子任务 #${s.index+1} · ${s.arm} 臂</h3><div class="kv">
    <div>抓取 / 释放</div><div>${s.t_close} / ${s.t_open}</div>
    <div>物体 seed</div><div>${qcText(s.object_qc)}</div>`;
  if(!p) return head + `</div>`;
  const mask = s.mask_image
    ? `<img src="${s.mask_image}" style="width:100%;border:1px solid var(--line);` +
      `border-radius:4px;margin-top:8px" alt="object mask">` : "";
  return head + `<div>阶段</div><div>${p.phase} · 帧 ${p.t_start}–${p.t_end} · ` +
    `${p.n_points} 点 / ${p.duration_steps} 步</div>` +
    `<div>物体有效</div><div>${p.object_valid}/${p.n_points}</div>` +
    `<div>数据</div><div>sidecar：整段轨迹（无 chunk）</div></div>` + mask;
}
init();
</script>
</body>
</html>
"""


def write_index(out_dir: Path, data: dict) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "index.html"
    path.write_text(_HTML.replace("__DATA__", json.dumps(data, ensure_ascii=False,
                                                         allow_nan=False)),
                    encoding="utf-8")
    return path
