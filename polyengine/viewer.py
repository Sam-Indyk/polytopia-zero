"""Self-contained HTML replay viewer.

`render_html(replay_dict)` returns a single HTML document with the replay JSON
embedded, so the output opens straight from the filesystem — no server, no CORS,
nothing to install. Canvas 2D, keyboard scrubbing, and a fog toggle that dims
tiles a chosen player has never explored (rendered from the recorded
`fog_explored` masks, the same memory contract the observation encoder uses).
"""
from __future__ import annotations

import json

_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>PolyZero replay</title>
<style>
 :root{color-scheme:dark}
 body{margin:0;background:#12151c;color:#e6e9ef;font:14px/1.5 ui-sans-serif,system-ui,sans-serif}
 header{padding:10px 16px;border-bottom:1px solid #2a3040;display:flex;gap:16px;align-items:center;flex-wrap:wrap}
 h1{font-size:15px;margin:0;font-weight:600}
 .muted{color:#8b93a7}
 main{display:flex;gap:16px;padding:16px;flex-wrap:wrap}
 canvas{background:#0c0e13;border:1px solid #2a3040;border-radius:6px;image-rendering:pixelated}
 aside{min-width:260px;flex:1}
 .row{display:flex;gap:8px;align-items:center;margin-bottom:8px;flex-wrap:wrap}
 button{background:#232838;color:#e6e9ef;border:1px solid #39415a;border-radius:5px;padding:5px 11px;cursor:pointer;font:inherit}
 button:hover{background:#2c3346}
 input[type=range]{flex:1;min-width:200px}
 table{border-collapse:collapse;width:100%;margin-top:8px}
 td,th{padding:3px 8px;border-bottom:1px solid #232838;text-align:left}
 .log{max-height:300px;overflow:auto;font-family:ui-monospace,monospace;font-size:12px;margin-top:8px;
      border:1px solid #232838;border-radius:5px}
 .log div{padding:2px 8px;white-space:nowrap}
 .log div.cur{background:#2a3552}
 .sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px;vertical-align:-1px}
</style></head><body>
<header>
  <h1>PolyZero replay</h1>
  <span class="muted" id="meta"></span>
</header>
<main>
 <div>
  <canvas id="cv" width="640" height="640"></canvas>
  <div class="row" style="margin-top:10px">
   <button id="first">&#124;&#9664;</button><button id="prev">&#9664;</button>
   <button id="play">&#9654; play</button>
   <button id="next">&#9654;</button><button id="last">&#9654;&#124;</button>
   <span class="muted" id="pos"></span>
  </div>
  <div class="row"><input type="range" id="scrub" min="0" value="0"></div>
  <div class="row">
   <label>fog of</label>
   <select id="fog"><option value="-1">off (ground truth)</option></select>
   <span class="muted">&#8592; &#8594; step &nbsp; space play</span>
  </div>
 </div>
 <aside>
  <div id="players"></div>
  <div class="log" id="log"></div>
 </aside>
</main>
<script>
const R = __REPLAY__;
const S = R.size, N = S*S, F = R.frames;
const cv = document.getElementById('cv'), cx = cv.getContext('2d');
const CELL = Math.floor(cv.width / S);
const PC = ['#4da3ff','#ff6b6b','#5fd68a','#ffd166'];       // player colours
const TERR = {ocean:'#12335c',water:'#1d5c96',field:'#2f4a2c',forest:'#1d3a20',mountain:'#4a4a52'};
let i = 0, timer = null, fogOf = -1;

document.getElementById('meta').textContent =
  `${S}x${S} · seed ${R.map_seed} · rules ${R.ruleset_version} · ${F.length} frames · ` +
  R.tribes.map((t,p)=>`P${p} ${t} (${R.agents[p]||'?'})`).join(' vs ') +
  (R.terminal ? ` · winner: ${R.winner>=0?'P'+R.winner:'draw'}` : ' · unfinished');

const fogSel = document.getElementById('fog');
R.tribes.forEach((t,p)=>{const o=document.createElement('option');o.value=p;o.textContent=`P${p} (${t})`;fogSel.appendChild(o);});
fogSel.onchange = e => { fogOf = +e.target.value; draw(); };

function terrColor(t){ return TERR[R.terrain_names[t]] || '#333'; }

function draw(){
  const f = F[i];
  cx.clearRect(0,0,cv.width,cv.height);
  for(let k=0;k<N;k++){
    const x=(k%S)*CELL, y=Math.floor(k/S)*CELL;
    const hidden = fogOf>=0 && !f.explored[fogOf][k];
    cx.fillStyle = terrColor(R.terrain[k]);
    cx.fillRect(x,y,CELL,CELL);
    if(f.owner[k]>=0){                                       // territory tint
      cx.globalAlpha=0.22; cx.fillStyle=PC[f.owner[k]%4];
      cx.fillRect(x,y,CELL,CELL); cx.globalAlpha=1;
    }
    cx.strokeStyle='rgba(0,0,0,.35)'; cx.strokeRect(x+.5,y+.5,CELL-1,CELL-1);
    if(hidden){ cx.fillStyle='rgba(6,8,12,.86)'; cx.fillRect(x,y,CELL,CELL); continue; }
    if(f.city_level[k]>0){                                   // city
      cx.fillStyle = f.city_owner[k]>=0 ? PC[f.city_owner[k]%4] : '#9aa3b8';
      const s=CELL*0.5, o=(CELL-s)/2;
      cx.fillRect(x+o,y+o,s,s);
      cx.strokeStyle = f.city_capital[k] ? '#fff' : 'rgba(0,0,0,.6)';
      cx.lineWidth = f.city_capital[k] ? 2 : 1;
      cx.strokeRect(x+o,y+o,s,s); cx.lineWidth=1;
      cx.fillStyle='#0c0e13'; cx.font=`bold ${Math.floor(CELL*0.3)}px monospace`;
      cx.textAlign='center'; cx.textBaseline='middle';
      cx.fillText(f.city_level[k], x+CELL/2, y+CELL/2+1);
    }
    if(f.unit_type[k]>=0){                                   // unit
      cx.beginPath(); cx.arc(x+CELL/2,y+CELL/2,CELL*0.29,0,7);
      cx.fillStyle=PC[f.unit_owner[k]%4]; cx.fill();
      cx.strokeStyle='rgba(0,0,0,.7)'; cx.stroke();
      cx.fillStyle='#0c0e13'; cx.font=`bold ${Math.floor(CELL*0.32)}px monospace`;
      cx.textAlign='center'; cx.textBaseline='middle';
      cx.fillText(R.unit_names[f.unit_type[k]][0].toUpperCase(), x+CELL/2, y+CELL/2+1);
    }
  }
  if(f.action && f.action.src>=0){                           // highlight last action
    const hl=(k,c)=>{const x=(k%S)*CELL,y=Math.floor(k/S)*CELL;
      cx.strokeStyle=c;cx.lineWidth=2;cx.strokeRect(x+1,y+1,CELL-2,CELL-2);cx.lineWidth=1;};
    hl(f.action.src,'#ffd166');
    if(f.action.dst>=0) hl(f.action.dst, f.action.name==='ATTACK'?'#ff6b6b':'#5fd68a');
  }
  let ph='<table><tr><th>player</th><th>stars</th><th>cities</th><th>units</th></tr>';
  for(let p=0;p<R.n_players;p++){
    let c=0,u=0;
    for(let k=0;k<N;k++){ if(f.city_owner[k]===p&&f.city_level[k]>0)c++; if(f.unit_owner[k]===p&&f.unit_type[k]>=0)u++; }
    ph+=`<tr><td><span class="sw" style="background:${PC[p%4]}"></span>P${p} ${R.tribes[p]}${f.alive[p]?'':' &#9760;'}</td>`+
        `<td>${f.stars[p]}</td><td>${c}</td><td>${u}</td></tr>`;
  }
  document.getElementById('players').innerHTML = ph+'</table>';
  document.getElementById('pos').textContent =
    `frame ${i}/${F.length-1} · turn ${f.turn} · to move P${f.cur}`;
  document.getElementById('scrub').value = i;
  renderLog();
}

function renderLog(){
  const lo=Math.max(0,i-40), hi=Math.min(F.length,i+8);
  let h='';
  for(let k=lo;k<hi;k++){
    const a=F[k].action;
    h+=`<div class="${k===i?'cur':''}" data-i="${k}">${String(k).padStart(4)} `+
       (a?`P${a.player} ${a.desc}`:'<span class="muted">initial state</span>')+`</div>`;
  }
  const el=document.getElementById('log'); el.innerHTML=h;
  const cur=el.querySelector('.cur'); if(cur) cur.scrollIntoView({block:'nearest'});
  el.querySelectorAll('div').forEach(d=>d.onclick=()=>{i=+d.dataset.i;draw();});
}

function go(n){ i=Math.max(0,Math.min(F.length-1,n)); draw(); }
document.getElementById('scrub').max=F.length-1;
document.getElementById('scrub').oninput=e=>go(+e.target.value);
document.getElementById('first').onclick=()=>go(0);
document.getElementById('prev').onclick=()=>go(i-1);
document.getElementById('next').onclick=()=>go(i+1);
document.getElementById('last').onclick=()=>go(F.length-1);
const playBtn=document.getElementById('play');
function toggle(){
  if(timer){clearInterval(timer);timer=null;playBtn.innerHTML='&#9654; play';}
  else{if(i>=F.length-1)i=0;
       timer=setInterval(()=>{ if(i>=F.length-1){toggle();return;} go(i+1); },90);
       playBtn.innerHTML='&#10074;&#10074; pause';}
}
playBtn.onclick=toggle;
addEventListener('keydown',e=>{
  if(e.key==='ArrowRight')go(i+1); else if(e.key==='ArrowLeft')go(i-1);
  else if(e.key===' '){e.preventDefault();toggle();}
  else if(e.key==='Home')go(0); else if(e.key==='End')go(F.length-1);
});
draw();
</script></body></html>
"""


def render_html(replay: dict) -> str:
    """Embed a replay dict into a standalone HTML viewer document."""
    payload = json.dumps(replay, separators=(",", ":"))
    # Guard against a literal </script> ever appearing inside the JSON.
    payload = payload.replace("</", "<\\/")
    return _HTML.replace("__REPLAY__", payload)


def save_html(replay: dict, path) -> "Path":
    from pathlib import Path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_html(replay), encoding="utf-8")
    return path
