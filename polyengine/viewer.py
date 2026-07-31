"""Self-contained HTML replay viewer.

`render_html(replay_dict)` returns a single HTML document with the replay JSON
embedded, so the output opens straight from the filesystem — no server, no CORS,
nothing to install.

Design rule: **nothing the engine knows is hidden.** The board renders terrain,
resources, territory, cities (level / population / walls / capital) and units
(type / HP / veteran / whether they've acted). Side panels show every player's
stars and full tech tree with live research costs, a click-anywhere tile
inspector, and the complete action log. A fog selector re-renders the board as a
chosen player actually saw it, from the recorded explored masks. Mechanics the
engine does not simulate yet are listed explicitly rather than silently absent.
"""
from __future__ import annotations

import json

_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>PolyZero replay</title>
<style>
 :root{color-scheme:dark}
 *{box-sizing:border-box}
 body{margin:0;background:#12151c;color:#e6e9ef;font:13px/1.5 ui-sans-serif,system-ui,sans-serif}
 header{padding:9px 16px;border-bottom:1px solid #2a3040;display:flex;gap:14px;align-items:baseline;flex-wrap:wrap}
 h1{font-size:14px;margin:0;font-weight:600}
 h2{font-size:11px;margin:0 0 6px;font-weight:600;text-transform:uppercase;letter-spacing:.07em;color:#8b93a7}
 .muted{color:#8b93a7}
 main{display:flex;gap:14px;padding:14px;align-items:flex-start;flex-wrap:wrap}
 canvas{background:#0c0e13;border:1px solid #2a3040;border-radius:6px;cursor:crosshair;max-width:100%}
 .col{display:flex;flex-direction:column;gap:12px}
 .side{min-width:290px;max-width:420px;flex:1}
 .card{border:1px solid #232838;border-radius:6px;padding:10px;background:#161a23}
 .row{display:flex;gap:7px;align-items:center;flex-wrap:wrap}
 button{background:#232838;color:#e6e9ef;border:1px solid #39415a;border-radius:5px;padding:5px 10px;cursor:pointer;font:inherit}
 button:hover{background:#2c3346}
 select{background:#232838;color:#e6e9ef;border:1px solid #39415a;border-radius:5px;padding:4px 7px;font:inherit}
 input[type=range]{flex:1;min-width:180px}
 table{border-collapse:collapse;width:100%}
 td,th{padding:3px 7px;border-bottom:1px solid #232838;text-align:left;vertical-align:top}
 th{color:#8b93a7;font-weight:600}
 .log{max-height:260px;overflow:auto;font-family:ui-monospace,monospace;font-size:11.5px}
 .log div{padding:2px 7px;white-space:nowrap;border-radius:3px;cursor:pointer}
 .log div:hover{background:#1e2431}
 .log div.cur{background:#2a3552}
 .sw{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:5px;vertical-align:0}
 .tech{display:flex;flex-wrap:wrap;gap:3px;margin-top:3px}
 .t{font-size:10.5px;padding:1px 6px;border-radius:9px;border:1px solid #39415a;color:#b9c2d6}
 .t.done{background:#1f4a2e;border-color:#2f7a49;color:#c9f0d6}
 .t.avail{border-color:#5a6a8c;color:#e6e9ef}
 .t.lock{opacity:.38}
 code{font-family:ui-monospace,monospace;font-size:11.5px;color:#b9c2d6}
 .warn{border-color:#4a3a1e;background:#1e1a12}
 .warn li{margin:1px 0}
 ul{margin:4px 0 0;padding-left:17px}
 details>summary{cursor:pointer;list-style:none;color:#8b93a7;font:600 11px/1 ui-monospace,monospace;
   letter-spacing:.07em;text-transform:uppercase;margin-bottom:6px}
 details>summary::-webkit-details-marker{display:none}
 details>summary:before{content:"▸ ";display:inline-block;transition:transform .15s}
 details[open]>summary:before{transform:rotate(90deg)}
 .key{display:grid;grid-template-columns:auto 1fr;gap:4px 8px;align-items:center;font-size:12px}
 .key .lbl{color:#b9c2d6}
 .key .grp{grid-column:1/-1;color:#8b93a7;font:600 10px/1 ui-monospace,monospace;
   letter-spacing:.08em;text-transform:uppercase;margin-top:7px}
 .chip{display:inline-flex;align-items:center;justify-content:center;
   width:22px;height:22px;border-radius:3px;border:1px solid rgba(0,0,0,.4);
   font:bold 11px/1 ui-monospace,monospace}
 .dot{display:inline-block;width:14px;height:14px;border-radius:50%;
   border:1px solid rgba(0,0,0,.6);vertical-align:-2px}
</style></head><body>
<header>
  <h1>PolyZero replay</h1>
  <span class="muted" id="meta"></span>
</header>
<main>
 <div class="col">
  <canvas id="cv" width="660" height="660"></canvas>
  <div class="row">
   <button id="first">&#124;&#9664;</button><button id="prev">&#9664;</button>
   <button id="play">&#9654; play</button>
   <button id="next">&#9654;</button><button id="last">&#9654;&#124;</button>
   <span class="muted" id="pos"></span>
  </div>
  <div class="row"><input type="range" id="scrub" min="0" value="0"></div>
  <div class="row">
   <label class="muted">view as</label>
   <select id="fog"><option value="-1">god's-eye (all)</option></select>
   <label class="muted">speed</label>
   <select id="speed">
     <option value="220">slow</option><option value="90" selected>normal</option>
     <option value="25">fast</option>
   </select>
   <span class="muted">&#8592;&#8594; step &middot; space play &middot; click a tile</span>
  </div>
 </div>
 <div class="col side">
  <div class="card"><h2>players</h2><div id="players"></div></div>
  <div class="card"><h2>tile inspector</h2><div id="tile" class="muted">click any tile</div></div>
  <div class="card"><h2>action log</h2><div class="log" id="log"></div></div>
 </div>
 <div class="col side">
  <div class="card"><h2>tech tree</h2><div id="techs"></div></div>
  <div class="card"><details open><summary>key &mdash; what am I looking at?</summary>
    <div class="key" id="key"></div></details></div>
  <div class="card warn"><h2>not simulated yet</h2>
    <div class="muted">These mechanics are not in the engine, so they never appear
    in the log &mdash; nothing is being hidden from you.</div>
    <ul id="missing" class="muted"></ul></div>
 </div>
</main>
<script>
const R = __REPLAY__;
const S = R.size, N = S*S, F = R.frames;
const cv = document.getElementById('cv'), cx = cv.getContext('2d');
const CELL = Math.floor(cv.width / S);
const PC = ['#4da3ff','#ff6b6b','#5fd68a','#ffd166'];
const TERR = {ocean:'#0f2b4d',water:'#1a5286',field:'#33512f',forest:'#1e3d22',mountain:'#4d4d56'};
const RES  = {fruit:['#ff8fb1','\\u25cf'],crop:['#ffd166','\\u25b2'],game:['#c58b5a','\\u25c6'],
              metal:['#c9ccd6','\\u25a0'],fish:['#7fe3e3','\\u25cf'],
              starfish:['#ffb3d9','\\u2731']};
// Buildings the player paid for; population is what these buy. The ruin is the
// exception -- nobody built it, and it is examined rather than harvested.
const IMP  = {lumber_hut:'L',farm:'F',mine:'M',port:'P',temple:'T',
              mountain_temple:'T',forest_temple:'T',water_temple:'T',
              sawmill:'S',windmill:'W',forge:'G',ruin:'\\u2302',
              road:'=',bridge:'≡',market:'$',embassy:'E',lighthouse:'✶',
              altar_of_peace:'♈',emperors_tomb:'♈',eye_of_god:'♈',
              gate_of_power:'♈',grand_bazaar:'♈',park_of_fortune:'♈',
              tower_of_wisdom:'♈'};
let i = 0, timer = null, fogOf = -1, sel = -1, speed = 90;

document.getElementById('meta').textContent =
  `${S}\\u00d7${S} \\u00b7 seed ${R.map_seed} \\u00b7 rules ${R.ruleset_version} \\u00b7 `+
  `${F.length} frames \\u00b7 ` +
  R.tribes.map((t,p)=>`P${p} ${t} (${R.agents[p]||'?'})`).join(' vs ') +
  (R.terminal ? ` \\u00b7 winner: ${R.winner>=0?'P'+R.winner:'draw'}` : ' \\u00b7 unfinished');
document.getElementById('missing').innerHTML =
  (R.not_simulated||[]).map(m=>`<li>${m}</li>`).join('');

const fogSel=document.getElementById('fog');
R.tribes.forEach((t,p)=>{const o=document.createElement('option');
  o.value=p;o.textContent=`P${p} (${t}) fog`;fogSel.appendChild(o);});
fogSel.onchange=e=>{fogOf=+e.target.value;draw();};
document.getElementById('speed').onchange=e=>{
  speed=+e.target.value; if(timer){toggle();toggle();}};

const hasTech=(mask,t)=>((mask>>>t)&1)===1;
const cityCount=(f,p)=>{let c=0;for(let k=0;k<N;k++)if(f.city_owner[k]===p&&f.city_level[k]>0)c++;return c;};
function researchCost(f,p,t){
  const base=R.tech_tier[t]*Math.max(1,cityCount(f,p))+4;
  const phil=R.tech_names.indexOf('philosophy');
  return (phil>=0&&hasTech(f.tech[p],phil)) ? Math.ceil(base*2/3) : base;
}

// Seat number whose biome a tile is, or -1 for an old replay without them.
function biomeName(k){
  const ids=R.tile_tribe||[], names=R.tribe_names||[];
  return (k<ids.length && ids[k]<names.length) ? names[ids[k]] : null;
}
function biomeSeat(k){
  const n=biomeName(k);
  return n===null ? -1 : (R.tribes||[]).indexOf(n);
}

function draw(){
  const f=F[i];
  cx.clearRect(0,0,cv.width,cv.height);
  for(let k=0;k<N;k++){
    const x=(k%S)*CELL, y=Math.floor(k/S)*CELL;
    cx.fillStyle=TERR[R.terrain_names[R.terrain[k]]]||'#333';
    cx.fillRect(x,y,CELL,CELL);
    // Biome tint: which tribe's spawn rates made this tile. It never changes
    // hands, so it is drawn faintly under the territory tint rather than
    // competing with it.
    if(biomeSeat(k)>=0){cx.globalAlpha=.10;cx.fillStyle=PC[biomeSeat(k)%4];
      cx.fillRect(x,y,CELL,CELL);cx.globalAlpha=1;}
    if(f.owner[k]>=0){cx.globalAlpha=.24;cx.fillStyle=PC[f.owner[k]%4];
      cx.fillRect(x,y,CELL,CELL);cx.globalAlpha=1;}
    cx.strokeStyle='rgba(0,0,0,.35)';cx.strokeRect(x+.5,y+.5,CELL-1,CELL-1);
    if(fogOf>=0 && !f.explored[fogOf][k]){
      cx.fillStyle='rgba(6,8,12,.88)';cx.fillRect(x,y,CELL,CELL);continue;}
    // resource
    const rn=R.resource_names[f.resource[k]], rv=RES[rn];
    if(rv){cx.fillStyle=rv[0];cx.font=`${Math.floor(CELL*.3)}px serif`;
      cx.textAlign='left';cx.textBaseline='top';cx.fillText(rv[1],x+2,y+1);}
    // improvement (building) marker
    const imp=f.improvement[k];
    if(imp>=0){
      const ch=IMP[R.improvement_names[imp]];
      if(ch){cx.fillStyle='#e9d8a6';cx.font=`bold ${Math.floor(CELL*.26)}px monospace`;
        cx.textAlign='right';cx.textBaseline='top';cx.fillText(ch,x+CELL-2,y+1);}
    }
    // city
    if(f.city_level[k]>0){
      const s=CELL*.52,o=(CELL-s)/2;
      cx.fillStyle=f.city_owner[k]>=0?PC[f.city_owner[k]%4]:'#9aa3b8';
      cx.fillRect(x+o,y+o,s,s);
      if(f.city_walls[k]){cx.strokeStyle='#fff';cx.lineWidth=2;
        cx.strokeRect(x+o-2,y+o-2,s+4,s+4);cx.lineWidth=1;}
      if(f.city_pending_reward[k]>0){       // owes a level-up choice
        cx.strokeStyle='#ffd166';cx.lineWidth=3;
        cx.strokeRect(x+o-4,y+o-4,s+8,s+8);cx.lineWidth=1;}
      cx.strokeStyle=f.city_capital[k]?'#ffd166':'rgba(0,0,0,.6)';
      cx.lineWidth=f.city_capital[k]?2:1;cx.strokeRect(x+o,y+o,s,s);cx.lineWidth=1;
      cx.fillStyle='#0c0e13';cx.font=`bold ${Math.floor(CELL*.3)}px monospace`;
      cx.textAlign='center';cx.textBaseline='middle';
      cx.fillText(f.city_level[k],x+CELL/2,y+CELL/2+1);
    }
    // unit
    if(f.unit_type[k]>=0){
      const spent=f.unit_moved[k]&&f.unit_attacked[k];
      // A hidden Cloak is drawn ghosted with a dashed outline: the god's-eye
      // view still shows it (nothing is hidden from YOU), but you can tell at a
      // glance that the opponent cannot see it.
      const hidden=(f.unit_hidden||[])[k];
      cx.globalAlpha=hidden?.3:(spent?.55:1);
      cx.beginPath();cx.arc(x+CELL/2,y+CELL*.62,CELL*.27,0,7);
      cx.fillStyle=PC[f.unit_owner[k]%4];cx.fill();
      if(hidden){cx.setLineDash([3,2]);cx.strokeStyle='#e6e9ef';cx.lineWidth=1.5;}
      else{cx.strokeStyle=f.unit_veteran[k]?'#ffd166':'rgba(0,0,0,.7)';
        cx.lineWidth=f.unit_veteran[k]?2:1;}
      cx.stroke();cx.setLineDash([]);cx.lineWidth=1;
      cx.fillStyle='#0c0e13';cx.font=`bold ${Math.floor(CELL*.28)}px monospace`;
      cx.textAlign='center';cx.textBaseline='middle';
      cx.fillText(R.unit_names[f.unit_type[k]][0].toUpperCase(),x+CELL/2,y+CELL*.62+1);
      cx.globalAlpha=1;
      // hp bar
      const mx=R.unit_max_hp[f.unit_type[k]], frac=Math.max(0,f.unit_hp[k]/mx);
      cx.fillStyle='rgba(0,0,0,.65)';cx.fillRect(x+3,y+CELL-6,CELL-6,3);
      cx.fillStyle=frac>.5?'#5fd68a':frac>.25?'#ffd166':'#ff6b6b';
      cx.fillRect(x+3,y+CELL-6,(CELL-6)*frac,3);
    }
  }
  const a=f.action;
  if(a&&a.src>=0){
    const hl=(k,c,w)=>{const x=(k%S)*CELL,y=Math.floor(k/S)*CELL;
      cx.strokeStyle=c;cx.lineWidth=w;cx.strokeRect(x+1.5,y+1.5,CELL-3,CELL-3);cx.lineWidth=1;};
    hl(a.src,'#ffd166',2);
    if(a.dst>=0&&a.dst!==a.src) hl(a.dst,a.name==='ATTACK'?'#ff6b6b':'#5fd68a',2);
    if(a.name==='CAPTURE') hl(a.src,'#5fd68a',3);
  }
  if(sel>=0){const x=(sel%S)*CELL,y=Math.floor(sel/S)*CELL;
    cx.strokeStyle='#fff';cx.lineWidth=2;cx.strokeRect(x+1,y+1,CELL-2,CELL-2);cx.lineWidth=1;}
  renderPlayers();renderTechs();renderTile();renderLog();
  document.getElementById('pos').textContent=
    `frame ${i}/${F.length-1} \\u00b7 turn ${f.turn} \\u00b7 to move P${f.cur}`;
  document.getElementById('scrub').value=i;
}

function renderPlayers(){
  const f=F[i];
  let h='<table><tr><th>player</th><th>stars</th><th>cities</th><th>units</th><th>cap</th><th>techs</th></tr>';
  for(let p=0;p<R.n_players;p++){
    let u=0,cap=0;
    for(let k=0;k<N;k++){
      if(f.unit_owner[k]===p&&f.unit_type[k]>=0)u++;
      if(f.city_owner[k]===p&&f.city_level[k]>0)cap+=f.city_level[k]+1;
    }
    let nt=0;for(let t=0;t<R.tech_names.length;t++)if(hasTech(f.tech[p],t))nt++;
    h+=`<tr><td><span class="sw" style="background:${PC[p%4]}"></span>P${p} ${R.tribes[p]}`+
       `${f.alive[p]?'':' <span class="muted">(out)</span>'}${f.cur===p?' \\u25c0':''}</td>`+
       `<td>${f.stars[p]}</td><td>${cityCount(f,p)}</td><td>${u}</td>`+
       `<td>${cap}</td><td>${nt}</td></tr>`;
  }
  document.getElementById('players').innerHTML=h+'</table>';
}

function renderTechs(){
  const f=F[i];let h='';
  for(let p=0;p<R.n_players;p++){
    h+=`<div style="margin-bottom:7px"><span class="sw" style="background:${PC[p%4]}"></span>`+
       `<b>P${p}</b> <span class="muted">${R.tribes[p]} \\u00b7 ${f.stars[p]}\\u2605</span><div class="tech">`;
    for(let t=0;t<R.tech_names.length;t++){
      const done=hasTech(f.tech[p],t), req=R.tech_requires[t];
      const open=req<0||hasTech(f.tech[p],req);
      const cost=researchCost(f,p,t);
      const cls=done?'done':(open?'avail':'lock');
      const tip=done?'researched':(open?`costs ${cost}\\u2605 (T${R.tech_tier[t]})`
                 :`needs ${R.tech_names[req]}`);
      const badge=done?'':' <span class="muted">'+(open?cost+'\\u2605':'locked')+'</span>';
      h+=`<span class="t ${cls}" title="${tip}">${R.tech_names[t]}${badge}</span>`;
    }
    h+='</div></div>';
  }
  document.getElementById('techs').innerHTML=h;
}

function renderTile(){
  const el=document.getElementById('tile');
  if(sel<0){el.className='muted';el.textContent='click any tile';return;}
  const f=F[i],k=sel;el.className='';
  let h=`<table><tr><th>tile</th><td>${k} (x=${k%S}, y=${Math.floor(k/S)})</td></tr>`+
        `<tr><th>terrain</th><td>${R.terrain_names[R.terrain[k]]}</td></tr>`+
        `<tr><th>resource</th><td>${R.resource_names[f.resource[k]]}</td></tr>`+
        (f.improvement[k]>=0
          ? `<tr><th>building</th><td>${R.improvement_names[f.improvement[k]]}</td></tr>` : '')+
        `<tr><th>territory</th><td>${f.owner[k]<0?'neutral':'P'+f.owner[k]}</td></tr>`+
        (biomeName(k)
          ? `<tr><th>biome</th><td>${biomeName(k)} \\u2014 the tribe whose spawn `
            + `rates generated this tile (fixed at map generation)</td></tr>` : '');
  if(f.city_level[k]>0)
    h+=`<tr><th>city</th><td>${f.city_owner[k]<0?'neutral':'P'+f.city_owner[k]}`+
       ` \\u00b7 level ${f.city_level[k]} \\u00b7 pop ${f.city_pop[k]}`+
       `${f.city_walls[k]?' \\u00b7 walls':''}${f.city_capital[k]?' \\u00b7 capital':''}</td></tr>`;
  if(f.unit_type[k]>=0){
    const ut=f.unit_type[k];
    if((f.unit_hidden||[])[k])
      h+=`<tr><th>stealth</th><td>hidden — invisible to the enemy; `+
         `their adjacent units only see an eye symbol</td></tr>`;
    const home=f.unit_home[k];
    h+=`<tr><th>unit</th><td>P${f.unit_owner[k]} ${R.unit_names[ut]}`+
       `${home>=0?` \u00b7 from city ${home}`:' \u00b7 unsupported'}`+
       ` \\u00b7 ${f.unit_hp[k]}/${R.unit_max_hp[ut]+(f.unit_veteran[k]?5:0)} hp`+
       `${f.unit_veteran[k]?' \\u00b7 veteran (+5 hp)':''}`+
       `${f.unit_kills[k]?' \\u00b7 '+f.unit_kills[k]+' kills':''}`+
       `${(!f.unit_veteran[k]&&f.unit_kills[k]>=3)
           ? ' \\u00b7 <b>promotion available</b> (free full heal)' : ''}`+
       `<br><span class="muted">${f.unit_moved[k]?'moved':'can move'} \\u00b7 `+
       `${f.unit_attacked[k]?'acted':'can act'}</span></td></tr>`;
  }
  const seen=R.tribes.map((t,p)=>f.explored[p][k]?`P${p}`:null).filter(Boolean);
  h+=`<tr><th>explored by</th><td>${seen.length?seen.join(', '):'nobody'}</td></tr>`;
  el.innerHTML=h+'</table>';
}

function renderLog(){
  const lo=Math.max(0,i-45),hi=Math.min(F.length,i+10);let h='';
  for(let k=lo;k<hi;k++){
    const a=F[k].action;
    h+=`<div class="${k===i?'cur':''}" data-i="${k}">${String(k).padStart(4)} `+
       (a?`<span style="color:${PC[a.player%4]}">P${a.player}</span> ${a.desc}`
         :'<span class="muted">initial state</span>')+`</div>`;
  }
  const el=document.getElementById('log');el.innerHTML=h;
  const c=el.querySelector('.cur');if(c)c.scrollIntoView({block:'nearest'});
  el.querySelectorAll('div').forEach(d=>d.onclick=()=>go(+d.dataset.i));
}

cv.onclick=e=>{const r=cv.getBoundingClientRect();
  const sx=cv.width/r.width, sy=cv.height/r.height;
  const x=Math.floor((e.clientX-r.left)*sx/CELL), y=Math.floor((e.clientY-r.top)*sy/CELL);
  sel=(x>=0&&x<S&&y>=0&&y<S)?y*S+x:-1;draw();};

function go(n){i=Math.max(0,Math.min(F.length-1,n));draw();}
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
    timer=setInterval(()=>{if(i>=F.length-1){toggle();return;}go(i+1);},speed);
    playBtn.innerHTML='&#10074;&#10074; pause';}
}
playBtn.onclick=toggle;
addEventListener('keydown',e=>{
  if(e.key==='ArrowRight')go(i+1);else if(e.key==='ArrowLeft')go(i-1);
  else if(e.key===' '){e.preventDefault();toggle();}
  else if(e.key==='Home')go(0);else if(e.key==='End')go(F.length-1);});
function buildKey(){
  const sw=(bg,txt,fg)=>`<span class="chip" style="background:${bg};color:${fg||'#0c0e13'}">${txt||''}</span>`;
  const rows=[];
  const add=(chip,label)=>rows.push(`<div>${chip}</div><div class="lbl">${label}</div>`);
  const grp=t=>rows.push(`<div class="grp">${t}</div>`);

  grp('players');
  R.tribes.forEach((t,p)=>add(sw(PC[p%4],'P'+p),
    `${t} <span class="muted">(${R.agents[p]||'?'})</span>`));
  add(`<span class="chip" style="background:${PC[0]};opacity:.24"></span>`,
      'faint tint = that player’s territory');

  grp('terrain');
  R.terrain_names.forEach(n=>{ if(TERR[n]) add(sw(TERR[n],''), n); });

  grp('resources (top-left glyph)');
  R.resource_names.forEach(n=>{const v=RES[n]; if(v)
    add(`<span class="chip" style="background:#222a33;color:${v[0]}">${v[1]}</span>`, n);});

  grp('buildings (top-right letter)');
  (R.improvement_names||[]).forEach(n=>{const ch=IMP[n]; if(ch)
    add(`<span class="chip" style="background:#222a33;color:#e9d8a6">${ch}</span>`, n);});
  add('<span class="muted" style="font-size:11px">—</span>',
      '<span class="muted">harvests (fruit/game/fish) consume the resource and leave no building</span>');
  add('<span class="muted" style="font-size:11px">—</span>',
      '<span class="muted">= road, ≡ bridge: halve movement between them and, when they reach '
      + 'the capital, connect a city (+1 population to <em>both</em>). Ports connect too.</span>');
  add('<span class="muted" style="font-size:11px">—</span>',
      '<span class="muted">$ market: stars per turn per adjacent sawmill/windmill/forge</span>');
  add('<span class="muted" style="font-size:11px">—</span>',
      '<span class="muted">\\u2302 ruin: a unit that <em>starts</em> its turn here may examine it '
      + 'for stars, a free tech, population, an explorer or a veteran unit. '
      + '\\u2731 starfish: harvested the same way for 8 stars, once Navigation is in.</span>');

  grp('cities (square, number = level)');
  add(sw(PC[0],'2'), 'level 2 city, owner’s colour');
  add(`<span class="chip" style="background:${PC[0]};box-shadow:0 0 0 2px #fff">1</span>`,
      'white outline = city walls');
  add(`<span class="chip" style="background:${PC[0]};box-shadow:0 0 0 2px #ffd166">1</span>`,
      'gold outline = capital');
  add(`<span class="chip" style="background:${PC[0]};box-shadow:0 0 0 3px #ffd166">3</span>`,
      'thick gold ring = owes a level-up reward choice');
  add(sw('#9aa3b8','1'), 'grey = neutral village (capturable)');

  grp('units (circle, letter = type)');
  add(`<span class="dot" style="background:${PC[0]}"></span>`,
      'first letter of the unit name, e.g. W = warrior');
  add(`<span class="dot" style="background:${PC[0]};border-color:#ffd166;border-width:2px"></span>`,
      'gold ring = veteran');
  add(`<span class="dot" style="background:${PC[0]};opacity:.55"></span>`,
      'dimmed = already moved and acted this turn');
  add('<span class="chip" style="background:linear-gradient(90deg,#5fd68a 60%,#000 60%)"></span>',
      'bar under the unit = health (green &rarr; yellow &rarr; red)');

  grp('last action highlight');
  add('<span class="chip" style="box-shadow:inset 0 0 0 2px #ffd166"></span>', 'source tile');
  add('<span class="chip" style="box-shadow:inset 0 0 0 2px #5fd68a"></span>', 'move / capture target');
  add('<span class="chip" style="box-shadow:inset 0 0 0 2px #ff6b6b"></span>', 'attack target');

  grp('fog (“view as” a player)');
  add('<span class="chip" style="background:rgba(6,8,12,.88)"></span>',
      'darkened = that player has never explored it');

  grp('reading the panels');
  add('<span class="muted" style="font-size:11px">cap</span>',
      'total units the player’s cities can support (sum of level+1)');
  add('<span class="muted" style="font-size:11px">tech</span>',
      'green = researched, outlined = affordable now, faded = prerequisite missing');

  document.getElementById('key').innerHTML = rows.join('');
}
buildKey();
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
