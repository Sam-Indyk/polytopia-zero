/* panels.js — the side panels: who is winning, who is at war with whom, what
 * each tile actually is, and a key that explains every mark on the board.
 *
 * These are shared by both modes. The replay controller and the play controller
 * hand the same (header, frame) pair to the same renderers, which is the only
 * reason a live game and a recorded one look identical.
 */

const el = id => document.getElementById(id);
const esc = s => String(s).replace(/[&<>]/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;'}[c]));
const pretty = s => String(s).replace(/_/g, ' ');
const bit = (mask, i) => ((mask >>> i) & 1) === 1;
// A fog-projected frame sends -1 for anything about an opponent you could not
// work out from your own screen. Printing it as a number would be a lie.
const known = v => (v === undefined || v === null || v < 0)
  ? '<span class="muted" title="not observable from your own screen">?</span>' : v;

function seatChip(H, f, p) {
  const cls = ['seat'];
  if (f.cur === p) cls.push('turn');
  if (!f.alive[p]) cls.push('dead');
  return `<div class="${cls.join(' ')}">` +
    `<span class="pip" style="background:${pcolor(p)}"></span>` +
    `<b>${esc(pretty(H.tribes[p]))}</b>` +
    `<span class="muted">${esc(H.agents && H.agents[p] ? H.agents[p] : 'P' + p)}</span>` +
    `<span class="stars">${known(f.stars[p])}★</span></div>`;
}

function renderSeats(H, f) {
  el('seats').innerHTML =
    H.tribes.map((_, p) => seatChip(H, f, p)).join('');
}

function renderPlayers(H, f) {
  const N = H.size * H.size;
  // "units" is written as used/supported: a city supports level+1 units, and
  // being at the cap is the thing that most often explains a stalled army.
  let h = '<table><tr><th>tribe</th><th class="num">score</th><th class="num">★</th>' +
          '<th class="num">cities</th><th class="num" title="used / supported">units</th>' +
          '<th class="num">tech</th><th class="num" title="kills / losses">k/l</th></tr>';
  const best = Math.max(...(f.score || [0]));
  for (let p = 0; p < H.n_players; p++) {
    let units = 0, cap = 0, cities = 0;
    for (let k = 0; k < N; k++) {
      if (f.unit_owner[k] === p && f.unit_type[k] >= 0) units++;
      if (f.city_owner[k] === p && f.city_level[k] > 0) { cities++; cap += f.city_level[k] + 1; }
    }
    let nt = 0;
    for (let t = 0; t < H.tech_names.length; t++) if (bit(f.tech[p], t)) nt++;
    const sc = f.score ? f.score[p] : 0;
    const lead = sc === best && best > 0;
    h += `<tr><td><span class="sw" style="background:${pcolor(p)}"></span>` +
         `${esc(pretty(H.tribes[p]))}${f.alive[p] ? '' : ' <span class="muted">(out)</span>'}` +
         `${f.cur === p ? ' <span class="stars">◀</span>' : ''}</td>` +
         `<td class="num"${lead ? ' style="color:#ffc94d;font-weight:700"' : ''}>${known(sc)}</td>` +
         `<td class="num stars">${known(f.stars[p])}</td><td class="num">${cities}</td>` +
         `<td class="num"${units >= cap && cap > 0 ? ' style="color:#ffc94d"' : ''}>${units}/${cap}</td>` +
         `<td class="num">${nt}/${H.tech_names.length}</td>` +
         `<td class="num">${known(f.kills && f.kills[p])}/${known(f.losses && f.losses[p])}</td></tr>`;
  }
  el('players').innerHTML = h + '</table>';
}

/** War is a memory, not a state: `attacked_now | attacked_prev` is "at most a
 *  turn ago" on either side. Peace needs both; a lone offer is pending. */
function renderDiplomacy(H, f) {
  if (!f.peace) { el('diplo').innerHTML = '<span class="muted">not recorded</span>'; return; }
  const rows = [];
  for (let p = 0; p < H.n_players; p++) {
    for (let q = p + 1; q < H.n_players; q++) {
      const peace = bit(f.peace[p], q);
      const war = bit(f.attacked_now[p] | f.attacked_prev[p], q) ||
                  bit(f.attacked_now[q] | f.attacked_prev[q], p);
      const offerPQ = bit(f.peace_offer[p], q), offerQP = bit(f.peace_offer[q], p);
      let label = '<span class="chip">neutral</span>';
      if (peace) label = '<span class="chip on">peace treaty</span>';
      else if (war) label = '<span class="chip bad">at war</span>';
      else if (offerPQ || offerQP)
        label = `<span class="chip warn">offer from ${esc(pretty(H.tribes[offerPQ ? p : q]))}</span>`;
      rows.push(`<tr><td><span class="sw" style="background:${pcolor(p)}"></span>` +
        `${esc(pretty(H.tribes[p]))} &middot; <span class="sw" style="background:${pcolor(q)}"></span>` +
        `${esc(pretty(H.tribes[q]))}</td><td>${label}</td></tr>`);
    }
  }
  el('diplo').innerHTML = rows.length
    ? `<table>${rows.join('')}</table>`
    : '<span class="muted">single player</span>';
}

/** Tasks latch: once earned a monument is never taken back, so "earned" and
 *  "built" are two different things and the panel shows both. */
function renderTasks(H, f) {
  if (!f.task_done) { el('tasks').innerHTML = '<span class="muted">not recorded</span>'; return; }
  let h = '<table><tr><th>task</th>' +
    H.tribes.map((t, p) => `<th class="num"><span class="sw" style="background:${pcolor(p)}"></span></th>`).join('') +
    '</tr>';
  H.task_names.forEach((name, t) => {
    h += `<tr><td>${esc(name)}</td>`;
    for (let p = 0; p < H.n_players; p++) {
      const done = bit(f.task_done[p], t), built = bit(f.monument_built[p], t);
      h += `<td class="num">${built ? '<span class="chip on">built</span>'
             : done ? '<span class="chip warn">earned</span>' : '<span class="muted">·</span>'}</td>`;
    }
    h += '</tr>';
  });
  el('tasks').innerHTML = h + '</table>';
}

// --- tile inspector ---------------------------------------------------------

const TEMPLES = ['temple', 'mountain_temple', 'forest_temple', 'water_temple'];

function renderTile(H, f, k, extra) {
  const box = el('tile');
  if (k < 0) { box.className = 'muted'; box.textContent = 'click any tile'; return; }
  box.className = '';
  const S = H.size;
  const terr = H.terrain_names[H.terrain[k]];
  const res = H.resource_names[f.resource[k]];
  const imp = f.improvement[k];
  const impName = imp >= 0 ? H.improvement_names[imp] : null;
  const rows = [];
  const row = (a, b) => rows.push(`<tr><th>${a}</th><td>${b}</td></tr>`);

  row('tile', `${k} <span class="muted">(x ${k % S}, y ${(k / S) | 0})</span>`);
  const rough = H.terrain_rough[H.terrain[k]];
  const def = H.terrain_defense_bonus[H.terrain[k]];
  row('terrain', esc(terr) +
    `<div class="chips"><span class="chip">move ${H.terrain_move_cost[H.terrain[k]]}</span>` +
    (rough ? '<span class="chip warn">rough — enterable, halts movement</span>' : '') +
    (def > 1 ? `<span class="chip">defence ×${def} with ` +
      esc(H.tech_names[H.terrain_defense_tech[H.terrain[k]]] || '?') + '</span>' : '') +
    '</div>');
  if (res !== 'none') {
    const hidden = (H.hidden_resource_tech || {})[res];
    row('resource', esc(res) + (hidden
      ? `<div class="chips"><span class="chip warn">hidden until ` +
        esc(hidden.map(pretty).join(' or ')) + '</span></div>' : ''));
  }
  if (impName) {
    const isTemple = TEMPLES.includes(impName);
    const built = f.improvement_turn ? f.improvement_turn[k] : -1;
    let note = '';
    if (isTemple && built >= 0) {
      // 100 points at construction, +50 every three turns, capped at 300 —
      // "including the turn of their construction" is why it is floor(/3) + 1.
      const lvl = Math.min(5, 1 + Math.floor((f.turn - built) / 3));
      note = `<div class="chips"><span class="chip on">level ${lvl} · ` +
             `${100 + 50 * (lvl - 1)} points</span>` +
             `<span class="chip">built turn ${built}</span></div>`;
    } else if (built >= 0) {
      note = `<div class="chips"><span class="chip">built turn ${built}</span></div>`;
    }
    const owner = f.improvement_owner ? f.improvement_owner[k] : -1;
    row('building', esc(pretty(impName)) +
      (owner >= 0 ? ` <span class="muted">(${esc(pretty(H.tribes[owner]))})</span>` : '') + note);
  }
  row('territory', f.owner[k] < 0 ? '<span class="muted">neutral</span>'
    : `<span class="sw" style="background:${pcolor(f.owner[k])}"></span>${esc(pretty(H.tribes[f.owner[k]]))}`);
  const biome = H.tile_tribe && H.tribe_names ? H.tribe_names[H.tile_tribe[k]] : null;
  if (biome)
    row('biome', esc(pretty(biome)) +
      ' <span class="muted">— whose spawn rates made this tile, fixed at generation</span>');

  if (f.city_level[k] > 0) {
    const c = [];
    if (f.city_capital[k]) c.push('<span class="chip on">capital</span>');
    if (f.city_walls[k]) c.push('<span class="chip">walls ×4</span>');
    if (f.city_workshop[k]) c.push('<span class="chip">workshop +1★</span>');
    if (f.city_park[k]) c.push('<span class="chip">park 250pt</span>');
    if (f.city_conn_pop && f.city_conn_pop[k]) c.push('<span class="chip">connected</span>');
    if (f.city_pending_reward[k] > 0) c.push('<span class="chip warn">owes a level-up choice</span>');
    row('city', (f.city_owner[k] < 0 ? '<span class="muted">neutral village</span>'
        : esc(pretty(H.tribes[f.city_owner[k]]))) +
      ` &middot; level ${f.city_level[k]} &middot; pop ${f.city_pop[k]}` +
      ` &middot; supports ${f.city_level[k] + 1}` +
      `<div class="chips">${c.join('')}</div>`);
  }

  if (f.unit_type[k] >= 0) {
    const t = f.unit_type[k];
    const maxHp = H.unit_max_hp[t] + (f.unit_veteran[k] ? 5 : 0);
    const chips = (H.unit_skills[t] || []).map(s => `<span class="chip">${esc(s)}</span>`);
    if (f.unit_veteran[k]) chips.unshift('<span class="chip on">veteran +5 hp</span>');
    else if (f.unit_kills[k] >= 3)
      chips.unshift('<span class="chip warn">promotion banked — free full heal, costs no action</span>');
    if ((f.unit_hidden || [])[k])
      chips.unshift('<span class="chip warn">hidden — adjacent enemies see only an eye</span>');
    if (f.unit_carried[k] >= 0)
      chips.push(`<span class="chip">carrying ${esc(H.unit_names[f.unit_carried[k]])}</span>`);
    row('unit',
      `<span class="sw" style="background:${pcolor(f.unit_owner[k])}"></span>` +
      `${esc(pretty(H.unit_names[t]))} &middot; ${f.unit_hp[k]}/${maxHp} hp` +
      ` &middot; atk ${H.unit_attack[t]} def ${H.unit_defense[t]}` +
      ` &middot; move ${H.unit_movement[t]}${H.unit_range[t] > 1 ? ' &middot; range ' + H.unit_range[t] : ''}` +
      `<br><span class="muted">${f.unit_moved[k] ? 'has moved' : 'can move'} &middot; ` +
      `${f.unit_attacked[k] ? 'has acted' : 'can act'} &middot; ` +
      `${f.unit_home[k] >= 0 ? 'supported by city ' + f.unit_home[k] : 'no support slot'} &middot; ` +
      `${f.unit_kills[k]} kills</span>` +
      `<div class="chips">${chips.join('')}</div>`);
  }

  // In play mode the frame carries a single explored row — your own — because
  // whether the opponent has seen a tile is not something you can know.
  if (f.explored && f.explored.length === H.n_players) {
    const seen = [];
    for (let p = 0; p < H.n_players; p++) if (f.explored[p][k]) seen.push(pretty(H.tribes[p]));
    row('explored by', seen.length ? esc(seen.join(', ')) : '<span class="muted">nobody</span>');
  }
  if (extra) rows.push(extra);
  box.innerHTML = `<table>${rows.join('')}</table>`;
}

// --- action log -------------------------------------------------------------

function renderLog(H, frames, i, onGo) {
  const lo = Math.max(0, i - 60), hi = Math.min(frames.length, i + 14);
  let h = '';
  for (let k = lo; k < hi; k++) {
    const a = frames[k].action;
    h += `<div class="${k === i ? 'cur' : ''}" data-i="${k}">` +
      `<span class="t">${String(k).padStart(4, ' ')}</span> ` +
      (a ? `<span style="color:${pcolor(a.player)}">${esc(pretty(H.tribes[a.player]))}</span> ${esc(a.desc)}`
         : '<span class="muted">initial position</span>') + '</div>';
  }
  const box = el('log');
  box.innerHTML = h;
  const cur = box.querySelector('.cur');
  if (cur) cur.scrollIntoView({block: 'nearest'});
  if (onGo) box.querySelectorAll('div').forEach(d => d.onclick = () => onGo(+d.dataset.i));
}

function installTabs(onSwitch) {
  const tabs = el('tabs');
  tabs.querySelectorAll('button').forEach(b => b.onclick = () => {
    tabs.querySelectorAll('button').forEach(x => x.classList.toggle('on', x === b));
    document.querySelectorAll('.pane').forEach(
      p => p.classList.toggle('on', p.dataset.pane === b.dataset.tab));
    document.body.classList.toggle('techwide', b.dataset.tab === 'tech');
    if (onSwitch) onSwitch(b.dataset.tab);
  });
  const want = /[#&]tab=(\w+)/.exec(location.hash);
  if (want) showTab(want[1]);
}

function showTab(name) {
  const b = el('tabs').querySelector(`[data-tab="${name}"]`);
  if (b) b.click();
}

/** The eye an adjacent enemy shows when a hidden unit is beside it. Flagged on
 *  the *watcher*, never on the Cloak — the player triangulates from three
 *  neighbouring units each showing an eye, which is how the game plays it. */
function senseMap(H, f) {
  const S = H.size, N = S * S, out = {};
  for (let k = 0; k < N; k++) {
    if (f.unit_type[k] < 0) continue;
    const me = f.unit_owner[k], x = k % S, y = (k / S) | 0;
    for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
      if (!dx && !dy) continue;
      const nx = x + dx, ny = y + dy;
      if (nx < 0 || ny < 0 || nx >= S || ny >= S) continue;
      const n = ny * S + nx;
      if (f.unit_type[n] >= 0 && (f.unit_hidden || [])[n] && f.unit_owner[n] !== me)
        out[k] = true;
    }
  }
  return out;
}

// --- key --------------------------------------------------------------------

/** Each key entry is a real 44×34 render from the same sprite functions the
 *  board uses, so the key cannot drift from what you are looking at. */
function keySwatch(paint) {
  const c = document.createElement('canvas');
  c.width = 132; c.height = 96;          // 2x backing store for a 66×48 box
  const g = c.getContext('2d');
  g.scale(2, 2);
  g.fillStyle = '#0f1520'; g.fillRect(0, 0, 66, 48);
  paint(g, 33, 34, 0.42);
  return c;
}

function buildKey(H, frame) {
  const box = el('key');
  box.replaceChildren();
  const grp = t => {
    const d = document.createElement('div');
    d.className = 'grp'; d.textContent = t; box.appendChild(d);
  };
  const add = (node, label) => {
    const a = document.createElement('div');
    if (typeof node === 'string') a.innerHTML = node; else a.appendChild(node);
    const b = document.createElement('div');
    b.className = 'lbl'; b.innerHTML = label;
    box.appendChild(a); box.appendChild(b);
  };
  const fake = over => Object.assign({
    owner: [-1], resource: [0], improvement: [-1], improvement_owner: [-1],
    city_owner: [0], city_level: [0], city_pop: [0], city_walls: [0],
    city_capital: [0], city_workshop: [0], city_park: [0], city_pending_reward: [0],
    unit_type: [0], unit_owner: [0], unit_hp: [10], unit_moved: [0],
    unit_attacked: [0], unit_veteran: [0], unit_kills: [0], unit_home: [0],
    unit_carried: [-1], unit_hidden: [0],
  }, over);

  grp('tribes');
  H.tribes.forEach((t, p) => add(
    `<span class="sw" style="background:${pcolor(p)};width:14px;height:14px"></span>`,
    `${esc(pretty(t))} <span class="muted">${esc((H.agents || [])[p] || '')}</span>`));
  add('<span class="muted">—</span>',
    'Territory is a <b>coloured rope along tile edges</b>, not a tint — ' +
    'drawn only where the neighbouring tile has a different owner.');

  grp('terrain');
  add('<span class="muted">—</span>',
    'Blocks, not squares. Water sits <b>lower</b> than land, so every coast has ' +
    'a visible lip; mountains rise above their tile and occlude what is behind.');
  add(keySwatch((g, x, y, z) => drawMountain(g, x, y + 4, z * 2)),
    'mountain — needs Climbing to enter, gives sight 2, rough');
  add(keySwatch((g, x, y, z) => drawForest(g, x, y + 4, z * 2, 3)),
    'forest — rough: enterable, but halts movement');

  grp('resources');
  ['fruit', 'crop', 'game', 'metal', 'fish', 'starfish'].forEach(r => {
    if (!H.resource_names.includes(r)) return;
    const hid = (H.hidden_resource_tech || {})[r];
    add(keySwatch((g, x, y, z) => drawResource(g, x, y + 6, z * 2, r, 5)),
      esc(r) + (hid ? ` <span class="muted">— hidden until ${esc(hid.map(pretty).join(' or '))}</span>` : ''));
  });

  grp('cities');
  add(keySwatch((g, x, y, z) => drawCity(g, x, y, z * 1.2,
      fake({city_level: [3]}), 0)),
    'one house per level, up to five; the badge on the left corner is the level');
  add(keySwatch((g, x, y, z) => drawCity(g, x, y, z * 1.2,
      fake({city_level: [2], city_capital: [1]}), 0)), 'pennant = capital');
  add(keySwatch((g, x, y, z) => drawCity(g, x, y, z * 1.2,
      fake({city_level: [2], city_walls: [1]}), 0)),
    'stone ring = walls (×4 defence, and only for the owner’s own fortifying units)');
  add(keySwatch((g, x, y, z) => drawCity(g, x, y, z * 1.2,
      fake({city_level: [2], city_pending_reward: [1]}), 0)),
    'gold ring on the badge = owes a level-up reward choice');

  grp('units');
  add(keySwatch((g, x, y, z) => drawUnit(g, x, y + 4, z * 1.6,
      fake({unit_type: [0], unit_hp: [4]}), 0, H, {})),
    'a health bar appears only when a unit is <b>damaged</b> — an undamaged army ' +
    'would otherwise be a field of bright rectangles');
  add(keySwatch((g, x, y, z) => drawUnit(g, x, y + 4, z * 1.6,
      fake({unit_type: [0], unit_veteran: [1]}), 0, H, {})),
    'gold star = veteran (+5 max HP)');
  add(keySwatch((g, x, y, z) => drawUnit(g, x, y + 4, z * 1.6,
      fake({unit_type: [0], unit_kills: [3]}), 0, H, {})),
    'faded star = <b>promotion banked</b> — three kills, not yet cashed in. ' +
    'Promoting is free and heals completely, so holding it until hurt is the play.');
  add(keySwatch((g, x, y, z) => drawUnit(g, x, y + 4, z * 1.6,
      fake({unit_type: [0], unit_moved: [1], unit_attacked: [1]}), 0, H, {})),
    'dimmed = has moved and acted');
  const cloak = H.unit_names.indexOf('cloak');
  if (cloak >= 0) add(keySwatch((g, x, y, z) => drawUnit(g, x, y + 4, z * 1.6,
      fake({unit_type: [cloak], unit_hidden: [1]}), 0, H, {})),
    'ghosted with a dashed ring = a <b>hidden Cloak</b>. The god’s-eye view shows ' +
    'it because nothing is hidden from you; the enemy sees only an eye on its own units.');

  grp('markers');
  add('<span class="chip" style="border-color:#ffc94d">gold</span>', 'the acting unit / selected tile');
  add('<span class="chip" style="border-color:#7fe3a0">green</span>', 'move or capture destination');
  add('<span class="chip" style="border-color:#ff6b6b">red</span>', 'attack target');
  add('<span class="chip">dashed arc</span>', 'the last action, from source to destination');

  grp('fog');
  add('<span class="chip" style="background:#141a26">dark</span>',
    'with “view as” set to a tribe, everything that tribe has never explored. ' +
    'This is the same memory the observation encoder reads — invariant 1.');
}
