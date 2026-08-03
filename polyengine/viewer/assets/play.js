/* play.js — play a game yourself, against the network or a baseline bot.
 *
 * The controlling idea: **the UI never invents a move.** Every button on this
 * page is one entry from `enumerate_actions`, rendered. There is no client-side
 * notion of what a Rider may do, no reimplementation of zone of control, no
 * "can I build here" — the server sends the legal action list and this file
 * groups it by source tile and draws it. That is invariant 2 turned into a user
 * interface: if the mask and the enumerator ever disagree, you will see it here
 * as a button that does not work, rather than as a silent capability ceiling.
 *
 * The consequence is that new engine actions need no work here. Tile abilities,
 * promotion, Heal Others and the diplomacy pair all appeared in this panel the
 * moment the engine enumerated them.
 */

const H = BOOT;
let ST = null;                 // current server state
let sel = -1;                  // selected tile
let pending = null;            // an action awaiting a target click
let busy = false;
let godsEye = false;

const board = new Board(el('cv'), H);
const tech = new TechTree(el('techtree'), H, el('techinfo'));

el('meta').textContent =
  `${H.size}×${H.size} ${H.map_type} · seed ${H.map_seed} · ` +
  `you are ${pretty(H.tribes[H.you])} · opponent ${H.opponent}`;
el('missing').innerHTML = (H.not_simulated || []).map(m => `<li>${esc(m)}</li>`).join('');
el('hint').textContent =
  'click a unit then a marked tile to move · click a city to train and build · ' +
  'drag to pan · wheel to zoom';

el('transport').innerHTML =
  '<button id="end" class="go">end turn ⏎</button>' +
  '<button id="undoall">restart game</button>' +
  '<span class="muted" id="pos"></span>';

el('barextra').innerHTML =
  '<label class="muted"><input type="checkbox" id="god"> reveal everything</label>';

// The fog selector is meaningless in play mode: the server already sends a
// frame projected through your own memory, so there is nothing to select.
el('fog').parentElement.style.display = 'none';

H.tribes.forEach((t, p) => {
  const o = document.createElement('option');
  o.value = p; o.textContent = pretty(t);
  el('techplayer').appendChild(o);
});
el('techplayer').value = H.you;

// --- server ---------------------------------------------------------------

async function call(path, body) {
  const r = await fetch(path, body === undefined ? {} : {
    method: 'POST',
    headers: {'content-type': 'application/json'},
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

function toast(msg, bad) {
  const t = el('toast');
  t.textContent = msg;
  t.className = 'toast' + (bad ? ' err' : '');
  clearTimeout(toast._t);
  toast._t = setTimeout(() => t.classList.add('hidden'), 3200);
}

async function guard(fn) {
  if (busy) return;
  busy = true;
  document.body.style.cursor = 'progress';
  try { await fn(); }
  catch (e) { toast(String(e.message || e), true); }
  finally { busy = false; document.body.style.cursor = ''; }
}

async function refresh() { ST = await call('/api/state'); render(); }

/** Apply one legal action by its index in the server's list. */
async function act(idx) {
  pending = null;
  ST = await call('/api/act', {index: idx});
  render();
  // Ending the turn hands over; let the opponent play it out and animate back.
  if (ST.cur !== H.you && !ST.terminal) await runOpponent();
}

async function runOpponent() {
  const res = await call('/api/ai', {});
  // Step through the opponent's moves rather than snapping to the end state —
  // watching where its units went is most of the point of playing it.
  for (const step of res.steps) {
    ST = step;
    render();
    await new Promise(r => setTimeout(r, 130));
  }
  ST = res.state;
  render();
}

// --- action grouping -------------------------------------------------------

const NEEDS_TARGET = new Set(['MOVE', 'ATTACK']);
// Action types whose subtype is a player, not a unit / tech / improvement.
const DIPLOMATIC = new Set(['PEACE_OFFER', 'PEACE_BREAK']);

function actionsAt(tile) {
  return (ST.legal || []).filter(a => a.src === tile);
}

function globalActions() {
  return (ST.legal || []).filter(a => a.src < 0 || a.name === 'RESEARCH' ||
                                       DIPLOMATIC.has(a.name));
}

/** Human-readable label for one legal action. */
function label(a) {
  switch (a.name) {
    case 'MOVE':      return {lbl: 'Move', sub: `to tile ${a.dst}`};
    case 'ATTACK':    return {lbl: 'Attack', sub: `tile ${a.dst}`};
    case 'CAPTURE':   return {lbl: 'Capture', sub: 'ends this unit’s turn'};
    case 'EXAMINE':   return {lbl: 'Examine', sub: 'ruin or starfish — ends the turn'};
    case 'PROMOTE':   return {lbl: 'Promote to veteran',
                              sub: '+5 max HP and a full heal — costs no action'};
    case 'HEAL_OTHERS': return {lbl: 'Heal others', sub: '+4 HP to every adjacent friendly'};
    case 'DISBAND':   return {lbl: 'Disband', sub: 'refunds half the unit’s cost'};
    case 'UPGRADE':   return {lbl: 'Upgrade hull', sub: pretty(H.unit_names[a.subtype] || '')};
    case 'TRAIN':     return {lbl: `Train ${pretty(H.unit_names[a.subtype])}`,
                              sub: `${H.unit_cost[a.subtype]}★ · ` +
                                   `${H.unit_max_hp[a.subtype]} hp · ` +
                                   `atk ${H.unit_attack[a.subtype]}`};
    case 'BUILD':     return {lbl: `Build ${pretty(H.improvement_names[a.subtype])}`,
                              sub: `${H.imp_cost[a.subtype]}★ · ` +
                                   `+${H.imp_population[a.subtype]} population`};
    case 'REWARD':    return {lbl: `Take ${pretty(H.reward_names[a.subtype])}`,
                              sub: 'city level-up reward'};
    case 'TILE_ABILITY': return {lbl: pretty(H.tile_ability_names[a.subtype]),
                                 sub: 'tile ability'};
    case 'RESEARCH':  return {lbl: `Research ${pretty(H.tech_names[a.subtype])}`, sub: ''};
    case 'PEACE_OFFER': return {lbl: `Offer peace to ${pretty(H.tribes[a.subtype])}`,
                                sub: 'offering to someone who already offered concludes it'};
    case 'PEACE_BREAK': return {lbl: `Break peace with ${pretty(H.tribes[a.subtype])}`,
                                sub: 'freezes all your units this turn; ' +
                                     'your units inside their borders are destroyed'};
    case 'END_TURN':  return {lbl: 'End turn', sub: ''};
    default:          return {lbl: a.desc, sub: ''};
  }
}

/** What the selected tile can do, drawn over the board.
 *
 * Every entry is one legal action straight from the engine. Move and Attack are
 * summarised rather than listed one row per destination — you pick those by
 * clicking the highlighted tile, and a Rider on a road can have twenty of them.
 */
function renderTileActions() {
  const bar = el('acts');
  const acts = sel >= 0 && ST.cur === H.you ? actionsAt(sel) : [];
  const moves = acts.filter(a => a.name === 'MOVE');
  const hits = acts.filter(a => a.name === 'ATTACK');
  const rest = acts.filter(a => !NEEDS_TARGET.has(a.name));
  if (!acts.length) { bar.classList.add('hidden'); return; }
  bar.classList.remove('hidden');

  const f = ST.frame;
  let what = `tile ${sel}`;
  if (f.city_level[sel] > 0)
    what = `${f.city_capital[sel] ? 'capital' : 'city'} · level ${f.city_level[sel]}`;
  else if (f.unit_type[sel] >= 0) what = pretty(H.unit_names[f.unit_type[sel]]);

  let h = `<h3><span>${esc(what)}</span>` +
          `<button id="deselect" title="Esc">clear</button></h3>`;
  if (moves.length) h += `<p class="note">${moves.length} move` +
    `${moves.length > 1 ? 's' : ''} marked in white — click one</p>`;
  if (hits.length) h += `<p class="note">${hits.length} target` +
    `${hits.length > 1 ? 's' : ''} marked in red — click to attack</p>`;
  h += '<div class="acts">';
  for (const a of rest) {
    const {lbl, sub} = label(a);
    h += `<button class="act" data-i="${a.i}">` +
         `<span><span class="lbl">${esc(lbl)}</span>` +
         (sub ? `<br><span class="sub">${esc(sub)}</span>` : '') + '</span></button>';
  }
  // A city with no Train option is the one case where showing only legal
  // actions is actively confusing — the panel goes quiet and the game looks as
  // though it has no way to build an army. State the observable facts (support
  // used, whether the tile is occupied, stars) rather than restating the rules,
  // which would be a second copy of the engine waiting to drift.
  if (f.city_level[sel] > 0 && f.city_owner[sel] === H.you &&
      !rest.some(a => a.name === 'TRAIN')) {
    let used = 0;
    for (let k = 0; k < H.size * H.size; k++)
      if (f.unit_home[k] === sel && f.unit_type[k] >= 0) used++;
    h += '<p class="note">Nothing can be trained here right now. ' +
         `Supports <b>${used}/${f.city_level[sel] + 1}</b> units` +
         (f.unit_type[sel] >= 0
           ? ' · a unit is standing on the city, and a new one would spawn there'
           : '') +
         ` · you have ${f.stars[H.you]}★.</p>`;
  }
  if (!rest.length && !moves.length && !hits.length)
    h += '<p class="note">nothing here can act this turn</p>';
  bar.innerHTML = h + '</div>';
  bar.querySelectorAll('.act').forEach(b =>
    b.onclick = () => guard(() => act(+b.dataset.i)));
  el('deselect').onclick = () => { sel = -1; render(); };
}

/** Anything not tied to a tile you have selected: research, diplomacy, and the
 *  level-up choices a city is owed (which must not be easy to miss). */
function renderHud() {
  const hud = el('hud');
  if (!ST || ST.cur !== H.you) {
    hud.innerHTML = ST && ST.terminal ? '' :
      '<div class="box muted">opponent is thinking…</div>';
    return;
  }
  const rewards = (ST.legal || []).filter(a => a.name === 'REWARD');
  const diplo = (ST.legal || []).filter(a => DIPLOMATIC.has(a.name));
  let h = `<div class="box"><span class="stars">${ST.frame.stars[H.you]}★</span> ` +
          `<span class="muted">turn ${ST.frame.turn} of ${H.turn_cap}</span></div>`;
  if (sel < 0) {
    // Without this the board looks inert until you happen to click the right
    // thing — which is exactly how "there is no way to train units" happens.
    const acting = (ST.legal || []).filter(a => a.src >= 0);
    const tiles = new Set(acting.map(a => a.src));
    h += `<div class="box muted">click one of your ${tiles.size} highlighted ` +
         'tiles — units move, cities train and build</div>';
  }
  if (rewards.length) {
    const cities = [...new Set(rewards.map(a => a.src))];
    h += `<div class="box"><b>${cities.length} city level-up</b> ` +
         '<span class="muted">— click the city with the gold ring</span></div>';
  }
  if (diplo.length) {
    h += '<div class="box acts">' + diplo.map(a => {
      const {lbl} = label(a);
      return `<button class="act" data-i="${a.i}"><span class="lbl">${esc(lbl)}</span></button>`;
    }).join('') + '</div>';
  }
  hud.innerHTML = h;
  hud.querySelectorAll('.act').forEach(b =>
    b.onclick = () => guard(() => act(+b.dataset.i)));
}

// --- rendering -------------------------------------------------------------

function render() {
  if (!ST) return;
  const f = ST.frame;
  const marks = {};
  if (ST.cur === H.you && !ST.terminal) {
    if (sel >= 0) {
      for (const a of actionsAt(sel)) {
        if (a.name === 'MOVE') marks[a.dst] = 'move';
        else if (a.name === 'ATTACK') marks[a.dst] = 'attack';
      }
      if (Object.keys(marks).length) marks[sel] = 'src';
    } else {
      // Nothing selected: outline every tile something can act from, so the
      // board tells you where to click instead of waiting to be guessed at.
      for (const a of ST.legal) if (a.src >= 0) marks[a.src] = 'ready';
    }
  }
  board.set(f, {
    sel, marks,
    fogOf: -1,                 // the frame is already projected; nothing to dim
    lastAction: ST.last_action,
    senseAt: senseMap(H, f),
  });
  renderSeats(H, f);
  renderPlayers(H, f);
  renderDiplomacy(H, f);
  renderTasks(H, f);
  renderTile(H, f, sel);
  renderTileActions();
  renderHud();
  renderLog(H, ST.log, ST.log.length - 1, null);
  tech.render(f, +el('techplayer').value, {});
  el('pos').textContent = ST.terminal ? 'game over'
    : `turn ${f.turn} · ${ST.cur === H.you ? 'your move' : pretty(H.tribes[ST.cur]) + ' to move'}`;
  el('end').disabled = ST.cur !== H.you || ST.terminal;

  const over = ST.terminal;
  el('banner').classList.toggle('hidden', !over);
  if (over) {
    const won = ST.winner === H.you;
    el('banner').innerHTML = (ST.winner < 0 ? 'Draw'
      : won ? 'You win' : `${esc(pretty(H.tribes[ST.winner]))} wins`) +
      `<small>${f.turn >= H.turn_cap ? 'turn cap — decided on score' : 'domination'}` +
      ' · press restart for a new game</small>';
  }
}

// --- wiring ----------------------------------------------------------------

board.onPick = k => {
  if (busy) return;
  // Clicking a highlighted destination performs the move; clicking anything
  // else just selects it. One click to pick a unit, one to send it.
  if (sel >= 0 && ST && ST.cur === H.you) {
    const hit = actionsAt(sel).find(
      a => a.dst === k && (a.name === 'MOVE' || a.name === 'ATTACK'));
    if (hit) { guard(() => act(hit.i)); return; }
  }
  sel = k;
  render();
};

tech.onPick = (t, cost, can) => {
  const a = (ST.legal || []).find(x => x.name === 'RESEARCH' && x.subtype === t);
  if (!a) { toast(`cannot research ${pretty(H.tech_names[t])} yet (${cost}★)`, true); return; }
  guard(() => act(a.i));
};

el('end').onclick = () => guard(async () => {
  const a = (ST.legal || []).find(x => x.name === 'END_TURN');
  if (a) await act(a.i);
});
el('undoall').onclick = () => guard(async () => {
  if (!confirm('Restart with a fresh map?')) return;
  await call('/api/new', {});
  // The terrain lives in the page's boot header, so a new map means a new page.
  location.reload();
});
el('god').onchange = e => {
  godsEye = e.target.checked;
  guard(async () => { await call('/api/reveal', {on: godsEye}); await refresh(); });
};
el('techplayer').onchange = render;
el('zin').onclick = () => board.zoom(1.25);
el('zout').onclick = () => board.zoom(1 / 1.25);
el('zfit').onclick = () => { board.fit(); board.draw(); };
installTabs(() => render());

addEventListener('keydown', e => {
  if (e.target.tagName === 'SELECT' || e.target.tagName === 'INPUT') return;
  if (e.key === 'Enter') el('end').click();
  else if (e.key === 'Escape') { sel = -1; render(); }
});

buildKey(H, {});
board.fit();
// `#sel=47` preselects a tile, which is how you hand someone a position to look
// at — and the only way to drive the board from a headless browser.
const want = /[#&]sel=(\d+)/.exec(location.hash);
if (want) sel = +want[1];
guard(refresh);
