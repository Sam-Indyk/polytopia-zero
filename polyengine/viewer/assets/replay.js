/* replay.js — playback controller for a recorded game.
 *
 * The whole replay is embedded in the page, so scrubbing is instant and the
 * file works from a filesystem with no server behind it. Everything here is
 * presentation; the frames are ground truth as the engine produced them.
 */

const H = BOOT, F = BOOT.frames;
let i = 0, timer = null, speed = 90, sel = -1, fogOf = -1;

const board = new Board(el('cv'), H);
const tech = new TechTree(el('techtree'), H, el('techinfo'));

el('meta').textContent =
  `${H.size}×${H.size} ${H.map_type} · seed ${H.map_seed} · ` +
  `${F.length} frames · rules ${H.ruleset_version} · ` +
  (H.terminal
    ? (H.winner >= 0 ? `winner: ${pretty(H.tribes[H.winner])}` : 'draw')
    : 'unfinished');

el('missing').innerHTML = (H.not_simulated || []).map(m => `<li>${esc(m)}</li>`).join('');

el('transport').innerHTML =
  '<button id="first" title="Home">|◀</button>' +
  '<button id="prev" title="←">◀</button>' +
  '<button id="play" class="go" title="space">▶ play</button>' +
  '<button id="next" title="→">▶</button>' +
  '<button id="last" title="End">▶|</button>' +
  '<input type="range" id="scrub" min="0" value="0">' +
  '<span class="muted" id="pos"></span>';

el('barextra').innerHTML =
  '<label class="muted">speed</label>' +
  '<select id="speed"><option value="260">slow</option>' +
  '<option value="90" selected>normal</option>' +
  '<option value="24">fast</option>' +
  '<option value="6">blur</option></select>' +
  '<button id="jumpturn">next turn ⇥</button>';

el('hint').textContent = '← → step · space play · drag to pan · wheel to zoom · click a tile';

H.tribes.forEach((t, p) => {
  const o = document.createElement('option');
  o.value = p; o.textContent = `${pretty(t)} — as they saw it`;
  el('fog').appendChild(o);
  const o2 = document.createElement('option');
  o2.value = p; o2.textContent = pretty(t);
  el('techplayer').appendChild(o2);
});

function frame() { return F[i]; }

function draw() {
  const f = frame();
  board.set(f, {
    fogOf, sel,
    lastAction: f.action,
    senseAt: senseMap(H, f),
  });
  renderSeats(H, f);
  renderPlayers(H, f);
  renderDiplomacy(H, f);
  renderTasks(H, f);
  renderTile(H, f, sel);
  renderLog(H, F, i, go);
  tech.render(f, +el('techplayer').value);
  el('pos').textContent = `frame ${i}/${F.length - 1} · turn ${f.turn} · ` +
    `${pretty(H.tribes[f.cur])} to move`;
  el('scrub').value = i;
  const done = H.terminal && i === F.length - 1;
  el('banner').classList.toggle('hidden', !done);
  if (done) {
    el('banner').innerHTML = H.winner >= 0
      ? `${esc(pretty(H.tribes[H.winner]))} wins<small>` +
        (f.turn >= H.turn_cap ? `turn cap — decided on score` : 'domination') + '</small>'
      : 'draw<small>no tribe left standing</small>';
  }
}

function go(n) {
  i = Math.max(0, Math.min(F.length - 1, n));
  // The frame index lives in the URL hash, so a position in a 1,800-frame game
  // is a link you can paste at someone rather than "scrub to about a third in".
  history.replaceState(null, '', '#f=' + i);
  draw();
}

function toggle() {
  const b = el('play');
  if (timer) { clearInterval(timer); timer = null; b.innerHTML = '▶ play'; b.className = 'go'; }
  else {
    if (i >= F.length - 1) i = 0;
    timer = setInterval(() => { if (i >= F.length - 1) { toggle(); return; } go(i + 1); }, speed);
    b.innerHTML = '❚❚ pause'; b.className = '';
  }
}

el('first').onclick = () => go(0);
el('prev').onclick = () => go(i - 1);
el('next').onclick = () => go(i + 1);
el('last').onclick = () => go(F.length - 1);
el('play').onclick = toggle;
el('scrub').max = F.length - 1;
el('scrub').oninput = e => go(+e.target.value);
el('speed').onchange = e => { speed = +e.target.value; if (timer) { toggle(); toggle(); } };
el('jumpturn').onclick = () => {
  const t = frame().turn;
  let n = i;
  while (n < F.length - 1 && F[n].turn === t) n++;
  go(n);
};
el('fog').onchange = e => { fogOf = +e.target.value; draw(); };
el('techplayer').onchange = draw;
el('zin').onclick = () => board.zoom(1.25);
el('zout').onclick = () => board.zoom(1 / 1.25);
el('zfit').onclick = () => { board.fit(); board.draw(); };

board.onPick = k => { sel = k; draw(); showTab('tile'); };
installTabs(() => draw());

addEventListener('keydown', e => {
  if (e.target.tagName === 'SELECT' || e.target.tagName === 'INPUT') return;
  if (e.key === 'ArrowRight') go(i + 1);
  else if (e.key === 'ArrowLeft') go(i - 1);
  else if (e.key === ' ') { e.preventDefault(); toggle(); }
  else if (e.key === 'Home') go(0);
  else if (e.key === 'End') go(F.length - 1);
});

buildKey(H, F[0]);
board.fit();
const anchor = /[#&]f=(\d+)/.exec(location.hash);
go(anchor ? +anchor[1] : 0);
