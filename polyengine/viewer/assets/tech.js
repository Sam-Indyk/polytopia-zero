/* tech.js — the technology tree, laid out the way the game lays it out.
 *
 * Polytopia does not draw a left-to-right dependency graph; it draws a wheel.
 * The tribe sits at the hub, five branches radiate out, and each fork pushes
 * its children further from the centre. That shape is the reason the real tree
 * is readable at a glance — you learn where Sailing *is*, spatially, and stop
 * reading labels.
 *
 * The layout is computed from `tech_requires`, not hardcoded to today's 5/10/10
 * shape: each root gets an equal slice of the circle and every node subdivides
 * its own slice among its children. If the ruleset ever grows a branch, the
 * wheel just gets denser rather than wrong.
 */

const TECH_GLYPH = {
  climbing: '⛰', fishing: '🐟', hunting: '🦌',
  organization: '🍎', riding: '🐎', archery: '🏹',
  farming: '🌾', forestry: '🪓', free_spirit: '🕊',
  meditation: '☯', mining: '⛏', ramming: '🛠',
  roads: '🛣', sailing: '⛵', strategy: '♟',
  aquatism: '🌊', chivalry: '🛡', construction: '🏗',
  diplomacy: '🤝', mathematics: '⚙', navigation: '🧭',
  philosophy: '📜', smithery: '🔨', spiritualism: '🌳',
  trade: '💰',
};

const SVGNS = 'http://www.w3.org/2000/svg';
const svgEl = (tag, attrs) => {
  const e = document.createElementNS(SVGNS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  return e;
};

const hasTech = (mask, t) => ((mask >>> t) & 1) === 1;

function countCities(f, p, N) {
  let c = 0;
  for (let k = 0; k < N; k++) if (f.city_owner[k] === p && f.city_level[k] > 0) c++;
  return c;
}

/** The live price, matching `actions.research_cost`: tier * cities + 4, then
 *  Philosophy's Literacy discount of a third, rounded up. */
function researchCost(H, f, p, t) {
  const base = H.tech_tier[t] * countCities(f, p, H.size * H.size) + 4;
  const phil = H.tech_names.indexOf('philosophy');
  return (phil >= 0 && hasTech(f.tech[p], phil)) ? Math.ceil(base * 2 / 3) : base;
}

/** Everything a technology gives you, read out of the ruleset tables rather
 *  than a hand-written list — so it cannot drift from the data. */
function techUnlocks(H, t) {
  const out = [];
  H.unit_names.forEach((n, i) => {
    if (H.unit_requires_tech[i] === t) out.push({kind: 'unit', name: n});
  });
  H.improvement_names.forEach((n, i) => {
    if (H.imp_requires_tech[i] === t && !H.imp_map_feature[i])
      out.push({kind: H.imp_kind[i] === 'harvest' ? 'harvest' : 'building', name: n});
  });
  H.terrain_names.forEach((n, i) => {
    if (H.terrain_requires_tech && H.terrain_requires_tech[i] === t)
      out.push({kind: 'movement', name: `enter ${n}`});
    if (H.terrain_defense_tech[i] === t)
      out.push({kind: 'defence', name: `${n} defence`});
  });
  for (const [ab, tech] of Object.entries(H.tile_ability_tech || {}))
    if (tech === H.tech_names[t]) out.push({kind: 'ability', name: ab.replace(/_/g, ' ')});
  for (const [res, tech] of Object.entries(H.hidden_resource_tech || {}))
    if ((Array.isArray(tech) ? tech : [tech]).includes(H.tech_names[t]))
      out.push({kind: 'reveals', name: res});
  return out;
}

/** Radial positions for every tech: roots evenly around the hub, each node's
 *  children sharing the parent's angular slice. */
function techLayout(H) {
  const n = H.tech_names.length;
  const kids = Array.from({length: n}, () => []);
  const roots = [];
  for (let t = 0; t < n; t++) {
    const r = H.tech_requires[t];
    if (r < 0) roots.push(t); else kids[r].push(t);
  }
  const pos = new Array(n);
  const RING = [0, 98, 168, 232];               // radius by tier
  const place = (t, a0, a1, depth) => {
    const mid = (a0 + a1) / 2;
    const r = RING[Math.min(depth, RING.length - 1)];
    pos[t] = {a: mid, r, x: Math.cos(mid) * r, y: Math.sin(mid) * r};
    const ks = kids[t];
    if (!ks.length) return;
    // Leave a margin so sibling sub-trees do not touch.
    const pad = (a1 - a0) * 0.06, w = (a1 - a0 - 2 * pad) / ks.length;
    ks.forEach((c, i) => place(c, a0 + pad + i * w, a0 + pad + (i + 1) * w, depth + 1));
  };
  const slice = (Math.PI * 2) / roots.length;
  // Start at the top; the first root reads as "12 o'clock" like the game's.
  roots.forEach((t, i) => place(t, -Math.PI / 2 + i * slice - slice / 2,
                                   -Math.PI / 2 + i * slice + slice / 2, 1));
  return pos;
}

class TechTree {
  constructor(el, header, infoEl) {
    this.el = el; this.H = header; this.info = infoEl;
    this.pos = techLayout(header);
    this.onPick = null;
    this.sel = -1;
  }

  render(frame, player, opts) {
    opts = opts || {};
    const H = this.H, f = frame, n = H.tech_names.length;
    const mask = f.tech[player];
    const R = 300;
    const svg = svgEl('svg', {viewBox: `${-R} ${-R} ${2 * R} ${2 * R}`});

    // Links first, so nodes sit on top of them.
    for (let t = 0; t < n; t++) {
      const req = H.tech_requires[t];
      const a = req < 0 ? {x: 0, y: 0} : this.pos[req];
      const b = this.pos[t];
      const done = hasTech(mask, t);
      svg.appendChild(svgEl('path', {
        class: 'link' + (done ? ' done' : ''),
        d: `M${a.x} ${a.y} L${b.x} ${b.y}`,
      }));
    }

    // The hub: the tribe, and what it is spending.
    const hub = svgEl('g', {});
    hub.appendChild(svgEl('circle', {class: 'hub', cx: 0, cy: 0, r: 40}));
    const tribe = svgEl('text', {x: 0, y: -2, style: 'font-size:11px'});
    tribe.textContent = H.tribes[player];
    hub.appendChild(tribe);
    const st = svgEl('text', {x: 0, y: 12, class: 'cost', style: 'font-size:11px'});
    st.textContent = `${f.stars[player]}★`;
    hub.appendChild(st);
    svg.appendChild(hub);

    for (let t = 0; t < n; t++) {
      const p = this.pos[t], done = hasTech(mask, t);
      const req = H.tech_requires[t];
      const open = req < 0 || hasTech(mask, req);
      const cost = researchCost(H, f, player, t);
      const can = !done && open && (opts.affordable ? f.stars[player] >= cost : true);
      const cls = done ? 'done' : open ? 'can' : 'lock';
      const g = svgEl('g', {
        class: `node ${cls}${t === this.sel ? ' sel' : ''}`,
        transform: `translate(${p.x} ${p.y})`,
      });
      g.appendChild(svgEl('circle', {class: 'disc', cx: 0, cy: 0, r: 22}));
      const glyph = svgEl('text', {class: 'glyph', x: 0, y: 1});
      glyph.textContent = TECH_GLYPH[H.tech_names[t]] || '✦';
      g.appendChild(glyph);
      const label = svgEl('text', {x: 0, y: 33});
      label.textContent = H.tech_names[t].replace(/_/g, ' ');
      g.appendChild(label);
      if (!done) {
        const c = svgEl('text', {class: 'cost', x: 0, y: 43});
        c.textContent = open ? `${cost}★` : 'locked';
        g.appendChild(c);
      }
      g.addEventListener('mouseenter', () => this.describe(t, frame, player));
      g.addEventListener('mouseleave', () => this.describe(-1));
      if (open && !done && this.onPick) {
        g.addEventListener('click', () => this.onPick(t, cost, can));
      }
      svg.appendChild(g);
    }

    this.el.replaceChildren(svg);
  }

  describe(t, frame, player) {
    if (!this.info) return;
    if (t < 0) {
      this.info.className = 'techinfo muted';
      this.info.textContent = 'hover a technology';
      return;
    }
    const H = this.H;
    const un = techUnlocks(H, t);
    const req = H.tech_requires[t];
    const cost = researchCost(H, frame, player, t);
    const done = hasTech(frame.tech[player], t);
    this.info.className = 'techinfo';
    const parts = un.length
      ? un.map(u => `<span class="chip">${u.kind}: ${u.name.replace(/_/g, ' ')}</span>`).join('')
      : '<span class="muted">no direct unlock</span>';
    this.info.innerHTML =
      `<b>${H.tech_names[t].replace(/_/g, ' ')}</b> ` +
      `<span class="muted">tier ${H.tech_tier[t]}` +
      (req >= 0 ? ` · after ${H.tech_names[req].replace(/_/g, ' ')}` : '') +
      ` · ${done ? 'researched' : cost + ' stars'}</span>` +
      `<div class="chips">${parts}</div>`;
  }
}
