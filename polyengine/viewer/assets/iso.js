/* iso.js — the board: isometric projection, camera, terrain, hit testing.
 *
 * Polytopia draws its world as a 2:1 isometric grid of chunky blocks, and the
 * single biggest reason the old viewer did not look like the game is that it
 * drew a flat grid of coloured squares. Everything here follows from that one
 * change: tiles are blocks with a top face and two lit side faces, water sits
 * *lower* than land so coastlines have a visible lip, mountains rise above
 * their tile and genuinely occlude what is behind them, and territory is a
 * coloured outline traced along tile edges rather than a wash of tint.
 *
 * Depth order is the painter's algorithm on (x + y): tiles further from the
 * camera are drawn first, and each tile draws its own contents immediately
 * after its own ground. That last part matters — drawing all terrain and then
 * all sprites would let a unit standing at the back paint over a mountain in
 * front of it.
 *
 * The content art (units, cities, resources, buildings) lives in sprites.js.
 */

const PC = ['#4da3ff', '#ff6b6b', '#5fd68a', '#ffd166',
            '#c58bff', '#ffa14d', '#4de0d0', '#ff8fc7'];
const pcolor = p => PC[((p % PC.length) + PC.length) % PC.length];

// Tile footprint in world units. TW = 2*TH is the classic 2:1 isometric ratio.
const TW = 76, TH = 38;
const BLOCK = 14;          // how thick a land tile's block is
// How far each terrain's top face sits above (negative) the base plane. Water
// being sunk is what gives the shore its lip; the game does the same.
const LIFT = {ocean: -7, water: -3, field: 0, forest: 0, mountain: 0};

// Top face, then the two visible sides. Sides are the same hue stepped down,
// which is what makes the blocks read as solid under a single light.
const TERRAIN = {
  ocean:    {top: '#1b5c96', l: '#154a7b', r: '#123f69'},
  water:    {top: '#3aa6dc', l: '#2f88b6', r: '#28749d'},
  field:    {top: '#82bd52', l: '#6aa03f', r: '#5a8a35'},
  forest:   {top: '#82bd52', l: '#6aa03f', r: '#5a8a35'},
  mountain: {top: '#9b9488', l: '#847d74', r: '#726c64'},
};

class Board {
  constructor(canvas, header) {
    this.cv = canvas;
    this.g = canvas.getContext('2d');
    this.H = header;
    this.S = header.size;
    this.N = this.S * this.S;
    this.cam = {x: 0, y: 0, z: 1};
    this.hover = -1;
    this.frame = null;
    this.opts = {};
    this._installInput();
    this._resize();
    addEventListener('resize', () => { this._resize(); this.draw(); });
    // The canvas also changes size without the window doing so — the side rail
    // widens for the tech tree — and a stale backing store shows up as a
    // stretched, blurry board.
    if (window.ResizeObserver)
      new ResizeObserver(() => { this._resize(); this.draw(); }).observe(canvas);
  }

  /** Terrain id at a tile.
   *
   * Normally static and read from the header. In play mode the server sends a
   * *fog-projected* frame whose terrain is erased outside what the player has
   * explored (-1), so the frame's copy wins when there is one — the page then
   * physically does not contain the map you have not walked. */
  terrainAt(k) {
    const t = this.frame && this.frame.terrain ? this.frame.terrain[k] : this.H.terrain[k];
    return t;
  }

  // --- projection --------------------------------------------------------
  world(k) {
    const x = k % this.S, y = (k / this.S) | 0;
    return {wx: (x - y) * TW / 2, wy: (x + y) * TH / 2};
  }

  /** Screen position of a tile's top-face centre, elevation included. */
  screen(k) {
    const {wx, wy} = this.world(k);
    const t = this.terrainAt(k);
    const lift = t < 0 ? 0 : (LIFT[this.H.terrain_names[t]] || 0);
    return this.toScreen(wx, wy + lift);
  }

  toScreen(wx, wy) {
    const z = this.cam.z;
    return {sx: (wx - this.cam.x) * z + this.cv.width / (2 * this.dpr),
            sy: (wy - this.cam.y) * z + this.cv.height / (2 * this.dpr)};
  }

  /** Which tile is under a screen point.
   *
   * Inverting the projection algebraically would ignore elevation, so a click
   * on a mountain's peak would select the tile behind it. Instead we test the
   * top-face diamonds front-to-back and take the first hit — a few hundred
   * point-in-diamond tests, far cheaper than a frame. */
  tileAt(px, py) {
    for (let k = this.N - 1; k >= 0; k--) {
      const j = this._order[k];
      const {sx, sy} = this.screen(j);
      const z = this.cam.z;
      const dx = Math.abs(px - sx) / (TW / 2 * z);
      const dy = Math.abs(py - sy) / (TH / 2 * z);
      if (dx + dy <= 1) return j;
    }
    return -1;
  }

  fit() {
    const S = this.S;
    const w = S * TW, h = S * TH + BLOCK * 2;
    const cw = this.cv.width / this.dpr, ch = this.cv.height / this.dpr;
    this.cam.z = Math.min(cw / (w + 60), ch / (h + 60));
    this.cam.x = 0;
    this.cam.y = ((S - 1) * TH) / 2;
    return this;
  }

  zoom(mult, ax, ay) {
    // Keep the point under the cursor fixed while zooming — anything else
    // feels like the map is sliding away from you.
    const cw = this.cv.width / (2 * this.dpr), ch = this.cv.height / (2 * this.dpr);
    if (ax === undefined) { ax = cw; ay = ch; }
    const wx = (ax - cw) / this.cam.z + this.cam.x;
    const wy = (ay - ch) / this.cam.z + this.cam.y;
    this.cam.z = Math.max(0.18, Math.min(3.2, this.cam.z * mult));
    this.cam.x = wx - (ax - cw) / this.cam.z;
    this.cam.y = wy - (ay - ch) / this.cam.z;
    this.draw();
  }

  centerOn(k) {
    const {wx, wy} = this.world(k);
    this.cam.x = wx; this.cam.y = wy;
    this.draw();
  }

  _resize() {
    this.dpr = Math.min(2, devicePixelRatio || 1);
    const r = this.cv.getBoundingClientRect();
    this.cv.width = Math.max(1, Math.round(r.width * this.dpr));
    this.cv.height = Math.max(1, Math.round(r.height * this.dpr));
  }

  _installInput() {
    const cv = this.cv;
    let drag = null;
    const at = e => {
      const r = cv.getBoundingClientRect();
      return [e.clientX - r.left, e.clientY - r.top];
    };
    cv.addEventListener('pointerdown', e => {
      const [x, y] = at(e);
      drag = {x, y, cx: this.cam.x, cy: this.cam.y, moved: 0};
      cv.setPointerCapture(e.pointerId);
    });
    cv.addEventListener('pointermove', e => {
      const [x, y] = at(e);
      if (drag) {
        drag.moved += Math.abs(x - drag.x) + Math.abs(y - drag.y);
        this.cam.x = drag.cx - (x - drag.x) / this.cam.z;
        this.cam.y = drag.cy - (y - drag.y) / this.cam.z;
        cv.classList.add('drag');
        this.draw();
        return;
      }
      const h = this.tileAt(x, y);
      if (h !== this.hover) { this.hover = h; this.draw(); this.onHover && this.onHover(h); }
    });
    cv.addEventListener('pointerup', e => {
      const [x, y] = at(e);
      const wasDrag = drag && drag.moved > 6;
      drag = null;
      cv.classList.remove('drag');
      if (!wasDrag && this.onPick) this.onPick(this.tileAt(x, y));
    });
    cv.addEventListener('pointerleave', () => {
      if (this.hover !== -1) { this.hover = -1; this.draw(); }
    });
    cv.addEventListener('wheel', e => {
      e.preventDefault();
      const [x, y] = at(e);
      this.zoom(e.deltaY < 0 ? 1.12 : 1 / 1.12, x, y);
    }, {passive: false});
  }

  set(frame, opts) { this.frame = frame; this.opts = opts || {}; this.draw(); }

  get _order() {
    // Back-to-front: increasing (x + y). Cached — it never changes.
    if (!this.__order) {
      this.__order = [...Array(this.N).keys()].sort((a, b) => {
        const ax = a % this.S, ay = (a / this.S) | 0;
        const bx = b % this.S, by = (b / this.S) | 0;
        return (ax + ay) - (bx + by) || ax - bx;
      });
    }
    return this.__order;
  }

  // --- drawing -----------------------------------------------------------
  draw() {
    const g = this.g, f = this.frame;
    g.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    const cw = this.cv.width / this.dpr, ch = this.cv.height / this.dpr;
    // Deep water beyond the map edge, so the board reads as an island in a sea
    // rather than a rectangle floating on the page background.
    g.fillStyle = '#0a2a47';
    g.fillRect(0, 0, cw, ch);
    if (!f) return;

    const o = this.opts, fogOf = o.fogOf ?? -1;
    const z = this.cam.z;
    for (const k of this._order) {
      const {sx, sy} = this.screen(k);
      // Cull generously: mountains and unit sprites reach well above the tile.
      if (sx < -TW * z * 2 || sx > cw + TW * z * 2 ||
          sy < -TH * z * 6 || sy > ch + TH * z * 6) continue;
      const dark = this.terrainAt(k) < 0 ||
                   (fogOf >= 0 && f.explored[fogOf] && !f.explored[fogOf][k]);
      this._tile(k, sx, sy, dark);
      if (!dark) this._contents(k, sx, sy);
    }
    this._overlays();
  }

  _diamond(sx, sy, sw, sh) {
    const g = this.g;
    g.beginPath();
    g.moveTo(sx, sy - sh); g.lineTo(sx + sw, sy);
    g.lineTo(sx, sy + sh); g.lineTo(sx - sw, sy);
    g.closePath();
  }

  _tile(k, sx, sy, dark) {
    const g = this.g, z = this.cam.z, f = this.frame;
    const t = this.terrainAt(k);
    const name = t < 0 ? 'field' : this.H.terrain_names[t];
    const sw = TW / 2 * z, sh = TH / 2 * z;
    const depth = (name === 'ocean' ? BLOCK * 0.5 : name === 'water' ? BLOCK * 0.8 : BLOCK) * z;

    if (dark) {
      // Unexplored: a flat slab of cloud. No terrain, no hint of shape — the
      // fog view has to be honest or it is worse than not having one.
      g.fillStyle = '#141a26';
      this._diamond(sx, sy, sw, sh); g.fill();
      g.strokeStyle = 'rgba(0,0,0,.35)'; g.lineWidth = 1; g.stroke();
      return;
    }

    const c = TERRAIN[name] || TERRAIN.field;
    // Sides first, so the top face's edge sits cleanly on them.
    g.fillStyle = c.l;
    g.beginPath();
    g.moveTo(sx - sw, sy); g.lineTo(sx, sy + sh);
    g.lineTo(sx, sy + sh + depth); g.lineTo(sx - sw, sy + depth);
    g.closePath(); g.fill();
    g.fillStyle = c.r;
    g.beginPath();
    g.moveTo(sx + sw, sy); g.lineTo(sx, sy + sh);
    g.lineTo(sx, sy + sh + depth); g.lineTo(sx + sw, sy + depth);
    g.closePath(); g.fill();

    g.fillStyle = c.top;
    this._diamond(sx, sy, sw, sh); g.fill();
    if (name === 'water' || name === 'ocean') {
      // A lighter inset diamond reads as a highlight on moving water and keeps
      // large stretches of sea from looking like flat paint.
      g.fillStyle = name === 'water' ? 'rgba(255,255,255,.10)' : 'rgba(255,255,255,.05)';
      this._diamond(sx, sy - sh * 0.18, sw * 0.55, sh * 0.5); g.fill();
    }
    g.strokeStyle = 'rgba(0,0,0,.13)'; g.lineWidth = 1;
    this._diamond(sx, sy, sw, sh); g.stroke();

    this._border(k, sx, sy, sw, sh);
  }

  /** Territory as an outline traced along the tile's own edges.
   *
   * The game draws a coloured rope around each empire, not a tint over it, and
   * the difference is not cosmetic: a tint fights every other colour on the
   * tile (terrain, resource, unit), which is why the old viewer had to keep it
   * down at 24% alpha where it was nearly unreadable. An edge is drawn only
   * where the neighbour's owner differs, so interiors stay clean. */
  _border(k, sx, sy, sw, sh) {
    const f = this.frame, S = this.S, g = this.g;
    const own = f.owner[k];
    if (own < 0) return;
    const x = k % S, y = (k / S) | 0;
    const diff = (nx, ny) =>
      nx < 0 || ny < 0 || nx >= S || ny >= S || f.owner[ny * S + nx] !== own;
    // The four diamond edges, in (dx, dy) grid terms.
    const edges = [
      [x, y - 1, [sx, sy - sh], [sx + sw, sy]],   // up-right on screen
      [x + 1, y, [sx + sw, sy], [sx, sy + sh]],   // down-right
      [x, y + 1, [sx, sy + sh], [sx - sw, sy]],   // down-left
      [x - 1, y, [sx - sw, sy], [sx, sy - sh]],   // up-left
    ];
    g.save();
    g.lineWidth = Math.max(1.6, 2.6 * this.cam.z);
    g.lineCap = 'round';
    for (const [nx, ny, a, b] of edges) {
      if (!diff(nx, ny)) continue;
      g.strokeStyle = 'rgba(0,0,0,.35)';
      g.beginPath(); g.moveTo(a[0], a[1] + 1.5); g.lineTo(b[0], b[1] + 1.5); g.stroke();
      g.strokeStyle = pcolor(own);
      g.beginPath(); g.moveTo(a[0], a[1]); g.lineTo(b[0], b[1]); g.stroke();
    }
    g.restore();
  }

  _contents(k, sx, sy) {
    const g = this.g, f = this.frame, H = this.H, z = this.cam.z;
    const terr = H.terrain_names[this.terrainAt(k)];

    if (terr === 'mountain') drawMountain(g, sx, sy, z);
    else if (terr === 'forest') drawForest(g, sx, sy, z, k);

    const imp = f.improvement[k];
    const impName = imp >= 0 ? H.improvement_names[imp] : null;
    const res = H.resource_names[f.resource[k]];
    const crowded = f.unit_type[k] >= 0 || f.city_level[k] > 0;
    if (res && res !== 'none' && !impName) drawResource(g, sx, sy, z, res, k, crowded);
    if (impName) drawImprovement(g, sx, sy, z, impName, f, k, this);

    // A garrisoned unit sits on exactly the same point as its city, and at
    // level 1 the city is a single house directly under the figure — it
    // disappeared completely. Separating them vertically is not enough (the
    // figure is tall), so the unit steps to the right of the settlement, which
    // is also where the game puts it.
    const shared = f.city_level[k] > 0 && f.unit_type[k] >= 0;
    if (f.city_level[k] > 0) drawCity(g, sx, sy, z, f, k);
    if (f.unit_type[k] >= 0)
      drawUnit(g, sx + (shared ? 14 * z : 0), sy + (shared ? 4 * z : 0),
               z, f, k, H, this.opts);

    // Move / attack markers for interactive play, and the last-action trail.
    const mark = this.opts.marks && this.opts.marks[k];
    if (mark) this._mark(sx, sy, z, mark);
    if (k === this.opts.sel) this._ring(sx, sy, z, '#ffffff', 2.4);
    if (k === this.hover && this.opts.hoverable !== false)
      this._ring(sx, sy, z, 'rgba(255,255,255,.45)', 1.6);
  }

  _ring(sx, sy, z, color, w) {
    const g = this.g;
    g.save(); g.strokeStyle = color; g.lineWidth = w;
    this._diamond(sx, sy, TW / 2 * z * 0.97, TH / 2 * z * 0.97); g.stroke(); g.restore();
  }

  _mark(sx, sy, z, kind) {
    const g = this.g;
    // Move markers are near-white, deliberately NOT green: almost every tile a
    // unit can walk to is grass, and a green highlight on green grass is
    // invisible at any sensible zoom. White reads on field, forest and water.
    const c = {move: '#eaf6ff', attack: '#ff6b6b', target: '#ffc94d',
               build: '#7fd0e3', src: '#ffc94d', ready: '#ffc94d'}[kind] || '#ffffff';
    g.save();
    if (kind === 'ready') {
      // "Something here can act." Outline only — filling every actionable tile
      // makes the whole board glow and says nothing.
      g.strokeStyle = c; g.lineWidth = 1.8; g.globalAlpha = 0.75;
      this._diamond(sx, sy, TW / 2 * z * 0.9, TH / 2 * z * 0.9); g.stroke();
      g.restore(); return;
    }
    g.globalAlpha = kind === 'move' ? 0.42 : 0.28; g.fillStyle = c;
    this._diamond(sx, sy, TW / 2 * z * 0.92, TH / 2 * z * 0.92); g.fill();
    g.globalAlpha = 1; g.strokeStyle = c; g.lineWidth = 2;
    this._diamond(sx, sy, TW / 2 * z * 0.92, TH / 2 * z * 0.92); g.stroke();
    if (kind === 'move') {                        // a pip, so it reads as "go here"
      g.fillStyle = c;
      g.beginPath(); g.arc(sx, sy, 2.2 * z, 0, 7); g.fill();
    }
    if (kind === 'attack') {
      g.strokeStyle = c; g.lineWidth = 2.4;
      const r = 7 * z;
      g.beginPath();
      g.moveTo(sx - r, sy - r * 0.5); g.lineTo(sx + r, sy + r * 0.5);
      g.moveTo(sx + r, sy - r * 0.5); g.lineTo(sx - r, sy + r * 0.5);
      g.stroke();
    }
    g.restore();
  }

  /** Things drawn in screen space, above the whole board. */
  _overlays() {
    const a = this.opts.lastAction;
    if (!a || a.src < 0) return;
    const g = this.g, z = this.cam.z;
    const from = this.screen(a.src);
    this._ring(from.sx, from.sy, z, '#ffc94d', 2.2);
    if (a.dst >= 0 && a.dst !== a.src) {
      const to = this.screen(a.dst);
      const hostile = a.name === 'ATTACK';
      this._ring(to.sx, to.sy, z, hostile ? '#ff6b6b' : '#7fe3a0', 2.2);
      // An arc rather than a straight line: on an isometric grid a straight
      // segment between two diamonds reads as ambiguous about which way it went.
      g.save();
      g.strokeStyle = hostile ? 'rgba(255,107,107,.85)' : 'rgba(127,227,160,.85)';
      g.lineWidth = 2.2; g.setLineDash([6, 4]);
      g.beginPath();
      const mx = (from.sx + to.sx) / 2, my = (from.sy + to.sy) / 2 - 26 * z;
      g.moveTo(from.sx, from.sy);
      g.quadraticCurveTo(mx, my, to.sx, to.sy);
      g.stroke(); g.restore();
    }
  }
}
