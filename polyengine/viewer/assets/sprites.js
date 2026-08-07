/* sprites.js — everything that stands on a tile.
 *
 * All of it is drawn with canvas primitives. That is a deliberate constraint,
 * not a shortcut: the viewer has to stay a single self-contained HTML file you
 * can double-click, and the moment art becomes an image asset it becomes either
 * a megabyte of base64 per replay or a second file to lose.
 *
 * Everything is scaled by `z` (the camera zoom) off a nominal tile width of 76
 * world units, so sprites keep their proportions at any zoom.
 *
 * Two conventions, both borrowed from the game:
 *   - a sprite's anchor is the *centre of the tile's top face*, and it grows
 *     upward from there, so it looks like it is standing on the tile;
 *   - everything gets a soft contact shadow, which is most of what sells the
 *     illusion that a flat canvas has depth.
 */

// A deterministic hash so per-tile jitter (which way a tree leans, where the
// bushes sit) is stable across frames. Random jitter would make the whole map
// shimmer every time the frame advances.
function jit(k, salt) {
  let h = (k * 374761393 + salt * 668265263) | 0;
  h = (h ^ (h >> 13)) * 1274126177 | 0;
  return ((h ^ (h >> 16)) >>> 0) / 4294967296;
}

function shadow(g, sx, sy, z, rx, ry, alpha) {
  g.save();
  g.fillStyle = `rgba(0,0,0,${alpha === undefined ? 0.22 : alpha})`;
  g.beginPath(); g.ellipse(sx, sy + 2 * z, rx * z, ry * z, 0, 0, 7); g.fill();
  g.restore();
}

/** A rounded blob used for foliage, hills and unit bodies. */
function blob(g, x, y, rx, ry, fill, stroke) {
  g.beginPath(); g.ellipse(x, y, rx, ry, 0, 0, 7);
  g.fillStyle = fill; g.fill();
  if (stroke) { g.strokeStyle = stroke; g.lineWidth = 1; g.stroke(); }
}

// --- terrain features -------------------------------------------------------

function drawMountain(g, sx, sy, z) {
  // A grey pyramid with a lit left face and a snow cap. Drawn tall enough to
  // occlude the tile behind it, which is what makes ranges read as ranges.
  const h = 30 * z, w = 24 * z;
  g.beginPath();
  g.moveTo(sx, sy - h); g.lineTo(sx + w, sy + 5 * z);
  g.lineTo(sx - w, sy + 5 * z); g.closePath();
  g.fillStyle = '#8d857a'; g.fill();
  g.beginPath();                                  // lit face
  g.moveTo(sx, sy - h); g.lineTo(sx, sy + 5 * z); g.lineTo(sx - w, sy + 5 * z);
  g.closePath();
  g.fillStyle = '#a49c90'; g.fill();
  g.beginPath();                                  // snow
  g.moveTo(sx, sy - h);
  g.lineTo(sx + w * 0.34, sy - h * 0.5);
  g.lineTo(sx + w * 0.14, sy - h * 0.56);
  g.lineTo(sx - w * 0.08, sy - h * 0.44);
  g.lineTo(sx - w * 0.34, sy - h * 0.5);
  g.closePath();
  g.fillStyle = '#eef2f6'; g.fill();
}

function drawForest(g, sx, sy, z, k) {
  // Three conifers, jittered per tile so a forest looks grown rather than
  // stamped. Back tree first so the front ones overlap it.
  const spots = [[-9, 3, 0.86], [8, 5, 0.8], [0, -3, 1.0]];
  for (let i = 0; i < spots.length; i++) {
    const [ox, oy, s] = spots[i];
    const jx = (jit(k, i) - 0.5) * 6, jy = (jit(k, i + 9) - 0.5) * 3;
    const x = sx + (ox + jx) * z, y = sy + (oy + jy) * z;
    const hgt = 20 * s * z, wid = 7.5 * s * z;
    g.fillStyle = '#5a3b26';                       // trunk
    g.fillRect(x - 1.4 * z * s, y - 5 * z * s, 2.8 * z * s, 6 * z * s);
    g.beginPath();                                 // canopy, two tiers
    g.moveTo(x, y - hgt); g.lineTo(x + wid, y - hgt * 0.42);
    g.lineTo(x - wid, y - hgt * 0.42); g.closePath();
    g.fillStyle = '#2f7d3c'; g.fill();
    g.beginPath();
    g.moveTo(x, y - hgt * 0.7); g.lineTo(x + wid * 1.15, y - hgt * 0.05);
    g.lineTo(x - wid * 1.15, y - hgt * 0.05); g.closePath();
    g.fillStyle = '#3a9349'; g.fill();
  }
}

// --- resources --------------------------------------------------------------

const RES_COLOR = {fruit: '#ff6f91', crop: '#f4c94b', game: '#b9793f',
                   metal: '#9fb7d4', fish: '#7fe3e3', starfish: '#ff9ad5'};

/** `crowded` means a unit or a city already owns the centre of this tile.
 *  Resources then move back and shrink instead of being drawn *under* the
 *  occupant — a Warrior standing on Game used to look like a man riding a deer. */
function drawResource(g, sx, sy, z, res, k, crowded) {
  if (crowded) { sx -= 15 * z; sy -= 5 * z; z *= 0.62; }
  const x = sx + (jit(k, 3) - 0.5) * 10 * z, y = sy + 2 * z;
  switch (res) {
    case 'fruit':                                  // a bush of berries
      blob(g, x, y - 5 * z, 9 * z, 6.5 * z, '#3f9c4f');
      for (let i = 0; i < 4; i++)
        blob(g, x + (jit(k, i + 20) - 0.5) * 12 * z,
                y - 4 * z - jit(k, i + 30) * 6 * z, 2.1 * z, 2.1 * z, '#ff5d7d');
      break;
    case 'crop': {                                 // rows of wheat
      g.strokeStyle = '#e3b53a'; g.lineWidth = 1.6 * z;
      for (let i = -2; i <= 2; i++) {
        const bx = x + i * 4.6 * z;
        g.beginPath(); g.moveTo(bx, y); g.lineTo(bx + 1.2 * z, y - 10 * z); g.stroke();
        blob(g, bx + 1.4 * z, y - 11 * z, 2.1 * z, 3 * z, '#f4c94b');
      }
      break;
    }
    case 'game':                                   // deer: body, neck, antlers
      shadow(g, x, y, z, 8, 3);
      blob(g, x, y - 7 * z, 8 * z, 5 * z, '#a4703c');
      blob(g, x + 6 * z, y - 13 * z, 3.4 * z, 3.4 * z, '#b9793f');
      g.strokeStyle = '#6d4826'; g.lineWidth = 1.4 * z;
      g.beginPath();
      g.moveTo(x + 5 * z, y - 16 * z); g.lineTo(x + 3.5 * z, y - 20 * z);
      g.moveTo(x + 7.5 * z, y - 16 * z); g.lineTo(x + 9 * z, y - 20 * z);
      g.stroke();
      break;
    case 'metal':                                  // ore crystals
      for (let i = 0; i < 3; i++) {
        const bx = x + (i - 1) * 6 * z, by = y - 3 * z - (i === 1 ? 4 * z : 0);
        g.beginPath();
        g.moveTo(bx, by - 9 * z); g.lineTo(bx + 4 * z, by);
        g.lineTo(bx, by + 2 * z); g.lineTo(bx - 4 * z, by); g.closePath();
        g.fillStyle = i === 1 ? '#cfe0f2' : '#9fb7d4'; g.fill();
        g.strokeStyle = '#6f8bab'; g.lineWidth = 0.9 * z; g.stroke();
      }
      break;
    case 'fish':                                   // a fin breaking the surface
      g.fillStyle = '#7fe3e3';
      g.beginPath();
      g.moveTo(x - 8 * z, y - 3 * z); g.quadraticCurveTo(x, y - 11 * z, x + 8 * z, y - 3 * z);
      g.quadraticCurveTo(x, y - 1 * z, x - 8 * z, y - 3 * z); g.closePath(); g.fill();
      g.beginPath();
      g.moveTo(x + 6 * z, y - 4 * z); g.lineTo(x + 12 * z, y - 9 * z);
      g.lineTo(x + 11 * z, y - 2 * z); g.closePath(); g.fill();
      break;
    case 'starfish': {
      g.fillStyle = '#ff9ad5'; g.beginPath();
      for (let i = 0; i < 10; i++) {
        const r = (i % 2 ? 3.4 : 8.5) * z, a = -Math.PI / 2 + i * Math.PI / 5;
        const px = x + Math.cos(a) * r, py = y - 4 * z + Math.sin(a) * r * 0.62;
        i ? g.lineTo(px, py) : g.moveTo(px, py);
      }
      g.closePath(); g.fill();
      break;
    }
  }
}

// --- buildings --------------------------------------------------------------

/** A little house: a box with a pitched roof, drawn in isometric-ish 2.5D. */
function house(g, x, y, z, s, wall, roof) {
  const w = 7 * s * z, h = 8 * s * z;
  g.fillStyle = wall; g.fillRect(x - w, y - h, w * 2, h);
  g.fillStyle = 'rgba(0,0,0,.16)'; g.fillRect(x, y - h, w, h);
  g.beginPath();
  g.moveTo(x - w * 1.28, y - h); g.lineTo(x, y - h - 6 * s * z);
  g.lineTo(x + w * 1.28, y - h); g.closePath();
  g.fillStyle = roof; g.fill();
}

function drawImprovement(g, sx, sy, z, name, f, k, board) {
  const y = sy + 1 * z;
  switch (name) {
    case 'road': case 'bridge': {
      // A path along both isometric axes, so a network reads as connected.
      const S = board.S, x0 = k % S, y0 = (k / S) | 0;
      const conn = d => {
        const nx = x0 + d[0], ny = y0 + d[1];
        if (nx < 0 || ny < 0 || nx >= S || ny >= S) return false;
        const n = ny * S + nx, im = f.improvement[n];
        if (f.city_level[n] > 0) return true;
        if (im < 0) return false;
        const nm = board.H.improvement_names[im];
        return nm === 'road' || nm === 'bridge' || nm === 'port';
      };
      g.save();
      g.strokeStyle = name === 'bridge' ? '#b08a5a' : '#c9b189';
      g.lineWidth = 6 * z; g.lineCap = 'round';
      const dirs = [[[1, 0], [19, 9.5]], [[-1, 0], [-19, -9.5]],
                    [[0, 1], [-19, 9.5]], [[0, -1], [19, -9.5]]];
      let any = false;
      for (const [d, off] of dirs) {
        if (!conn(d)) continue;
        any = true;
        g.beginPath(); g.moveTo(sx, sy); g.lineTo(sx + off[0] * z, sy + off[1] * z); g.stroke();
      }
      if (!any) { g.beginPath(); g.arc(sx, sy, 4 * z, 0, 7); g.fillStyle = '#c9b189'; g.fill(); }
      g.restore();
      break;
    }
    case 'lumber_hut':  house(g, sx, y, z, 0.8, '#8a6a45', '#5f4a30'); break;
    case 'farm':
      g.fillStyle = '#d8b45c';
      g.save(); g.translate(sx, y - 2 * z); g.scale(1, 0.5); g.rotate(Math.PI / 4);
      g.fillRect(-11 * z, -11 * z, 22 * z, 22 * z);
      g.strokeStyle = '#b08f3f'; g.lineWidth = 1.4 * z;
      for (let i = -8; i <= 8; i += 5) {
        g.beginPath(); g.moveTo(i * z, -11 * z); g.lineTo(i * z, 11 * z); g.stroke();
      }
      g.restore();
      break;
    case 'mine':
      g.fillStyle = '#3a3630';
      g.beginPath(); g.ellipse(sx, y - 6 * z, 8 * z, 8 * z, 0, Math.PI, 0); g.fill();
      g.fillStyle = '#c8ccd4';
      g.fillRect(sx - 9 * z, y - 7 * z, 18 * z, 2 * z);
      break;
    case 'port': {
      house(g, sx - 6 * z, y - 3 * z, z, 0.62, '#8a6a45', '#4f6f8f');
      g.strokeStyle = '#8a6a45'; g.lineWidth = 2.4 * z;
      g.beginPath(); g.moveTo(sx + 1 * z, y); g.lineTo(sx + 14 * z, y + 5 * z); g.stroke();
      break;
    }
    case 'temple': case 'mountain_temple': case 'forest_temple': case 'water_temple': {
      // A stepped shrine; temples gain value with age, so the taller it looks
      // the better. The level is shown in the inspector, not on the tile.
      const c = {temple: '#e6dcc4', mountain_temple: '#cfd6de',
                 forest_temple: '#cfe6c4', water_temple: '#c4dfe6'}[name];
      shadow(g, sx, y, z, 11, 4);
      g.fillStyle = c; g.fillRect(sx - 11 * z, y - 4 * z, 22 * z, 4 * z);
      g.fillStyle = c; g.fillRect(sx - 8 * z, y - 15 * z, 16 * z, 11 * z);
      g.fillStyle = 'rgba(0,0,0,.16)'; g.fillRect(sx, y - 15 * z, 8 * z, 11 * z);
      g.beginPath();
      g.moveTo(sx - 11 * z, y - 15 * z); g.lineTo(sx, y - 23 * z);
      g.lineTo(sx + 11 * z, y - 15 * z); g.closePath();
      g.fillStyle = '#d8b45c'; g.fill();
      break;
    }
    case 'sawmill':  house(g, sx, y, z, 0.9, '#7d6242', '#3f6f4a'); break;
    case 'windmill':
      house(g, sx, y, z, 0.72, '#d8cdb4', '#8a6a45');
      g.strokeStyle = '#5f4a30'; g.lineWidth = 1.8 * z;
      for (let i = 0; i < 4; i++) {
        const a = i * Math.PI / 2 + 0.5;
        g.beginPath(); g.moveTo(sx, y - 15 * z);
        g.lineTo(sx + Math.cos(a) * 9 * z, y - 15 * z + Math.sin(a) * 9 * z); g.stroke();
      }
      break;
    case 'forge':
      house(g, sx, y, z, 0.85, '#6b6560', '#4a443f');
      g.fillStyle = '#3a3530'; g.fillRect(sx + 4 * z, y - 20 * z, 4 * z, 8 * z);
      g.fillStyle = 'rgba(255,160,60,.75)';
      g.beginPath(); g.arc(sx + 6 * z, y - 22 * z, 3.2 * z, 0, 7); g.fill();
      break;
    case 'market':
      g.fillStyle = '#c8542f';                      // a striped stall awning
      g.beginPath();
      g.moveTo(sx - 12 * z, y - 10 * z); g.lineTo(sx + 12 * z, y - 10 * z);
      g.lineTo(sx + 9 * z, y - 4 * z); g.lineTo(sx - 9 * z, y - 4 * z); g.closePath(); g.fill();
      g.fillStyle = '#eee6d2';
      for (let i = -1; i <= 1; i++)
        g.fillRect(sx + i * 8 * z - 1.6 * z, y - 10 * z, 3.2 * z, 6 * z);
      g.fillStyle = '#8a6a45';
      g.fillRect(sx - 10 * z, y - 4 * z, 2 * z, 5 * z);
      g.fillRect(sx + 8 * z, y - 4 * z, 2 * z, 5 * z);
      break;
    case 'embassy': {
      house(g, sx, y, z, 0.8, '#dcd6c8', '#8a8f9c');
      const owner = f.improvement_owner[k];
      g.strokeStyle = '#7a6a50'; g.lineWidth = 1.4 * z;
      g.beginPath(); g.moveTo(sx + 8 * z, y - 12 * z); g.lineTo(sx + 8 * z, y - 24 * z); g.stroke();
      g.fillStyle = owner >= 0 ? pcolor(owner) : '#aaa';
      g.fillRect(sx + 8 * z, y - 24 * z, 9 * z, 6 * z);
      break;
    }
    case 'lighthouse':
      shadow(g, sx, y, z, 8, 3);
      g.fillStyle = '#eee6d2';
      g.beginPath();
      g.moveTo(sx - 6 * z, y); g.lineTo(sx - 3.5 * z, y - 22 * z);
      g.lineTo(sx + 3.5 * z, y - 22 * z); g.lineTo(sx + 6 * z, y); g.closePath(); g.fill();
      g.fillStyle = '#d2453f';
      g.fillRect(sx - 5 * z, y - 12 * z, 10 * z, 4 * z);
      g.fillStyle = '#ffe08a';
      g.beginPath(); g.arc(sx, y - 24 * z, 4 * z, 0, 7); g.fill();
      break;
    case 'ruin':
      // Broken pillars. A ruin is examined, not harvested — the inspector says
      // so, but the silhouette should already suggest "go stand on this".
      shadow(g, sx, y, z, 11, 4);
      g.fillStyle = '#b9b2a4';
      g.fillRect(sx - 10 * z, y - 12 * z, 4.5 * z, 12 * z);
      g.fillRect(sx - 1 * z, y - 17 * z, 4.5 * z, 17 * z);
      g.fillRect(sx + 7 * z, y - 8 * z, 4.5 * z, 8 * z);
      g.fillStyle = '#8f887c'; g.fillRect(sx - 12 * z, y, 24 * z, 3 * z);
      break;
    default:
      // Monuments. All seven used to share one gilded obelisk, which made a
      // board look like it was covered in them and made counting impossible —
      // there is exactly one per task, seven per player at most. Each now has
      // its own silhouette so you can read which ones a tribe has earned.
      drawMonument(g, sx, y, z, name);
  }
}

/** The seven monuments, one per task, each with its own silhouette.
 *
 *  They are the densest scoring tiles in the game at 400 points apiece, so
 *  "how many has this tribe built, and which" is a question the board should
 *  answer at a glance rather than hide behind seven identical obelisks. */
function drawMonument(g, sx, y, z, name) {
  const plinth = (w, col) => {
    shadow(g, sx, y, z, 12, 4.5);
    g.fillStyle = col;
    g.fillRect(sx - w * z, y - 5 * z, 2 * w * z, 5 * z);
  };
  switch (name) {
    case 'altar_of_peace': {          // pacifist: a pale altar under a dove
      plinth(11, '#cfd8e6');
      g.fillStyle = '#eef3fa';
      g.fillRect(sx - 7 * z, y - 16 * z, 14 * z, 11 * z);
      g.fillStyle = '#ffffff';
      g.beginPath();                  // dove
      g.ellipse(sx, y - 23 * z, 7 * z, 4 * z, -0.3, 0, 7); g.fill();
      g.beginPath();
      g.moveTo(sx - 2 * z, y - 25 * z); g.lineTo(sx + 5 * z, y - 32 * z);
      g.lineTo(sx + 6 * z, y - 24 * z); g.closePath(); g.fill();
      break;
    }
    case 'emperors_tomb': {           // wealth: a stepped pyramid
      plinth(12, '#c9b08a');
      g.fillStyle = '#e2c79c';
      for (let i = 0; i < 4; i++) {
        const w = (10 - i * 2.4) * z;
        g.fillRect(sx - w, y - 10 * z - i * 5 * z, 2 * w, 5 * z);
      }
      break;
    }
    case 'eye_of_god': {              // explorer: an eye on a pillar
      plinth(9, '#b9a7d8');
      g.fillStyle = '#d8ccee';
      g.fillRect(sx - 4 * z, y - 20 * z, 8 * z, 15 * z);
      g.fillStyle = '#ffffff';
      g.beginPath(); g.ellipse(sx, y - 25 * z, 9 * z, 5.5 * z, 0, 0, 7); g.fill();
      g.fillStyle = '#4a3f7a';
      g.beginPath(); g.arc(sx, y - 25 * z, 3.4 * z, 0, 7); g.fill();
      break;
    }
    case 'gate_of_power': {           // killer: a war arch
      plinth(13, '#8d6a6a');
      g.fillStyle = '#b98a8a';
      g.fillRect(sx - 12 * z, y - 24 * z, 6 * z, 19 * z);
      g.fillRect(sx + 6 * z, y - 24 * z, 6 * z, 19 * z);
      g.fillRect(sx - 13 * z, y - 30 * z, 26 * z, 7 * z);
      break;
    }
    case 'grand_bazaar': {            // network: a domed market
      plinth(12, '#c88a4a');
      g.fillStyle = '#e8a95e';
      g.fillRect(sx - 10 * z, y - 15 * z, 20 * z, 10 * z);
      g.beginPath();                  // dome
      g.ellipse(sx, y - 15 * z, 11 * z, 9 * z, 0, Math.PI, 0); g.fill();
      g.fillStyle = '#ffd9a0';
      g.beginPath(); g.arc(sx, y - 26 * z, 2.6 * z, 0, 7); g.fill();
      break;
    }
    case 'park_of_fortune': {         // metropolis: a garden with a fountain
      plinth(12, '#7fae6a');
      blob(g, sx - 8 * z, y - 14 * z, 7 * z, 6 * z, '#4fb35e');
      blob(g, sx + 8 * z, y - 13 * z, 6 * z, 5 * z, '#3f9c4f');
      g.fillStyle = '#bcd9ee';
      g.beginPath(); g.ellipse(sx, y - 8 * z, 7 * z, 3.2 * z, 0, 0, 7); g.fill();
      g.strokeStyle = '#dceaf6'; g.lineWidth = 2 * z;
      g.beginPath(); g.moveTo(sx, y - 9 * z); g.lineTo(sx, y - 20 * z); g.stroke();
      break;
    }
    default: {                        // tower_of_wisdom (genius): a tall spire
      plinth(9, '#a8b7c9');
      g.fillStyle = '#cddaea';
      g.beginPath();
      g.moveTo(sx - 6 * z, y - 5 * z); g.lineTo(sx - 3.5 * z, y - 30 * z);
      g.lineTo(sx + 3.5 * z, y - 30 * z); g.lineTo(sx + 6 * z, y - 5 * z);
      g.closePath(); g.fill();
      g.fillStyle = '#ffe08a';
      g.beginPath();
      for (let i = 0; i < 10; i++) {
        const r = (i % 2 ? 2.2 : 5.5) * z, a = -Math.PI / 2 + i * Math.PI / 5;
        const px = sx + Math.cos(a) * r, py = y - 35 * z + Math.sin(a) * r;
        i ? g.lineTo(px, py) : g.moveTo(px, py);
      }
      g.closePath(); g.fill();
    }
  }
}

// --- cities -----------------------------------------------------------------

function drawCity(g, sx, sy, z, f, k) {
  const lvl = f.city_level[k], owner = f.city_owner[k];
  const col = owner >= 0 ? pcolor(owner) : '#9aa3b8';
  shadow(g, sx, sy, z, 15, 6);

  // Walls first: they sit around the settlement, so the houses overlap them.
  if (f.city_walls[k]) {
    g.strokeStyle = '#d8d2c4'; g.lineWidth = 3.2 * z;
    g.beginPath();
    g.moveTo(sx, sy - 13 * z); g.lineTo(sx + 26 * z, sy);
    g.lineTo(sx, sy + 13 * z); g.lineTo(sx - 26 * z, sy); g.closePath(); g.stroke();
    g.fillStyle = '#eee6d2';
    for (const [dx, dy] of [[0, -13], [26, 0], [0, 13], [-26, 0]])
      g.fillRect(sx + dx * z - 2.4 * z, sy + dy * z - 4 * z, 4.8 * z, 6 * z);
  }

  // One house per level, up to five, ringed around the tile centre. Past five
  // they would overlap into mush, and the number badge carries the rest.
  const n = Math.max(1, Math.min(5, lvl));
  const spots = [[0, -2], [-13, 3], [13, 3], [-7, -7], [8, -8]];
  for (let i = 0; i < n; i++) {
    const [ox, oy] = spots[i];
    house(g, sx + ox * z, sy + oy * z, z, i === 0 ? 0.86 : 0.66, '#efe7d6', col);
  }

  if (f.city_capital[k]) {
    // A short mast on the main roof with a broad pennant. It used to be a
    // twenty-pixel pole with a small flag at the top, which floated clear of
    // the city and read as scenery on the tile behind it.
    const top = sy - 24 * z;
    g.strokeStyle = '#6b5c42'; g.lineWidth = 2 * z;
    g.beginPath(); g.moveTo(sx, sy - 12 * z); g.lineTo(sx, top); g.stroke();
    g.fillStyle = col;
    g.beginPath();
    g.moveTo(sx + 1 * z, top); g.lineTo(sx + 16 * z, top + 5 * z);
    g.lineTo(sx + 1 * z, top + 10 * z); g.closePath(); g.fill();
    g.strokeStyle = 'rgba(0,0,0,.3)'; g.lineWidth = 1 * z; g.stroke();
  }
  if (f.city_park[k]) {                             // Park: a lone broad tree
    const x = sx + 17 * z, y = sy + 6 * z;
    g.fillStyle = '#5a3b26'; g.fillRect(x - 1.4 * z, y - 6 * z, 2.8 * z, 6 * z);
    blob(g, x, y - 10 * z, 7 * z, 6 * z, '#4fb35e');
  }
  if (f.city_workshop[k]) {                         // Workshop: a chimney
    g.fillStyle = '#6b6560'; g.fillRect(sx - 20 * z, sy - 14 * z, 4 * z, 10 * z);
    g.fillStyle = 'rgba(220,225,235,.5)';
    g.beginPath(); g.arc(sx - 18 * z, sy - 18 * z, 3.4 * z, 0, 7); g.fill();
  }

  // Level badge, always legible: a dark disc with the number. Parked on the
  // tile's left corner rather than under the settlement, because a unit
  // garrisoned in the city puts its health bar exactly there — and a city you
  // cannot read the level of is the one thing the board must never do.
  const bx = sx - 22 * z, by = sy + 4 * z;
  g.beginPath(); g.arc(bx, by, 8 * z, 0, 7);
  g.fillStyle = 'rgba(12,15,22,.86)'; g.fill();
  g.strokeStyle = col; g.lineWidth = 1.8 * z; g.stroke();
  g.fillStyle = '#fff';
  g.font = `700 ${Math.round(10 * z)}px ui-monospace,monospace`;
  g.textAlign = 'center'; g.textBaseline = 'middle';
  g.fillText(String(lvl), bx, by + 0.5 * z);

  if (f.city_pending_reward[k] > 0) {               // owes a level-up choice
    g.save();
    g.strokeStyle = '#ffc94d'; g.lineWidth = 2.6 * z;
    g.beginPath(); g.arc(bx, by, 12 * z, 0, 7); g.stroke();
    g.restore();
  }
}

// --- units ------------------------------------------------------------------

/** Per-unit accessory, drawn on top of the shared figure.
 *  Keyed by unit name so a new unit in the data shows up as a plain figure
 *  rather than crashing the renderer. */
const UNIT_ART = {
  warrior:  (g, x, y, z, ink) => weapon(g, x, y, z, ink, 'club'),
  archer:   (g, x, y, z, ink) => bow(g, x, y, z, ink),
  rider:    (g, x, y, z, ink) => mount(g, x, y, z),
  defender: (g, x, y, z, ink) => shield(g, x, y, z, ink),
  swordsman:(g, x, y, z, ink) => weapon(g, x, y, z, ink, 'sword'),
  knight:   (g, x, y, z, ink) => { mount(g, x, y, z); weapon(g, x, y, z, ink, 'lance'); },
  giant:    () => {},
  cloak:    () => {},
  dagger:   (g, x, y, z, ink) => weapon(g, x, y, z, ink, 'dagger'),
  mind_bender: (g, x, y, z, ink) => staff(g, x, y, z),
};

function weapon(g, x, y, z, ink, kind) {
  g.strokeStyle = ink; g.lineCap = 'round';
  const len = {club: 9, sword: 13, lance: 18, dagger: 7}[kind];
  g.lineWidth = (kind === 'club' ? 3 : 2) * z;
  g.beginPath();
  g.moveTo(x + 7 * z, y - 10 * z);
  g.lineTo(x + 7 * z + (kind === 'lance' ? len * z : 3 * z), y - 10 * z - len * z * 0.7);
  g.stroke();
  if (kind === 'sword' || kind === 'dagger') {      // crossguard
    g.lineWidth = 1.6 * z;
    g.beginPath(); g.moveTo(x + 6 * z, y - 13 * z); g.lineTo(x + 11 * z, y - 13 * z); g.stroke();
  }
}

function bow(g, x, y, z, ink) {
  g.strokeStyle = '#8a6a45'; g.lineWidth = 1.8 * z;
  g.beginPath(); g.arc(x + 8 * z, y - 14 * z, 7 * z, -1.2, 1.2); g.stroke();
  g.strokeStyle = ink; g.lineWidth = 1 * z;
  g.beginPath();
  g.moveTo(x + 8 * z + Math.cos(-1.2) * 7 * z, y - 14 * z + Math.sin(-1.2) * 7 * z);
  g.lineTo(x + 8 * z + Math.cos(1.2) * 7 * z, y - 14 * z + Math.sin(1.2) * 7 * z);
  g.stroke();
}

function shield(g, x, y, z, ink) {
  g.fillStyle = '#c8ccd4';
  g.beginPath();
  g.moveTo(x + 8 * z, y - 20 * z); g.lineTo(x + 15 * z, y - 17 * z);
  g.lineTo(x + 15 * z, y - 9 * z); g.lineTo(x + 8 * z, y - 5 * z);
  g.closePath(); g.fill();
  g.strokeStyle = ink; g.lineWidth = 1.2 * z; g.stroke();
}

function staff(g, x, y, z) {
  g.strokeStyle = '#8a6a45'; g.lineWidth = 1.8 * z;
  g.beginPath(); g.moveTo(x + 8 * z, y - 4 * z); g.lineTo(x + 10 * z, y - 26 * z); g.stroke();
  g.fillStyle = '#b78bff';
  g.beginPath(); g.arc(x + 10 * z, y - 28 * z, 3.4 * z, 0, 7); g.fill();
}

function mount(g, x, y, z) {
  g.fillStyle = '#8a5f3c';
  g.beginPath(); g.ellipse(x, y - 8 * z, 13 * z, 5.5 * z, 0, 0, 7); g.fill();
  g.fillStyle = '#7a5232';
  g.beginPath(); g.ellipse(x - 12 * z, y - 14 * z, 4 * z, 5 * z, 0, 0, 7); g.fill();
  g.strokeStyle = '#6b4526'; g.lineWidth = 2.4 * z;
  for (const dx of [-8, 0, 8]) {
    g.beginPath(); g.moveTo(x + dx * z, y - 5 * z); g.lineTo(x + dx * z, y + 1 * z); g.stroke();
  }
}

function boat(g, x, y, z, col, hull) {
  g.fillStyle = hull;
  g.beginPath();
  g.moveTo(x - 18 * z, y - 8 * z); g.lineTo(x + 18 * z, y - 8 * z);
  g.lineTo(x + 12 * z, y + 2 * z); g.lineTo(x - 12 * z, y + 2 * z);
  g.closePath(); g.fill();
  g.strokeStyle = 'rgba(0,0,0,.3)'; g.lineWidth = 1 * z; g.stroke();
  g.strokeStyle = '#6b5c42'; g.lineWidth = 2 * z;
  g.beginPath(); g.moveTo(x, y - 8 * z); g.lineTo(x, y - 28 * z); g.stroke();
  g.fillStyle = col;
  g.beginPath();
  g.moveTo(x + 1 * z, y - 28 * z); g.lineTo(x + 15 * z, y - 17 * z);
  g.lineTo(x + 1 * z, y - 12 * z); g.closePath(); g.fill();
}

function drawUnit(g, sx, sy, z, f, k, H, opts) {
  const t = f.unit_type[k], owner = f.unit_owner[k];
  const name = H.unit_names[t];
  const col = pcolor(owner), ink = 'rgba(20,24,32,.85)';
  const hidden = (f.unit_hidden || [])[k] === 1;
  const spent = f.unit_moved[k] === 1 && f.unit_attacked[k] === 1;
  // A fog-projected frame remembers an enemy's type and owner but not its
  // health — `unit_hp` comes back as -1 and nothing about its condition may be
  // drawn, because the player genuinely does not know it.
  const known = f.unit_hp[k] >= 0;

  g.save();
  // A hidden Cloak is ghosted, not removed: the god's-eye view hides nothing
  // from *you*, but you can see at a glance that the enemy cannot see it.
  if (hidden) g.globalAlpha = 0.42;
  else if (spent) g.globalAlpha = 0.62;

  const y = sy - 2 * z;
  shadow(g, sx, sy + 1 * z, z, 12, 4.5);

  if (H.unit_naval[t]) {
    boat(g, sx, y, z, col, name === 'juggernaut' ? '#6b6560' : '#8a6a45');
    if (name === 'juggernaut') {                    // armoured, oversized
      g.fillStyle = '#4a443f'; g.fillRect(sx - 12 * z, y - 18 * z, 24 * z, 10 * z);
    }
    if (name === 'bomber' || name === 'catapult') {
      g.fillStyle = '#4a443f';
      g.beginPath(); g.arc(sx - 8 * z, y - 13 * z, 4 * z, 0, 7); g.fill();
    }
    if (f.unit_carried[k] >= 0) {                   // the passenger, riding
      g.fillStyle = col;
      g.beginPath(); g.arc(sx - 7 * z, y - 14 * z, 3.4 * z, 0, 7); g.fill();
    }
  } else if (name === 'catapult') {
    g.fillStyle = '#8a6a45';
    g.fillRect(sx - 12 * z, y - 10 * z, 24 * z, 5 * z);
    g.strokeStyle = '#5f4a30'; g.lineWidth = 3 * z;
    g.beginPath(); g.moveTo(sx - 8 * z, y - 10 * z); g.lineTo(sx + 8 * z, y - 24 * z); g.stroke();
    g.fillStyle = '#4a443f';
    for (const dx of [-8, 8]) { g.beginPath(); g.arc(sx + dx * z, y - 4 * z, 4.5 * z, 0, 7); g.fill(); }
    g.fillStyle = col;
    g.beginPath(); g.arc(sx + 9 * z, y - 26 * z, 3.6 * z, 0, 7); g.fill();
  } else {
    // The shared figure: legs, a body in the owner's colour, a head. The head
    // is deliberately about half the body's width — at anything much larger the
    // silhouette stops reading as a person and starts reading as scenery, which
    // is exactly what happened to the Giant when its head matched its body.
    const big = name === 'giant';
    const s = big ? 1.5 : 1;
    if (name === 'rider' || name === 'knight') UNIT_ART[name](g, sx, y, z, ink);
    g.strokeStyle = ink; g.lineWidth = 2 * z * s; g.lineCap = 'round';
    if (name !== 'rider' && name !== 'knight') {
      g.beginPath();
      g.moveTo(sx - 3 * z * s, y - 5 * z * s); g.lineTo(sx - 3.5 * z * s, y);
      g.moveTo(sx + 3 * z * s, y - 5 * z * s); g.lineTo(sx + 3.5 * z * s, y);
      g.stroke();
    }
    const bodyY = (name === 'rider' || name === 'knight') ? y - 19 * z : y - 11 * z * s;
    g.fillStyle = big ? '#5aa856' : col;
    g.beginPath();
    g.ellipse(sx, bodyY, 6.4 * z * s, 8 * z * s, 0, 0, 7); g.fill();
    g.strokeStyle = 'rgba(0,0,0,.32)'; g.lineWidth = 1 * z; g.stroke();
    g.fillStyle = big ? '#7cc276' : '#f0d4b4';
    g.beginPath(); g.arc(sx, bodyY - 9 * z * s, 3.9 * z * s, 0, 7); g.fill();
    g.strokeStyle = 'rgba(0,0,0,.25)'; g.lineWidth = 0.8 * z; g.stroke();
    if (name === 'cloak') {                         // a hood, no face
      g.fillStyle = '#2b3446';
      g.beginPath(); g.arc(sx, bodyY - 11 * z, 6 * z, Math.PI, 0); g.fill();
      g.fillRect(sx - 6 * z, bodyY - 11 * z, 12 * z, 5 * z);
    }
    const art = UNIT_ART[name];
    if (art && name !== 'rider' && name !== 'knight') art(g, sx, y, z, ink);
  }
  g.restore();

  // --- status, always at full opacity so a spent unit's HP stays readable ---
  // A full-health unit gets no bar at all. Drawing one on every unit turned the
  // board into a field of bright rectangles that outweighed the units they
  // belonged to; the game only shows damage, and so do we.
  const maxHp = H.unit_max_hp[t] + (f.unit_veteran[k] === 1 ? 5 : 0);
  const frac = Math.max(0, Math.min(1, f.unit_hp[k] / maxHp));
  if (known && frac < 0.999) {
    const bw = 19 * z, bx = sx - bw / 2, by = sy + 8 * z, bh = 3.4 * z;
    g.fillStyle = 'rgba(8,10,16,.72)';
    g.fillRect(bx - 1, by - 1, bw + 2, bh + 2);
    g.fillStyle = frac > 0.55 ? '#5fd68a' : frac > 0.28 ? '#ffc94d' : '#ff6b6b';
    g.fillRect(bx, by, bw * frac, bh);
  }

  if (f.unit_veteran[k] === 1) star(g, sx + 19 * z, sy - 4 * z, 5 * z, '#ffc94d');
  // A banked promotion is the same information the client's star icon gives:
  // three kills, not yet cashed in for the free full heal.
  else if (known && f.unit_kills[k] >= 3)
    star(g, sx + 19 * z, sy - 4 * z, 5 * z, 'rgba(255,201,77,.55)');
  if (hidden) {                                     // dashed ring = invisible
    g.save(); g.setLineDash([4, 3]); g.strokeStyle = '#cfd8ff'; g.lineWidth = 1.6;
    g.beginPath(); g.ellipse(sx, sy, 15 * z, 8 * z, 0, 0, 7); g.stroke(); g.restore();
  }
  // The eye an adjacent enemy shows when a Cloak is near it — flagged on the
  // *watcher*, never on the Cloak, exactly as the game does it.
  if (opts && opts.senseAt && opts.senseAt[k]) eye(g, sx + 16 * z, sy - 6 * z, z);
}

function star(g, x, y, r, fill) {
  g.beginPath();
  for (let i = 0; i < 10; i++) {
    const rr = i % 2 ? r * 0.45 : r, a = -Math.PI / 2 + i * Math.PI / 5;
    const px = x + Math.cos(a) * rr, py = y + Math.sin(a) * rr;
    i ? g.lineTo(px, py) : g.moveTo(px, py);
  }
  g.closePath(); g.fillStyle = fill; g.fill();
}

function eye(g, x, y, z) {
  g.save();
  g.fillStyle = 'rgba(12,15,22,.85)';
  g.beginPath(); g.ellipse(x, y, 7 * z, 4.6 * z, 0, 0, 7); g.fill();
  g.fillStyle = '#ffe08a';
  g.beginPath(); g.arc(x, y, 2.6 * z, 0, 7); g.fill();
  g.restore();
}
