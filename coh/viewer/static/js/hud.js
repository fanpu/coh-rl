/* Everything outside the map surface: the resource HUD, the winner banner,
 * the timeline, the inspector panel, and the help/legend overlay.
 *
 * Shared by both views — the 3D battlefield and the 2D tactical map show the
 * same numbers and the same inspector, and the inspector applies the same fog
 * rules (it must never report live state the watched team cannot observe).
 */

import { S } from './state.js';
import { NEUTRAL, TERRAIN, clamp, factionName, mmss, title } from './util.js';
import {
  R, buildingFogMode, currentFrame, fog, gameOverAt, lastTick, playerColour,
  queueFrac, seenNow, squadHidden, squadMeta, teamColour, teamOf
} from './data.js';

const $ = function (id) { return document.getElementById(id); };

export function updateHud(frame) {
  (R.D.players || []).forEach(function (p, i) {
    const el = $('p' + i);
    if (!el) return;
    const r = frame.players[i] || { mp: 0, mu: 0, fu: 0, pop: 0, cap: 0 };
    el.style.setProperty('--team', playerColour(i));
    const over = r.pop > r.cap ? ' over' : '';
    el.innerHTML =
      '<div class="who">Player ' + i + ' <span class="fac">' + factionName(p.faction) +
      ' &middot; team ' + p.team + '</span></div>' +
      '<div class="res">' +
      '<span class="mp"><i>MP</i><b>' + Math.round(r.mp) + '</b></span>' +
      '<span class="mu"><i>MU</i><b>' + Math.round(r.mu) + '</b></span>' +
      '<span class="fu"><i>FU</i><b>' + Math.round(r.fu) + '</b></span>' +
      '<span class="pop' + over + '"><i>POP</i><b>' + r.pop + '/' + r.cap + '</b></span>' +
      '</div>';
  });

  $('clock').textContent = mmss(frame.t / R.TPS);
  $('tickets').innerHTML = (frame.tickets || []).map(function (v, team) {
    return '<span style="color:' + teamColour(team) + '">' + Math.round(v) + '</span>';
  }).join('<span style="color:#6b6656">/</span>');

  $('mapname').textContent = title(R.D.map.name);
  updateBanner();
}

/** The `game_over` event, shown as a banner once the playhead reaches it. */
function updateBanner() {
  const banner = $('banner');
  const over = gameOverAt(S.tick);
  if (over === null) { banner.hidden = true; return; }

  let winner = over.d.winner;
  if (winner === null || winner === undefined) winner = R.D.winner;
  const drawn = winner === -1 || winner === null || winner === undefined;
  banner.hidden = false;
  banner.style.setProperty('--team', drawn ? NEUTRAL : teamColour(winner));
  banner.innerHTML =
    '<b>' + (drawn ? 'Draw' : 'Team ' + winner + ' wins') + '</b>' +
    (over.d.reason ? '<span>' + title(over.d.reason) + '</span>' : '');
}

export function drawTimeline() {
  const tl = $('timeline');
  const g = tl.getContext('2d');
  const w = tl.clientWidth, h = tl.clientHeight;
  g.setTransform(S.dpr, 0, 0, S.dpr, 0, 0);
  g.clearRect(0, 0, w, h);

  const total = lastTick() || 1;
  g.fillStyle = 'rgba(255,255,255,0.08)';
  g.fillRect(0, h / 2 - 4, w, 8);
  g.fillStyle = 'rgba(160,190,140,0.30)';
  g.fillRect(0, h / 2 - 4, w * clamp(S.tick / total, 0, 1), 8);

  R.markers.forEach(function (m) {
    g.fillStyle = m.c;
    g.fillRect(clamp(m.t / total, 0, 1) * w - 0.5, h / 2 - 9, 1.4, 18);
  });

  const x = clamp(S.tick / total, 0, 1) * w;
  g.fillStyle = '#f2e9c8';
  g.fillRect(x - 1.5, 1, 3, h - 2);

  g.fillStyle = 'rgba(200,192,170,0.75)';
  g.font = '10px system-ui, sans-serif';
  g.textAlign = 'right';
  g.fillText(mmss(total / R.TPS), w - 3, h - 2);
}

// ------------------------------------------------------------------ panel --

export function closePanel() { S.selection = null; $('panel').hidden = true; }

export function updatePanel(frame) {
  const el = $('panel');
  const body = $('panel-body');
  let entity = null;
  let seenTick = null;

  if (S.selection.kind === 'squad') {
    entity = frame.squads.filter(function (s) { return s.id === S.selection.id; })[0];
    // An enemy squad out of vision is not reported on at all: the inspector
    // must never leak live state the watched team cannot observe.
    if (entity && squadHidden(entity)) {
      showHiddenPanel(el, 'Enemy squad', 'out of vision');
      return;
    }
  } else {
    entity = frame.buildings.filter(function (b) { return b.id === S.selection.id; })[0];
    const seen = seenNow();
    const mode = entity ? buildingFogMode(entity, seen)
                        : (seen && seen[S.selection.id] ? 'ghost' : 'live');
    if (mode === 'hidden') {
      showHiddenPanel(el, 'Enemy building', 'never seen');
      return;
    }
    if (mode === 'ghost' || (!entity && seen && seen[S.selection.id])) {
      entity = seen[S.selection.id].b;
      seenTick = seen[S.selection.id].t;
    }
  }

  if (!entity) {
    el.hidden = false;
    $('panel-title').textContent = title(S.selection.kind) + ' ' + S.selection.id;
    body.innerHTML = '<p style="color:var(--dim)">not present in this frame</p>';
    return;
  }

  el.hidden = false;
  const swatch = seenTick !== null ? '#7d7a70' : (entity.nu ? NEUTRAL : playerColour(entity.o));
  $('panel-title').innerHTML =
    '<span style="color:' + swatch + '">■</span> ' + title(entity.def) +
    ' <span style="color:var(--dim);font-weight:400">#' + entity.id +
    (seenTick !== null ? ' · last known' : '') + '</span>';

  const rows = [];
  function kv(k, v) { rows.push('<dt>' + k + '</dt><dd>' + v + '</dd>'); }
  function owner(e) {
    return e.o === null || e.o === undefined ? 'neutral'
      : 'player ' + e.o + ' (team ' + teamOf(e.o) + ')';
  }

  if (seenTick !== null) {
    // Last known: position and identity only, never live HP, progress or queue.
    kv('owner', owner(entity));
    kv('cell', entity.cx + ', ' + entity.cy);
    kv('footprint', entity.w + ' × ' + entity.h + ' cells');
    kv('last seen', mmss(seenTick / R.TPS));
    body.innerHTML = '<dl class="kv">' + rows.join('') + '</dl>' +
      '<p style="color:var(--dim);margin:9px 0 0">Out of vision — showing the last ' +
      'thing team ' + fog.team + ' saw here.</p>';
    return;
  }

  if (S.selection.kind === 'squad') {
    const meta = squadMeta(entity.def);
    kv('owner', owner(entity));
    kv('kind', title(entity.kind));
    kv('state', title(entity.st));
    if (entity.ord) kv('order', entity.ord);
    kv('position', entity.x.toFixed(1) + ', ' + entity.y.toFixed(1) + ' m');
    kv('heading', Math.round(entity.h * 180 / Math.PI) + '°');
    if (entity.kind === 'vehicle') kv('turret', Math.round(entity.th * 180 / Math.PI) + '°');
    if (entity.fa !== undefined) {
      kv('arc centre', Math.round(entity.fa * 180 / Math.PI) + '° ±' + ((meta.arc_deg || 0) / 2) + '°');
    }
    kv('members', entity.n + ' / ' + entity.max);
    kv('health', Math.round(entity.hp * 100) + '%');
    kv('suppression', ['none', 'suppressed', 'pinned'][entity.sup] || '?');
    if (entity.tg !== undefined) kv('target', '#' + entity.tg);
    if (entity.g !== undefined) kv('garrisoned in', '#' + entity.g);
    if (entity.ab) kv('crew', 'abandoned');
    if (entity.re) kv('reinforcing', 'yes');
    if (entity.setup !== undefined) kv('setting up', (entity.setup / R.TPS).toFixed(1) + ' s left');
    if (meta.sight) kv('sight', meta.sight + ' m');
    if (meta.range) kv('range', meta.range + ' m');

    const hp = (entity.mhp || []).map(function (v, i) {
      const name = (entity.w || [])[i];
      return '<div>' + (name ? title(name) : 'unarmed') + ' &mdash; ' + Math.round(v) + ' hp</div>';
    }).join('') || '<div style="color:var(--dim)">none</div>';
    body.innerHTML = '<dl class="kv">' + rows.join('') + '</dl>' +
      '<div class="sect">Members</div>' + hp;
  } else {
    kv('owner', owner(entity));
    kv('cell', entity.cx + ', ' + entity.cy);
    kv('footprint', entity.w + ' × ' + entity.h + ' cells');
    kv('health', Math.round(entity.hp * 100) + '%');
    kv('construction', Math.round(entity.prog * 100) + '%');
    kv('garrison', entity.n);
    const queue = (entity.q || []).map(function (q) {
      return '<div>' + title(q[1]) + ' <span style="color:var(--dim)">(' + q[0] + ')</span>' +
        '<div class="bar"><i style="width:' + (queueFrac(q) * 100).toFixed(0) + '%"></i></div>' +
        q[2].toFixed(1) + ' s left</div>';
    }).join('') || '<div style="color:var(--dim)">empty</div>';
    body.innerHTML = '<dl class="kv">' + rows.join('') + '</dl>' +
      '<div class="sect">Production queue</div>' + queue;
  }
}

/** The panel for something the watched team cannot report on. */
function showHiddenPanel(el, what, why) {
  el.hidden = false;
  $('panel-title').textContent = what;
  $('panel-body').innerHTML =
    '<p style="color:var(--dim)">Hidden from team ' + (fog ? fog.team : '?') + ' (' + why + ').</p>';
}

// ----------------------------------------------------------------- legend --

export function buildLegend() {
  const items = [];
  Object.keys(TERRAIN).forEach(function (ch) {
    items.push(['<s style="background:' + TERRAIN[ch].base + '"></s>', TERRAIN[ch].name]);
  });
  (R.D.players || []).forEach(function (p, i) {
    items.push(['<s style="background:' + playerColour(i) + '"></s>',
                'Player ' + i + ' (' + p.faction + ', team ' + p.team + ')']);
  });
  items.push(['<s style="background:#5ae16e"></s>', 'Heavy cover']);
  items.push(['<s style="background:#ebcd46"></s>', 'Light cover / suppressed']);
  items.push(['<s style="background:#e2705f"></s>', 'Pinned / destroyed']);
  $('legend').innerHTML = items.map(function (it) {
    return '<div>' + it[0] + '<span>' + it[1] + '</span></div>';
  }).join('');
}

export function toggleHelp() {
  const h = $('help');
  h.hidden = !h.hidden;
}

/** A one-line, dismissable notice — used when WebGL is unavailable. */
export function notice(text) {
  $('notice-msg').textContent = text;
  $('notice').hidden = false;
}

/** Refresh the HUD chrome that depends on the current frame. */
export function updateChrome(frame) {
  updateHud(frame);
  drawTimeline();
  if (S.selection) updatePanel(frame);
}

export { currentFrame };
