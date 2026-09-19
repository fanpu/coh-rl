"""One synthetic frame holding every unit presentation and every event effect.

The scripted fixture agents only ever field infantry, so a real match never
produces a tank duel, a mortar barrage, a garrisoned farmhouse under fire or a
collapsing barn. This builds all of that by hand and injects it through the
viewer's test hook, which makes it both the regression test for those
rendering paths (in 2D *and* 3D) and the source of the showcase screenshot.

It also paints a bocage hedgerow run, a fence line and shell craters into the
terrain, so the vertical features have something to be.
"""

from __future__ import annotations

# The action sits on the central crossroads of `hedgerow_crossing`: the
# east-west road is cell row 47-48 (world y 94-97) and the north-south road is
# cell column 47-48. The hedgerow run is laid just north of the junction, with
# the HMG dug in behind it looking down onto the road.
SHOWCASE_JS = r"""(() => {
  var data = JSON.parse(JSON.stringify(window.__viewer.data()));
  var f = JSON.parse(JSON.stringify(window.__viewer.frame()));

  // --- terrain: a bocage run, a fence line, trees and craters -------------
  function paint(cy, from, to, ch) {
    var row = data.map.terrain[cy];
    if (row === undefined) return;
    data.map.terrain[cy] = row.slice(0, from) + ch.repeat(to - from) + row.slice(to);
  }
  paint(44, 34, 53, 'H');      // the bocage the HMG fires from behind
  paint(45, 34, 36, 'H');
  paint(41, 57, 63, 'H');      // a second bank across the field
  paint(51, 40, 50, 'f');      // a crushable fence south of the road
  paint(53, 30, 34, 'T');      // a tree cluster
  paint(54, 31, 33, 'T');
  paint(50, 58, 61, 'c');      // shell craters where the mortar has been landing
  paint(49, 59, 61, 'c');
  paint(43, 24, 27, 'w');      // a low stone wall

  var base = f.squads[0] || {id: 1, o: 0, def: 'rifles', kind: 'infantry', x: 0, y: 0, h: 0,
                             th: 0, n: 1, max: 1, hp: 1, mhp: [60], w: [''], sup: 0, st: 'idle'};
  function sq(o) { var s = JSON.parse(JSON.stringify(base)); for (var k in o) s[k] = o[k]; return s; }

  f.squads = [
    // a heavy machine gun dug in behind the bocage, covering the road
    sq({id: 900, o: 0, def: 'hmg_team', kind: 'team_weapon', x: 82, y: 86, h: 1.5, fa: 1.45,
        n: 3, max: 3, hp: 1, st: 'set_up'}),
    // a mortar team still setting up, back in the field
    sq({id: 901, o: 0, def: 'mortar_team', kind: 'team_weapon', x: 66, y: 78, h: 1.0,
        n: 3, max: 3, hp: 1, st: 'setting_up', setup: 18}),
    // an abandoned anti-tank gun, crew gone
    sq({id: 902, o: 1, def: 'at_team', kind: 'team_weapon', x: 122, y: 84, h: 3.0, fa: 3.2,
        n: 0, max: 3, hp: 0, ab: 1}),
    // two tanks mid-duel across the crossroads
    sq({id: 903, o: 0, def: 'sherman_tank', kind: 'vehicle', x: 86, y: 104, h: -0.15, th: 0.32,
        n: 1, max: 1, hp: 0.85}),
    sq({id: 904, o: 1, def: 'panzer_iv', kind: 'vehicle', x: 122, y: 98, h: 3.05, th: 2.86,
        n: 1, max: 1, hp: 0.55}),
    // a halftrack coming up the road
    sq({id: 905, o: 1, def: 'sdkfz_halftrack', kind: 'vehicle', x: 108, y: 116, h: 4.3, th: 4.3,
        n: 1, max: 1, hp: 1}),
    // rifles advancing along the road
    sq({id: 906, o: 0, def: 'rifles', kind: 'infantry', x: 92, y: 97, h: 0.05,
        n: 6, max: 6, hp: 0.92, st: 'moving'}),
    // pinned and retreating under the MG
    sq({id: 907, o: 1, def: 'pioneers', kind: 'infantry', x: 104, y: 92, h: 3.0,
        n: 3, max: 4, hp: 0.3, sup: 2, st: 'retreating'}),
    // suppressed, and being reinforced
    sq({id: 908, o: 0, def: 'engineers', kind: 'infantry', x: 78, y: 100, h: 0.2,
        n: 4, max: 6, hp: 0.65, sup: 1, re: 1}),
    // garrisoned in the farmhouse, so it shows as lit windows and a badge
    sq({id: 909, o: 0, def: 'rifles', kind: 'infantry', x: 116, y: 110, h: 0, n: 4, max: 6,
        hp: 0.8, g: 912}),
    sq({id: 910, o: 0, def: 'rifles', kind: 'infantry', x: 116, y: 110, h: 0, n: 2, max: 6,
        hp: 0.5, g: 912}),
    // a kind this page has never heard of must degrade, not explode
    sq({id: 911, o: 1, def: 'flying_saucer', kind: 'hovercraft', x: 136, y: 76, h: 0.8,
        n: 1, max: 1, hp: 1})
  ];

  var b0 = f.buildings[0];
  function bl(o) { var b = JSON.parse(JSON.stringify(b0)); for (var k in o) b[k] = o[k]; return b; }
  f.buildings = [
    // the farmhouse the rifles are holding, taking fire
    bl({id: 912, o: null, nu: 1, def: 'barn', cx: 57, cy: 54, w: 5, h: 4, hp: 0.42, prog: 1, n: 2, q: []}),
    // a barracks going up
    bl({id: 913, o: 0, def: 'barracks', cx: 33, cy: 50, w: 3, h: 3, hp: 1, prog: 0.45, n: 0, q: []}),
    // a damaged tank depot, smoking
    bl({id: 914, o: 1, def: 'tank_depot', cx: 62, cy: 42, w: 3, h: 3, hp: 0.35, prog: 1, n: 0, q: []}),
    // the HQ, whose production queue the inspector shows
    bl({id: 915, o: 0, def: 'hq_us', cx: 30, cy: 58, w: 4, h: 4, hp: 0.9, prog: 1, n: 0,
        q: [['train', 'rifles', 9], ['research', 'phase_2', 20]]}),
    // a cottage collapsing right now
    bl({id: 916, o: null, nu: 1, def: 'house_small', cx: 53, cy: 39, w: 3, h: 3, hp: 0, prog: 1, n: 0, q: []}),
    // an observation post on the central victory point
    bl({id: 917, o: 0, def: 'observation_post', cx: 43, cy: 43, w: 1, h: 1, hp: 1, prog: 1, n: 0, q: []})
  ];

  f.events = [
    {k: 'shot', t: f.t, d: {src: 900, dst: 907, hit: true, src_pos: [82, 86], dst_pos: [104, 92]}},
    {k: 'shot', t: f.t, d: {src: 907, dst: 906, hit: false, src_pos: [104, 92], dst_pos: [92, 97]}},
    {k: 'explosion', t: f.t, d: {src: 901, pos: [118, 100], radius: 7}},
    {k: 'explosion', t: f.t, d: {src: 901, pos: [126, 108], radius: 5}},
    {k: 'vehicle_destroyed', t: f.t, d: {id: 990, owner: 1, def_id: 'panzer_iv', pos: [132, 92]}},
    {k: 'squad_destroyed', t: f.t, d: {id: 991, owner: 1, def_id: 'pioneers', pos: [110, 88]}},
    {k: 'building_destroyed', t: f.t, d: {id: 916, owner: null, def_id: 'house_small', pos: [110, 80]}},
    {k: 'weapon_abandoned', t: f.t, d: {id: 902, owner: 1, def_id: 'at_team', pos: [122, 84]}},
    {k: 'weapon_recrewed', t: f.t, d: {id: 900, owner: 0, def_id: 'hmg_team', by: 906, pos: [82, 86]}},
    {k: 'garrison_ejected', t: f.t, d: {squad: 910, building: 912, owner: 0, pos: [116, 114]}},
    {k: 'garrison_entered', t: f.t, d: {squad: 909, building: 912, owner: 0}},
    {k: 'building_completed', t: f.t, d: {building: 915, def_id: 'hq_us', owner: 0}},
    {k: 'construction_started', t: f.t, d: {building: 913, def_id: 'barracks', owner: 0, cell: [33, 50]}},
    {k: 'unit_trained', t: f.t, d: {building: 915, unit: 'rifles', squad: 906, owner: 0}},
    {k: 'research_completed', t: f.t, d: {building: 915, upgrade: 'phase_2', owner: 0}},
    {k: 'upgrade_bought', t: f.t, d: {squad: 906, upgrade: 'bar', owner: 0}},
    {k: 'reinforced', t: f.t, d: {squad: 908, owner: 0, def_id: 'engineers'}},
    // a trained unit with nowhere to stand: a word over the building, no map effect
    {k: 'unit_blocked', t: f.t, d: {building: 913, unit: 'rifles', owner: 0}},
    {k: 'quantum_strike', t: f.t, d: {pos: [136, 76]}},
    {k: 'no_position_at_all', t: f.t, d: {}}
  ];
  f.terrain_delta = [];

  // A second, event-free frame a few ticks later so the playhead can sit
  // mid-animation rather than parked on age 0, and so squads interpolate
  // (which is what makes the infantry bob).
  var after = JSON.parse(JSON.stringify(f));
  after.t = f.t + 8;
  after.events = [];
  after.vis = {};
  after.squads.forEach(function (s) { if (s.id === 906) { s.x += 2.2; } });

  data.frames = [f, after];
  data.winner = 0;
  window.__viewer.inject(data);
  window.__viewer.seekTick(f.t + 4);
  window.__viewer.select('building', 915);
  window.__viewer.camera(100, 96, 7.2);
  return f.events.length;
})()"""

SHOWCASE_EVENTS = 20
