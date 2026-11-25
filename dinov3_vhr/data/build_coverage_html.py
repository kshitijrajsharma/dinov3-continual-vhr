"""Render the H3 coverage GeoJSON into the static coverage-map page.

Reads the footprint GeoJSON (per-cell source counts plus ground sample distance
statistics) and writes a self-contained HTML page: a canvas world map of the
pretraining tiles, colored either by source or by resolution, with a resolution
range filter. The page has no external dependencies; all data is inlined.
"""

import argparse
import json
from pathlib import Path

SOURCE_LABEL = {
    "fmow": "fMoW",
    "maxar": "Maxar",
    "oam": "OpenAerialMap",
    "naip": "NAIP",
    "ramp": "RAMP",
    "loveda": "LoveDA",
    "worldstrat": "WorldStrat",
    "planet": "Planet",
}


def to_mapdata(geojson: dict) -> dict:
    """Convert the footprint FeatureCollection into the compact map payload."""
    hexes = []
    totals: dict[str, int] = {}
    for feature in geojson["features"]:
        props = feature["properties"]
        counts = {key[2:]: value for key, value in props.items() if key.startswith("n_")}
        for source, count in counts.items():
            totals[source] = totals.get(source, 0) + count
        ring = [
            [round(lon, 5), round(lat, 5)] for lon, lat in feature["geometry"]["coordinates"][0]
        ]
        hexagon = {
            "r": ring,
            "d": max(counts, key=lambda source: counts[source]),
            "t": props["total"],
            "p": 1 if props["precise"] else 0,
            "c": counts,
        }
        if "gsd_med" in props:
            hexagon["g"] = [props["gsd_min"], props["gsd_med"], props["gsd_max"]]
        hexes.append(hexagon)
    sources = sorted(totals, key=lambda source: totals[source], reverse=True)
    return {
        "hexes": hexes,
        "sources": sources,
        "totals": totals,
        "total_tiles": sum(totals.values()),
    }


PAGE = r"""<meta charset="utf-8">
<title>Coverage map</title>
<style>
  :root {
    --bg:#0a0e16; --panel:#10172340; --panel-solid:#111a28; --ink:#e7eef8;
    --muted:#808fa8; --faint:#4a586e; --line:#1b2636; --grid:#16202f;
    --accent:#4fd0e0; --hair:#223247;
    --fmow:#f2b134; --maxar:#37c8d0; --oam:#63c95b; --naip:#ee6a41;
    --ramp:#a887e6; --loveda:#ec6aa0; --worldstrat:#5b8dff; --planet:#2fbf9f;
  }
  @media (prefers-color-scheme: light) {
    :root {
      --bg:#eef1f7; --panel:#ffffffcc; --panel-solid:#ffffff; --ink:#141d2b;
      --muted:#586274; --faint:#95a1b5; --line:#d6deea; --grid:#e2e8f2;
      --accent:#0d8fa4; --hair:#c4cedd;
    }
  }
  :root[data-theme="dark"] {
    --bg:#0a0e16; --panel:#10172340; --panel-solid:#111a28; --ink:#e7eef8;
    --muted:#808fa8; --faint:#4a586e; --line:#1b2636; --grid:#16202f;
    --accent:#4fd0e0; --hair:#223247;
  }
  :root[data-theme="light"] {
    --bg:#eef1f7; --panel:#ffffffcc; --panel-solid:#ffffff; --ink:#141d2b;
    --muted:#586274; --faint:#95a1b5; --line:#d6deea; --grid:#e2e8f2;
    --accent:#0d8fa4; --hair:#c4cedd;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--ink);
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif; -webkit-font-smoothing: antialiased;
  }
  .mono { font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace; }
  .wrap { max-width: 1180px; margin: 0 auto; padding: 26px 22px 40px; }
  .top { display: flex; align-items: baseline; gap: 14px; flex-wrap: wrap; }
  h1 { font-size: clamp(22px, 3.4vw, 30px); margin: 0; font-weight: 660; letter-spacing: -.01em; }
  .meta { color: var(--muted); font-size: 12.5px; }
  .meta b { color: var(--ink); font-weight: 600; font-variant-numeric: tabular-nums; }

  .board {
    position: relative; margin-top: 16px; border: 1px solid var(--line); border-radius: 14px;
    background: radial-gradient(120% 120% at 50% 0%, color-mix(in oklab, var(--accent) 6%, transparent), transparent 60%), var(--panel-solid);
    overflow: hidden;
  }
  canvas { display: block; width: 100%; height: auto; touch-action: none; cursor: grab; }
  canvas.drag { cursor: grabbing; }
  .reset {
    position: absolute; top: 10px; right: 10px; z-index: 5; opacity: 0; pointer-events: none;
    transition: opacity .15s; background: var(--panel-solid); border: 1px solid var(--hair);
    color: var(--ink); border-radius: 8px; padding: 5px 11px; font-size: 12px; cursor: pointer;
  }
  .reset.show { opacity: 1; pointer-events: auto; }

  .tip {
    position: fixed; pointer-events: none; z-index: 10; opacity: 0;
    transform: translate(-50%, calc(-100% - 12px)); transition: opacity .1s;
    background: var(--panel-solid); border: 1px solid var(--hair); border-radius: 9px;
    padding: 8px 11px; font-size: 12px; box-shadow: 0 8px 30px #0008; min-width: 148px;
  }
  .tip h4 { margin: 0 0 5px; font-size: 11px; color: var(--muted); font-weight: 600; }
  .tip .gsd { margin: 0 0 5px; font-size: 11px; color: var(--accent); font-variant-numeric: tabular-nums; }
  .tip .row { display: flex; justify-content: space-between; gap: 14px; padding: 1.5px 0; }
  .tip .row span:first-child { display: flex; align-items: center; gap: 6px; }
  .dot { width: 8px; height: 8px; border-radius: 2px; flex: none; }
  .v { font-variant-numeric: tabular-nums; }

  .controls { display: flex; flex-wrap: wrap; align-items: center; gap: 7px; margin-top: 15px; }
  .grouplabel { font-size: 10.5px; letter-spacing: .14em; text-transform: uppercase; color: var(--faint); margin-right: 2px; }
  .chip {
    display: inline-flex; align-items: center; gap: 7px; cursor: pointer; user-select: none;
    background: var(--panel); border: 1px solid var(--line); border-radius: 999px;
    padding: 5px 12px 5px 9px; font-size: 12.5px; color: var(--ink);
    transition: opacity .12s, border-color .12s;
  }
  .chip:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
  .chip.off { opacity: .34; }
  .chip .sw { width: 10px; height: 10px; border-radius: 3px; flex: none; }
  .chip .ct { color: var(--muted); font-variant-numeric: tabular-nums; font-size: 11px; }
  .chip.plat { border-style: dashed; }
  .chip.seg { padding: 5px 12px; }
  .chip.seg.active { border-color: var(--accent); color: var(--ink); }

  .slider { position: relative; width: 190px; height: 22px; }
  .slider .track, .slider .fill { position: absolute; top: 10px; height: 3px; border-radius: 2px; }
  .slider .track { left: 0; right: 0; background: var(--line); }
  .slider .fill { background: var(--accent); }
  .slider input {
    position: absolute; top: 0; left: 0; width: 100%; height: 22px; margin: 0;
    background: none; pointer-events: none; -webkit-appearance: none; appearance: none;
  }
  .slider input::-webkit-slider-runnable-track { height: 22px; background: none; }
  .slider input::-webkit-slider-thumb {
    -webkit-appearance: none; pointer-events: auto; margin-top: 4px;
    width: 14px; height: 14px; border-radius: 50%; cursor: pointer;
    background: var(--panel-solid); border: 2px solid var(--accent);
  }
  .slider input::-moz-range-thumb {
    pointer-events: auto; width: 14px; height: 14px; border-radius: 50%; cursor: pointer;
    background: var(--panel-solid); border: 2px solid var(--accent);
  }
  .gsdval { font-size: 11.5px; color: var(--muted); font-variant-numeric: tabular-nums; }
  .reslegend { display: none; align-items: center; gap: 8px; }
  .reslegend.show { display: inline-flex; }
  .reslegend .bar { width: 150px; height: 8px; border-radius: 4px; }
  .reslegend span { font-size: 11px; color: var(--muted); font-variant-numeric: tabular-nums; }

  .foot { margin-top: 14px; color: var(--faint); font-size: 11.5px; }
  .foot b { color: var(--muted); font-weight: 600; }
</style>

<div class="wrap">
  <div class="top">
    <h1>Coverage map</h1>
    <div class="meta"><b id="mtiles"></b> tiles &middot; <b>7</b> sources &middot; H3 res-3</div>
  </div>

  <div class="board">
    <canvas id="map" aria-label="World coverage map of pretraining tiles"></canvas>
    <button class="reset" id="reset" type="button">Reset view</button>
  </div>

  <div class="controls">
    <span class="grouplabel">Color</span>
    <button class="chip seg active" id="mode-source" data-mode="source" type="button">Source</button>
    <button class="chip seg" id="mode-resolution" data-mode="resolution" type="button">Resolution</button>
    <span style="width:14px"></span>
    <span class="grouplabel">Resolution</span>
    <div class="slider">
      <div class="track"></div><div class="fill" id="gfill"></div>
      <input type="range" id="gmin" min="0" max="1000" value="0" aria-label="minimum resolution">
      <input type="range" id="gmax" min="0" max="1000" value="1000" aria-label="maximum resolution">
    </div>
    <span class="gsdval mono" id="gsdval"></span>
    <div class="reslegend" id="reslegend">
      <span id="legfine"></span><div class="bar" id="legbar"></div><span id="legcoarse"></span>
    </div>
  </div>
  <div class="controls" id="platforms"></div>
  <div class="controls" id="sensors"></div>
  <div class="controls" id="sources"></div>
  <p class="foot"><b>&#9670; real coordinates.</b> Scroll to zoom, drag to pan, hover a cell for its source counts and resolution. Cell resolution is the median ground sample distance of its tiles.</p>
</div>

<div class="tip" id="tip"></div>

<script id="mapdata" type="application/json">__MAPDATA__</script>
<script>
  const DATA = JSON.parse(document.getElementById('mapdata').textContent);
  const PRECISE = new Set(['maxar','naip','oam','worldstrat','planet']);
  const LABEL = { fmow:'fMoW', maxar:'Maxar', oam:'OpenAerialMap', naip:'NAIP', ramp:'RAMP', loveda:'LoveDA', worldstrat:'WorldStrat', planet:'Planet' };
  const PLATFORM = {
    'Satellite': ['fmow','maxar','ramp','loveda','worldstrat','planet'],
    'Aerial': ['naip','oam'],
  };
  const SENSOR = {
    'WorldView / GeoEye / QuickBird': ['fmow','maxar','ramp'],
    'SPOT 6/7': ['worldstrat'],
    'SkySat / Pelican': ['planet'],
    'NAIP': ['naip'],
    'drone / mixed': ['oam'],
  };
  const css = n => getComputedStyle(document.documentElement).getPropertyValue('--'+n).trim();
  const fmt = n => n.toLocaleString('en-US');
  const clamp = (v, a, b) => Math.min(b, Math.max(a, v));

  const off = new Set();
  const maxT = Math.max(...DATA.hexes.map(h => h.t));

  // resolution domain (per-cell median gsd), log-scaled
  const meds = DATA.hexes.filter(h => h.g).map(h => h.g[1]);
  const GLO = Math.min(...meds), GHI = Math.max(...meds);
  const lnLo = Math.log(GLO), lnHi = Math.log(GHI);
  const posToGsd = p => Math.exp(lnLo + (p / 1000) * (lnHi - lnLo));
  const gsdToPos = g => clamp(Math.round((Math.log(g) - lnLo) / (lnHi - lnLo) * 1000), 0, 1000);
  const resColor = g => { const t = clamp((Math.log(g) - lnLo) / (lnHi - lnLo), 0, 1); return `hsl(${200 - t * 178} 70% 52%)`; };
  let colorMode = 'source';
  let gsdLo = GLO, gsdHi = GHI;

  const canvas = document.getElementById('map');
  const ctx = canvas.getContext('2d');
  let W = 0, H = 0, dpr = 1;
  const view = { s: 1, x: 0, y: 0 };
  const project = (lon, lat) => [ (lon + 180) / 360 * W, (90 - lat) / 180 * H ];

  const colorFor = h => colorMode === 'source' ? css(h.d) : (h.g ? resColor(h.g[1]) : css('faint'));
  const gsdVisible = h => !h.g || (h.g[1] >= gsdLo - 1e-9 && h.g[1] <= gsdHi + 1e-9);
  const shown = h => !off.has(h.d) && gsdVisible(h);

  function clampPan() {
    view.x = clamp(view.x, W - W * view.s, 0);
    view.y = clamp(view.y, H - H * view.s, 0);
    document.getElementById('reset').classList.toggle('show', view.s > 1.01);
  }
  function resize() {
    const cssW = canvas.parentElement.clientWidth, cssH = Math.round(cssW / 2);
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    W = cssW; H = cssH; canvas.width = cssW * dpr; canvas.height = cssH * dpr;
    canvas.style.height = cssH + 'px'; clampPan(); draw();
  }
  function draw() {
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    ctx.save();
    ctx.translate(view.x, view.y); ctx.scale(view.s, view.s);
    const lw = 1 / view.s;
    ctx.strokeStyle = css('grid'); ctx.lineWidth = lw; ctx.beginPath();
    for (let lon = -150; lon <= 150; lon += 30) { const [x] = project(lon, 0); ctx.moveTo(x, 0); ctx.lineTo(x, H); }
    for (let lat = -60; lat <= 60; lat += 30) { const [, y] = project(0, lat); ctx.moveTo(0, y); ctx.lineTo(W, y); }
    ctx.stroke();
    ctx.strokeStyle = css('line'); ctx.beginPath(); const [, eq] = project(0, 0); ctx.moveTo(0, eq); ctx.lineTo(W, eq); ctx.stroke();
    for (const h of DATA.hexes.filter(shown).sort((a, b) => a.t - b.t)) {
      const c = colorFor(h), a = 0.32 + 0.6 * (Math.log10(h.t + 1) / Math.log10(maxT + 1));
      ctx.beginPath();
      for (let i = 0; i < h.r.length; i++) { const [x, y] = project(h.r[i][0], h.r[i][1]); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }
      ctx.closePath();
      ctx.globalAlpha = a; ctx.fillStyle = c; ctx.fill();
      ctx.globalAlpha = Math.min(1, a + 0.25); ctx.lineWidth = 0.6 * lw; ctx.strokeStyle = c; ctx.stroke();
    }
    ctx.globalAlpha = 1; ctx.restore();
  }
  function inside(pt, ring) {
    let c = false;
    for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
      const xi = ring[i][0], yi = ring[i][1], xj = ring[j][0], yj = ring[j][1];
      if (((yi > pt[1]) !== (yj > pt[1])) && (pt[0] < (xj - xi) * (pt[1] - yi) / (yj - yi) + xi)) c = !c;
    }
    return c;
  }
  function screenToLonLat(clientX, clientY) {
    const rect = canvas.getBoundingClientRect();
    const bx = ((clientX - rect.left) - view.x) / view.s;
    const by = ((clientY - rect.top) - view.y) / view.s;
    return [ bx / W * 360 - 180, 90 - by / H * 180 ];
  }

  canvas.addEventListener('wheel', e => {
    e.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left, my = e.clientY - rect.top;
    const ns = clamp(view.s * Math.exp(-e.deltaY * 0.0016), 1, 48);
    const k = ns / view.s;
    view.x = mx - (mx - view.x) * k; view.y = my - (my - view.y) * k; view.s = ns;
    clampPan(); draw();
  }, { passive: false });
  let dragging = false, moved = false, last = null;
  canvas.addEventListener('pointerdown', e => { dragging = true; moved = false; last = [e.clientX, e.clientY]; canvas.setPointerCapture(e.pointerId); canvas.classList.add('drag'); });
  canvas.addEventListener('pointerup', e => { dragging = false; canvas.classList.remove('drag'); canvas.releasePointerCapture(e.pointerId); });
  canvas.addEventListener('dblclick', () => { view.s = 1; view.x = 0; view.y = 0; clampPan(); draw(); });
  document.getElementById('reset').onclick = () => { view.s = 1; view.x = 0; view.y = 0; clampPan(); draw(); };

  const tip = document.getElementById('tip');
  canvas.addEventListener('pointermove', e => {
    if (dragging) {
      moved = true; view.x += e.clientX - last[0]; view.y += e.clientY - last[1];
      last = [e.clientX, e.clientY]; clampPan(); draw(); tip.style.opacity = 0; return;
    }
    const [lon, lat] = screenToLonLat(e.clientX, e.clientY);
    let hit = null;
    for (const h of DATA.hexes) if (shown(h) && inside([lon, lat], h.r)) { hit = h; break; }
    if (!hit) { tip.style.opacity = 0; return; }
    const rows = Object.entries(hit.c).sort((a, b) => b[1] - a[1]).map(([s, n]) =>
      `<div class="row"><span><i class="dot" style="background:${css(s)}"></i>${LABEL[s]}</span><span class="v mono">${fmt(n)}</span></div>`).join('');
    const gsd = hit.g ? `<div class="gsd">GSD ${hit.g[0]}&ndash;${hit.g[2]} m &middot; med ${hit.g[1]} m</div>` : '';
    tip.innerHTML = `<h4>${fmt(hit.t)} tiles &middot; ${hit.p ? 'precise' : 'representative'}</h4>${gsd}${rows}`;
    tip.style.left = e.clientX + 'px'; tip.style.top = e.clientY + 'px'; tip.style.opacity = 1;
  });
  canvas.addEventListener('pointerleave', () => tip.style.opacity = 0);

  document.getElementById('mtiles').textContent = fmt(DATA.total_tiles);

  const GROUPS = { platforms: { label: 'Platform', map: PLATFORM }, sensors: { label: 'Sensor', map: SENSOR } };
  const groupOff = list => list.every(s => off.has(s));
  function sync() {
    document.querySelectorAll('.chip[data-src]').forEach(c => c.classList.toggle('off', off.has(c.dataset.src)));
    document.querySelectorAll('.chip[data-grp]').forEach(c => c.classList.toggle('off', groupOff(GROUPS[c.dataset.box].map[c.dataset.grp])));
    draw();
  }
  for (const [boxId, { label, map }] of Object.entries(GROUPS)) {
    const box = document.getElementById(boxId);
    box.innerHTML = `<span class="grouplabel">${label}</span>`;
    Object.keys(map).forEach(g => {
      const b = document.createElement('button'); b.className = 'chip plat'; b.type = 'button';
      b.dataset.grp = g; b.dataset.box = boxId; b.textContent = g;
      b.onclick = () => { const list = map[g], allOff = groupOff(list); list.forEach(s => allOff ? off.delete(s) : off.add(s)); sync(); };
      box.appendChild(b);
    });
  }
  const srcBox = document.getElementById('sources');
  srcBox.innerHTML = '<span class="grouplabel">Source</span>';
  DATA.sources.slice().sort((a, b) => (DATA.totals[b] || 0) - (DATA.totals[a] || 0)).forEach(s => {
    const b = document.createElement('button'); b.className = 'chip'; b.type = 'button'; b.dataset.src = s;
    b.innerHTML = `<i class="sw" style="background:${css(s)}"></i>${LABEL[s]} <span class="ct mono">${fmt(DATA.totals[s] || 0)}</span>${PRECISE.has(s) ? ' &#9670;' : ''}`;
    b.onclick = () => { off.has(s) ? off.delete(s) : off.add(s); sync(); };
    srcBox.appendChild(b);
  });

  // color mode + resolution filter
  const gmin = document.getElementById('gmin'), gmax = document.getElementById('gmax');
  const gfill = document.getElementById('gfill'), gsdval = document.getElementById('gsdval');
  const legend = document.getElementById('reslegend');
  function setMode(mode) {
    colorMode = mode;
    document.getElementById('mode-source').classList.toggle('active', mode === 'source');
    document.getElementById('mode-resolution').classList.toggle('active', mode === 'resolution');
    legend.classList.toggle('show', mode === 'resolution');
    draw();
  }
  document.getElementById('mode-source').onclick = () => setMode('source');
  document.getElementById('mode-resolution').onclick = () => setMode('resolution');
  function syncGsd() {
    let a = +gmin.value, b = +gmax.value;
    if (a > b) { [a, b] = [b, a]; }
    gsdLo = posToGsd(a); gsdHi = posToGsd(b);
    gfill.style.left = (a / 10) + '%'; gfill.style.right = (100 - b / 10) + '%';
    gsdval.textContent = `${gsdLo.toFixed(2)}–${gsdHi.toFixed(2)} m`;
    draw();
  }
  gmin.oninput = syncGsd; gmax.oninput = syncGsd;
  const stops = Array.from({ length: 9 }, (_, i) => resColor(posToGsd(i * 125))).join(', ');
  document.getElementById('legbar').style.background = `linear-gradient(90deg, ${stops})`;
  document.getElementById('legfine').textContent = GLO.toFixed(2) + ' m';
  document.getElementById('legcoarse').textContent = GHI.toFixed(2) + ' m';
  syncGsd();

  new ResizeObserver(resize).observe(canvas.parentElement);
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', draw);
  resize();
</script>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geojson", type=Path, default=Path("data/footprint_h3_gsd.geojson"))
    parser.add_argument("--out", type=Path, default=Path("data/index.html"))
    args = parser.parse_args()

    geojson = json.loads(args.geojson.read_text())
    mapdata = to_mapdata(geojson)
    page = PAGE.replace("__MAPDATA__", json.dumps(mapdata, separators=(",", ":")))
    args.out.write_text(page)

    covered = sum(1 for h in mapdata["hexes"] if "g" in h)
    print(f"hexes {len(mapdata['hexes'])}, with gsd {covered}, tiles {mapdata['total_tiles']}")
    print(f"wrote {args.out} ({len(page):,} bytes)")


if __name__ == "__main__":
    main()
