import json
import math
import os
import re
import shutil
import sys
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from utils import get_abs_path, prt, get_tmp_path
from src.getters import get_zones_df, get_zone_dam_prices, get_cap_limits_df

OUTPUT_HTML = Path(get_abs_path("../index.html"))
MAP_DATA_DIR = Path(get_abs_path("../sample_data/map"))
GEO_DIR = MAP_DATA_DIR / "geo"
COUNTRIES_FILE = MAP_DATA_DIR / "countries.geo.json"
D3_FILE = MAP_DATA_DIR / "d3.v7.min.js"
LEGACY_CACHE = Path(get_tmp_path(".geo_cache"))
PRICE_DP = 2
FLOW_DP = 1
FLOW_SIGN = 1
NTC_TRADER_FACTOR = 1
SIMPLIFY_TOL = 0.02
MIN_RING_AREA = 0.005
ENTSOE_GEO_URL = (
    "https://raw.githubusercontent.com/EnergieID/entsoe-py/master/"
    "entsoe/geo/geojson/{zone}.geojson"
)
COUNTRY_GEO_URL = (
    "https://raw.githubusercontent.com/johan/world.geo.json/master/"
    "countries.geo.json"
)
D3_URL = "https://cdn.jsdelivr.net/npm/d3@7/dist/d3.min.js"

ISO2_TO_ISO3 = {
    "AL": "ALB", "AT": "AUT", "BA": "BIH", "BE": "BEL", "BG": "BGR",
    "CH": "CHE", "CZ": "CZE", "DE": "DEU", "DK": "DNK", "EE": "EST",
    "ES": "ESP", "FI": "FIN", "FR": "FRA", "GB": "GBR", "GR": "GRC",
    "HR": "HRV", "HU": "HUN", "IE": "IRL", "IT": "ITA", "LT": "LTU",
    "LU": "LUX", "LV": "LVA", "ME": "MNE", "MK": "MKD", "NL": "NLD",
    "NO": "NOR", "PL": "POL", "PT": "PRT", "RO": "ROU", "RS": "SRB",
    "SE": "SWE", "SI": "SVN", "SK": "SVK", "TR": "TUR", "UA": "UKR",
    "UK": "GBR", "XK": "XKX",
}
def _has(p):
    return p.exists() and p.stat().st_size > 0

def read_local(p):
    p = Path(p)
    if not _has(p):
        raise FileNotFoundError(str(p))
    return p.read_text(encoding="utf-8")


def _download(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("utf-8")


def migrate_legacy_cache():
    if not LEGACY_CACHE.exists():
        return
    GEO_DIR.mkdir(parents=True, exist_ok=True)
    moved = 0
    for src in LEGACY_CACHE.iterdir():
        if src.stat().st_size == 0 and not src.name.endswith(".missing"):
            continue
        if src.name == "d3.v7.min.js":
            dst = D3_FILE
        elif src.name == "countries.geo.json":
            dst = COUNTRIES_FILE
        elif src.name.startswith("zone_") and src.name.endswith(".geojson"):
            dst = GEO_DIR / (src.name[len("zone_"):])
        elif src.name.startswith("zone_") and src.name.endswith(".geojson.missing"):
            dst = GEO_DIR / (src.name[len("zone_"):-len(".geojson.missing")] + ".missing")
        else:
            continue
        if not dst.exists():
            shutil.copy2(src, dst)
            moved += 1
    if moved:
        prt(f"Copied {moved} cached asset(s) from {LEGACY_CACHE.name} to {MAP_DATA_DIR}",
            color="gray")


def download_assets(zone_codes, force=False):
    GEO_DIR.mkdir(parents=True, exist_ok=True)
    prt(f"Downloading map assets into {MAP_DATA_DIR} ...", color="cyan")

    for url, dst in ((D3_URL, D3_FILE), (COUNTRY_GEO_URL, COUNTRIES_FILE)):
        if force or not _has(dst):
            try:
                dst.write_text(_download(url), encoding="utf-8")
                prt(f"  saved {dst.name}", color="gray")
            except Exception as e:
                prt(f"  {dst.name} failed: {e}", color="red")

    got = skipped = missing = 0
    for code in zone_codes:
        dst = GEO_DIR / f"{code}.geojson"
        mark = GEO_DIR / f"{code}.missing"
        if not force and (_has(dst) or mark.exists()):
            skipped += 1
            continue
        try:
            dst.write_text(_download(ENTSOE_GEO_URL.format(zone=code)),
                           encoding="utf-8")
            if mark.exists():
                mark.unlink()
            got += 1
        except urllib.error.HTTPError as e:
            if e.code == 404:
                mark.write_text("", encoding="utf-8")
                missing += 1
            else:
                prt(f"  {code}: HTTP {e.code}", color="yellow")
        except Exception as e:
            prt(f"  {code}: {e}", color="yellow")
    prt(f"  zones: {got} downloaded, {skipped} already present, "
        f"{missing} without polygon upstream", color="gray")


def _rdp(points, tol):
    n = len(points)
    if n < 3:
        return list(points)
    keep = [False] * n
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        ax, ay = points[i]
        bx, by = points[j]
        dx, dy = bx - ax, by - ay
        seg = math.hypot(dx, dy)
        best_d, best_k = -1.0, None
        for k in range(i + 1, j):
            px, py = points[k]
            if seg == 0:
                d = math.hypot(px - ax, py - ay)
            else:
                d = abs(dx * (ay - py) - (ax - px) * dy) / seg
            if d > best_d:
                best_d, best_k = d, k
        if best_d > tol:
            keep[best_k] = True
            stack.append((i, best_k))
            stack.append((best_k, j))
    return [p for p, k in zip(points, keep) if k]


def _signed_area(ring):
    s = 0.0
    for (x1, y1), (x2, y2) in zip(ring, ring[1:]):
        s += x1 * y2 - x2 * y1
    return s / 2.0


def _orient(ring, clockwise):
    if (_signed_area(ring) < 0) != clockwise:
        ring = ring[::-1]
    return ring


def _ring_area(ring):
    s = 0.0
    for (x1, y1), (x2, y2) in zip(ring, ring[1:]):
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def _simplify_ring(ring):
    pts = _rdp([(p[0], p[1]) for p in ring], SIMPLIFY_TOL)
    if len(pts) < 4 or _ring_area(pts) < MIN_RING_AREA:
        return None
    if pts[0] != pts[-1]:
        pts.append(pts[0])
    return [[round(x, 4), round(y, 4)] for x, y in pts]


def _simplify_polygon(poly):
    outer = _simplify_ring(poly[0])
    if outer is None:
        return None
    holes = [h for h in (_simplify_ring(r) for r in poly[1:]) if h]
    return [_orient(outer, True)] + [_orient(h, False) for h in holes]


def simplify_geometry(geom):
    if not geom:
        return None
    t = geom.get("type")
    if t == "Polygon":
        p = _simplify_polygon(geom["coordinates"])
        return {"type": "Polygon", "coordinates": p} if p else None
    if t == "MultiPolygon":
        ps = [p for p in (_simplify_polygon(x) for x in geom["coordinates"]) if p]
        return {"type": "MultiPolygon", "coordinates": ps} if ps else None
    return geom


def feature_collection_geometry(fc):
    if not isinstance(fc, dict):
        return None
    t = fc.get("type")
    if t == "Feature":
        return fc.get("geometry")
    if t != "FeatureCollection":
        return fc
    polys = []
    for f in fc.get("features", []):
        g = (f or {}).get("geometry") or {}
        gt = g.get("type")
        if gt == "Polygon":
            polys.append(g["coordinates"])
        elif gt == "MultiPolygon":
            polys.extend(g["coordinates"])
    if not polys:
        return None
    if len(polys) == 1:
        return {"type": "Polygon", "coordinates": polys[0]}
    return {"type": "MultiPolygon", "coordinates": polys}

def zone_price_series(code):
    df = get_zone_dam_prices(code)
    if df is None or len(df) == 0:
        return None
    ts = pd.to_datetime(df["delivery_ts"])
    s = pd.Series(pd.to_numeric(df["dam_price"], errors="coerce").values,
                  index=pd.DatetimeIndex(ts), name=code)
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s if len(s) else None


def load_zone_polygons(zones, broader_sizes):
    geo = {}
    country_fc = None
    for z in zones:
        g = None
        try:
            fc = json.loads(read_local(GEO_DIR / f"{z['code']}.geojson"))
            g = simplify_geometry(feature_collection_geometry(fc))
        except Exception:
            pass
        if g is None and z["broader"] in ISO2_TO_ISO3 \
                and broader_sizes.get(z["broader"], 0) == 1:
            try:
                if country_fc is None:
                    country_fc = json.loads(read_local(COUNTRIES_FILE))
                iso3 = ISO2_TO_ISO3[z["broader"]]
                for f in country_fc.get("features", []):
                    if f.get("id") == iso3 and f.get("geometry"):
                        g = simplify_geometry(f["geometry"])
                        break
            except Exception as e:
                prt(f"  country polygon for {z['broader']} failed: {e}",
                    color="yellow")
        if g is not None:
            geo[z["code"]] = g
    return geo


def net_flows(ex, codes, window_index):
    raw = {}
    for col in ex.columns:
        if "-" not in col:
            continue
        a, b = col.rsplit("-", 1)
        if a not in codes or b not in codes or a == b:
            continue
        raw[(a, b)] = pd.to_numeric(ex[col], errors="coerce").reindex(window_index)

    flows = {}
    done = set()
    for (a, b), fwd in raw.items():
        lo, hi = (a, b) if a < b else (b, a)
        if (lo, hi) in done:
            continue
        done.add((lo, hi))
        f = raw.get((lo, hi))
        r = raw.get((hi, lo))
        if f is not None and r is not None:
            both = f.notna() & r.notna()
            mirror = bool(both.any()) and np.allclose(f[both], -r[both]) \
                and float(f[both].abs().sum()) > 0
            net = f.combine_first(-r) if mirror else f.sub(r, fill_value=0)
        elif f is not None:
            net = f
        else:
            net = -r
        if not net.notna().any() or float(net.abs().fillna(0).sum()) == 0:
            continue
        flows[f"{lo}-{hi}"] = [None if pd.isna(v) else round(float(v), FLOW_DP)
                               for v in net]
    return flows


def _col_series(df, col):
    if col not in df.columns:
        return None
    idx = pd.DatetimeIndex(pd.to_datetime(df["delivery_ts"]))
    s = pd.Series(pd.to_numeric(df[col], errors="coerce").values, index=idx)
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s if s.notna().any() else None


def _ntc_series(df):
    idx = pd.DatetimeIndex(pd.to_datetime(df["delivery_ts"]))

    def col(name):
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce").values
        return np.full(len(df), np.nan)

    ntc = np.fmax(col("coupling_capacity"), col("trader_capacity") * NTC_TRADER_FACTOR)
    s = pd.Series(ntc, index=idx)
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s if s.notna().any() else None


def load_border_data(codes):
    codes = sorted(codes)
    ex_series, caps = {}, {}
    n_files = 0
    for a in codes:
        for b in codes:
            if a == b:
                continue
            try:
                df = get_cap_limits_df(a, b)
            except FileNotFoundError:
                continue
            except Exception as e:
                prt(f"  {a}-{b}: {e}", color="yellow")
                continue
            if df is None or df.empty or "delivery_ts" not in df.columns:
                continue
            n_files += 1
            s = _ntc_series(df)
            if s is not None:
                caps[(a, b)] = s
            s = _col_series(df, "scheduled_commercial_exchange")
            if s is not None:
                ex_series[f"{a}-{b}"] = s
    prt(f"  {n_files} border files read: {len(ex_series)} with exchanges, "
        f"{len(caps)} with NTC", color="gray")
    ex = pd.DataFrame(ex_series).sort_index() if ex_series else None
    return ex, caps


def border_caps(caps, window_index):
    def to_list(s):
        if s is None:
            return None
        v = s.reindex(window_index)
        return [None if pd.isna(x) else round(float(x), FLOW_DP) for x in v]

    keys = {tuple(sorted(k)) for k in caps}
    out = {}
    for lo, hi in sorted(keys):
        f = to_list(caps.get((lo, hi)))
        b = to_list(caps.get((hi, lo)))
        has_pos = any((x or 0) > 0 for lst in (f, b) if lst for x in lst)
        if has_pos:
            out[f"{lo}-{hi}"] = {"f": f, "b": b}
    return out


def build_data(download=False, refresh=False):
    migrate_legacy_cache()

    prt("Loading zones ...", color="cyan")
    zones_df = get_zones_df().copy()
    if "broader_county_code" not in zones_df.columns:
        zones_df["broader_county_code"] = np.nan
    zones_df["broader"] = zones_df["broader_county_code"].fillna(zones_df["zone"])
    broader_sizes = Counter(zones_df["broader"])

    zones = []
    for _, row in zones_df.iterrows():
        lon, lat = row["coord_lon"], row["coord_lat"]
        zones.append({
            "code": str(row["zone"]),
            "name": str(row["name"]),
            "lon": float(lon) if pd.notna(lon) else None,
            "lat": float(lat) if pd.notna(lat) else None,
            "broader": str(row["broader"]),
        })
    codes = {z["code"] for z in zones}

    if download or refresh:
        download_assets([z["code"] for z in zones], force=refresh)
    prt(f"Loading zone polygons from {GEO_DIR} (offline) ...", color="cyan")
    geo = load_zone_polygons(zones, broader_sizes)
    prt(f"  polygons found for {len(geo)}/{len(zones)} zones", color="gray")
    unknown = [z["code"] for z in zones if z["code"] not in geo
               and not (GEO_DIR / f"{z['code']}.missing").exists()]
    if unknown and not (download or refresh):
        prt(f"  {len(unknown)} zone(s) have no local polygon file yet "
            f"(drawn as dots). Run with download=True once to fetch them.",
            color="yellow")

    dot_groups = defaultdict(list)
    for z in zones:
        if z["code"] not in geo and z["lon"] is not None:
            dot_groups[(round(z["lon"], 2), round(z["lat"], 2))].append(z)
    for (lon, lat), zs in dot_groups.items():
        n = len(zs)
        if n == 1:
            continue
        for j, z in enumerate(zs):
            a = 2 * math.pi * j / n - math.pi / 2
            z["lon"] = round(lon + 1.6 * math.cos(a), 3)
            z["lat"] = round(lat + 1.1 * math.sin(a), 3)

    prt(f"Loading DAM prices for {len(zones)} zones ...", color="cyan")
    series_map = {}
    for i, z in enumerate(zones):
        if i % 10 == 0:
            prt(f"  {i}/{len(zones)} zones ...", color="gray")
        try:
            s = zone_price_series(z["code"])
        except FileNotFoundError:
            s = None
        except Exception as e:
            prt(f"  {z['code']}: {e}", color="yellow")
            s = None
        if s is not None:
            series_map[z["code"]] = s
    if not series_map:
        raise RuntimeError("get_zone_dam_prices() returned no data at all.")
    prices_df = pd.DataFrame(series_map).sort_index()

    prt("Loading scheduled exchanges and capacity limits (NTC) ...", color="cyan")
    try:
        ex, caps_raw = load_border_data(codes)
    except Exception as e:
        prt(f"  border data unavailable: {e}", color="yellow")
        ex, caps_raw = None, {}

    full_index = prices_df.index
    if ex is not None and len(ex.index):
        full_index = full_index.union(ex.index)
    window = prices_df.reindex(full_index)
    times = [ts.strftime("%Y-%m-%d %H:%M") for ts in window.index]
    prt(f"  prices:    {prices_df.index.min()} -> {prices_df.index.max()}",
        color="gray")
    if ex is not None and len(ex.index):
        prt(f"  exchanges: {ex.index.min()} -> {ex.index.max()}", color="gray")

    flows = net_flows(ex, codes, window.index) if ex is not None else {}
    caps = border_caps(caps_raw, window.index) if caps_raw else {}

    prices = {code: [None if pd.isna(v) else round(float(v), PRICE_DP)
                     for v in window[code]]
              for code in window.columns}

    return {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "flow_sign": FLOW_SIGN,
        "times": times,
        "zones": [{"code": z["code"], "name": z["name"],
                   "lon": z["lon"], "lat": z["lat"]} for z in zones],
        "geo": geo,
        "prices": prices,
        "flows": flows,
        "caps": caps,
    }

def d3_tag():
    try:
        src = read_local(D3_FILE).replace("</", "<\\/")
    except FileNotFoundError:
        raise RuntimeError(
            f"{D3_FILE} not found. Run once with download=True (needs internet) "
            "to save D3 and the map polygons into sample_data/map.")
    return "<script>" + src + "</script>"


def check_offline(html):
    pats = [r"""(?:src|href)\s*=\s*["']?\s*(?:https?:)?//""",
            r"""url\(\s*["']?\s*(?:https?:)?//""",
            r"""@import\s+["']?(?:https?:)?//"""]
    bad = [m.group(0) for p in pats for m in re.finditer(p, html)]
    if bad:
        prt(f"WARNING: generated HTML has external references: {bad[:5]}",
            color="red")
    else:
        prt("Offline check passed: the HTML has no external references.",
            color="green")


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en" data-theme="light">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>European Day-Ahead Prices &amp; Exchanges</title>
<style>
  :root {
    --ink:#1f2f42; --muted:#6b7a8a; --line:#e2e8f0; --panel:#f8fafc; --panel2:#fbfcfe;
    --sea:#d9e8f4; --grat:#c2d6e6; --zstroke:#5a6b7c; --nodata:#dfe4ea;
    --lbl:#222d38; --halo:#ffffff; --arrow:#1c2b3a; --accent:#2563eb;
    --card:rgba(255,255,255,.94); --cardline:#d5dde7; --btn:#ffffff; --btnh:#eef3f9;
    --out:#d73027; --in:#4575b4; --shadow:0 3px 14px rgba(20,40,70,.16);
    --warn:#f59e0b; --cong:#dc2626;
  }
  [data-theme="dark"] {
    --ink:#e6edf5; --muted:#8fa1b3; --line:#223142; --panel:#111c28; --panel2:#0e1822;
    --sea:#0a131c; --grat:#15222f; --zstroke:#4a5c70; --nodata:#243140;
    --lbl:#e4ecf4; --halo:#0a131c; --arrow:#f3f6fb; --accent:#60a5fa;
    --card:rgba(17,28,40,.95); --cardline:#2a3a4c; --btn:#172433; --btnh:#1f3044;
    --shadow:0 3px 14px rgba(0,0,0,.5);
  }
  * { box-sizing:border-box; }
  html, body { height:100%; }
  body { margin:0; overflow:hidden; background:var(--sea); color:var(--ink);
         font-family:"Segoe UI",system-ui,-apple-system,sans-serif; }
  #app { height:100vh; display:flex; flex-direction:column; background:var(--panel2); overflow:hidden; }
  button, select { font:inherit; color:var(--ink); }
  header { display:flex; align-items:center; gap:18px; padding:8px 18px; background:var(--panel); border-bottom:1px solid var(--line); flex:0 0 auto; }
  .titlebox { min-width:0; }
  h1 { margin:0; font-size:16px; font-weight:650; letter-spacing:.2px; white-space:nowrap; }
  .pill { font-size:9.5px; font-weight:700; letter-spacing:.6px; padding:2px 6px; border-radius:99px; background:var(--accent); color:#fff; vertical-align:2px; margin-left:6px; }
  .sub { font-size:11px; color:var(--muted); margin-top:2px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
  #stats { flex:1; display:flex; gap:8px; justify-content:center; flex-wrap:wrap; min-width:0; }
  .chip { display:flex; flex-direction:column; align-items:flex-start; padding:3px 11px; border:1px solid var(--cardline); border-radius:8px; background:var(--btn); font-size:10px; color:var(--muted); cursor:default; line-height:1.25; }
  .chip b { font-size:14px; color:var(--ink); font-variant-numeric:tabular-nums; }
  .chip[data-z] { cursor:pointer; } .chip[data-z]:hover { background:var(--btnh); }
  .right { display:flex; align-items:center; gap:12px; }
  #clock { text-align:right; white-space:nowrap; }
  #clockHour { display:block; font-size:20px; font-weight:650; font-variant-numeric:tabular-nums; line-height:1.1; }
  #clockDate { font-size:10.5px; color:var(--muted); }
  #clockNote { display:block; font-size:10px; font-weight:700; color:#d97706; min-height:12px; }
  .ibtn { border:1px solid var(--cardline); background:var(--btn); border-radius:7px; padding:4px 10px; cursor:pointer; font-size:13px; line-height:1.3; }
  .ibtn:hover { background:var(--btnh); }

  #controls { display:flex; align-items:center; gap:10px; padding:6px 18px; background:var(--panel2); border-bottom:1px solid var(--line); flex:0 0 auto; }
  #playBtn { min-width:84px; }
  .sliderwrap { position:relative; flex:1; padding-bottom:13px; }
  #timeSlider { width:100%; margin:0; accent-color:var(--accent); }
  #ticks { position:absolute; left:8px; right:8px; bottom:0; height:12px; pointer-events:none; }
  #ticks span { position:absolute; transform:translateX(-50%); font-size:9px; color:var(--muted); white-space:nowrap; padding-top:3px; }
  #ticks span::before { content:""; position:absolute; left:50%; top:-1px; height:3px; border-left:1px solid var(--muted); opacity:.6; }
  #rangeLabel { font-size:10.5px; color:var(--muted); white-space:nowrap; }
  select { background:var(--btn); border:1px solid var(--cardline); border-radius:7px; padding:3px 6px; font-size:12px; }

  #mapwrap { position:relative; flex:1 1 auto; min-height:0; background:var(--sea); overflow:hidden; }
  #mapwrap svg { display:block; cursor:grab; }
  #mapwrap svg:active { cursor:grabbing; }
  .sea { fill:var(--sea); }
  .graticule { fill:none; stroke:var(--grat); stroke-width:.6px; vector-effect:non-scaling-stroke; opacity:.85; }
  .zone { stroke:var(--zstroke); stroke-width:.9px; stroke-linejoin:round; vector-effect:non-scaling-stroke; cursor:pointer; transition:fill .25s ease; }
  .zone:hover { stroke:var(--ink); stroke-width:1.9px; }
  .zone.sel { stroke:var(--accent); stroke-width:2.6px; }
  .zlabel { font-size:9.5px; font-weight:650; fill:var(--lbl); paint-order:stroke; stroke:var(--halo); stroke-width:2.6px; stroke-linejoin:round; pointer-events:none; text-anchor:middle; font-variant-numeric:tabular-nums; }
  .zval { font-weight:500; font-size:9px; }
  .flow line { vector-effect:non-scaling-stroke; stroke-linecap:round; }
  .flow line.band { stroke:var(--zstroke); opacity:.3; }
  .flow line.use { stroke:var(--accent); opacity:.92; }
  .flow line.hit { stroke:transparent; stroke-width:12px; pointer-events:stroke; }
  .flow:hover line.band { opacity:.5; }
  .flow .mover path { fill:var(--arrow); stroke:var(--halo); stroke-width:.7px; }
  .flow { cursor:pointer; }

  .tooltip { position:absolute; z-index:20; display:none; pointer-events:none; background:var(--card); border:1px solid var(--cardline); border-radius:8px; padding:7px 11px; font-size:12px; line-height:1.5; box-shadow:var(--shadow); max-width:280px; }
  .tooltip .mut { color:var(--muted); }
  .tooltip .warn { color:var(--warn); font-weight:700; } .tooltip .cong { color:var(--cong); font-weight:700; }
  .overlay { position:absolute; z-index:6; background:var(--card); border:1px solid var(--cardline); border-radius:10px; box-shadow:var(--shadow); }
  .layers { top:12px; left:12px; padding:9px 12px; display:flex; flex-direction:column; gap:7px; font-size:11.5px; }
  .layers .row { display:flex; align-items:center; justify-content:space-between; gap:10px; }
  .layers label.cb { display:flex; align-items:center; gap:6px; cursor:pointer; }
  .zoomctl { position:absolute; top:12px; right:12px; display:flex; flex-direction:column; gap:4px; z-index:6; }
  .zoomctl button { width:30px; height:30px; border:1px solid var(--cardline); background:var(--card); border-radius:7px; font-size:15px; line-height:1; cursor:pointer; }
  .zoomctl button:hover { background:var(--btnh); }

  .legend { left:12px; bottom:12px; display:flex; align-items:center; gap:14px; padding:8px 14px; font-size:11px; color:var(--muted); flex-wrap:wrap; max-width:calc(100% - 24px); }
  .legend .lt { color:var(--ink); font-weight:650; }
  .gradwrap { display:flex; flex-direction:column; gap:2px; width:300px; max-width:40vw; }
  #grad { height:10px; border:1px solid var(--cardline); border-radius:3px; }
  #gradTicks { display:flex; justify-content:space-between; font-size:10px; font-variant-numeric:tabular-nums; }
  .swatch { display:inline-block; width:11px; height:11px; background:var(--nodata); border:1px solid var(--zstroke); border-radius:3px; vertical-align:-2px; margin-right:5px; }
  .swatch.capsw { background:var(--zstroke); opacity:.4; }
  .swatch.usesw { background:var(--accent); }
  .swatch.warnsw { background:var(--warn); } .swatch.congsw { background:var(--cong); }
  #capLegend { white-space:nowrap; }
  #capLegend i { margin-left:8px; }
  #capLegend i:first-child { margin-left:0; }
  .legend .gen { color:var(--muted); opacity:.8; font-size:10px; }

  .panel { top:12px; right:54px; width:340px; max-height:calc(100% - 24px); overflow:auto; padding:12px 14px 14px; font-size:12px; }
  .hidden { display:none !important; }
  .ph { display:flex; justify-content:space-between; align-items:flex-start; gap:8px; }
  .ph b { font-size:16px; } .ph .nm { display:block; color:var(--muted); font-size:11px; margin-top:1px; }
  .ph .btns { display:flex; gap:5px; }
  .pgrid { display:grid; grid-template-columns:1fr 1fr; gap:10px; margin:10px 0 6px; }
  .pgrid label { display:block; font-size:10px; text-transform:uppercase; letter-spacing:.5px; color:var(--muted); }
  .pgrid big { font-size:19px; font-weight:650; font-variant-numeric:tabular-nums; }
  .pgrid small { color:var(--muted); font-size:10.5px; }
  .spark { width:100%; height:auto; display:block; cursor:crosshair; touch-action:none; }
  .spark .area { fill:var(--accent); opacity:.14; } .spark .ln { fill:none; stroke:var(--accent); stroke-width:1.6; }
  .spark .mk { stroke:var(--ink); stroke-width:1; stroke-dasharray:2 2; opacity:.7; } .spark .dot { fill:var(--accent); stroke:var(--halo); stroke-width:1.5; }
  .sparkrange { display:flex; justify-content:space-between; font-size:10px; color:var(--muted); margin-top:1px; font-variant-numeric:tabular-nums; }
  .ptitle { margin:10px 0 4px; font-size:10px; text-transform:uppercase; letter-spacing:.5px; color:var(--muted); }
  .prow { display:grid; grid-template-columns:18px 54px 1fr 112px; align-items:center; gap:6px; padding:2px 0; font-variant-numeric:tabular-nums; }
  .prow .dir.out { color:var(--out); font-weight:700; } .prow .dir.in { color:var(--in); font-weight:700; }
  .prow .dir.zero { color:var(--muted); }
  .prow .bar { height:6px; background:var(--line); border-radius:3px; overflow:hidden; }
  .prow .bar i { display:block; height:100%; border-radius:3px; }
  .prow .bar i.out { background:var(--out); } .prow .bar i.in { background:var(--in); }
  .prow .bar i.warn { background:var(--warn); } .prow .bar i.cong { background:var(--cong); }
  .prow .mw { text-align:right; }
  .prow .mw .cap { color:var(--muted); }
  .muted { color:var(--muted); }
</style>
__D3_TAG__
</head>
<body>
<div id="app">
  <header>
    <div class="titlebox">
      <h1>European Day-Ahead Prices<span class="pill">ENTSO-E</span></h1>
      <div class="sub">Zone colour = DAM price or net position &middot; arrows = scheduled exchanges &middot; grey band = NTC, coloured line = used share &middot; click a zone for details</div>
    </div>
    <div id="stats"></div>
    <div class="right">
      <button id="themeBtn" class="ibtn" title="Toggle light / dark theme">&#9680;</button>
      <div id="clock"><span id="clockHour"></span><span id="clockDate"></span><span id="clockNote"></span></div>
    </div>
  </header>
  <div id="controls">
    <button id="prevBtn" class="ibtn" title="Previous hour (&larr;)">&#8249;</button>
    <button id="playBtn" class="ibtn">&#9654; Play</button>
    <button id="nextBtn" class="ibtn" title="Next hour (&rarr;)">&#8250;</button>
    <div class="sliderwrap">
      <input id="timeSlider" type="range" min="0" max="0" step="1" value="0"/>
      <div id="ticks"></div>
    </div>
    <select id="speed" title="Playback speed">
      <option value="0.5">0.5&times;</option><option value="1" selected>1&times;</option>
      <option value="2">2&times;</option><option value="4">4&times;</option>
    </select>
    <button id="nowBtn" class="ibtn" title="Jump to the latest hour (End)">Latest</button>
    <span id="rangeLabel"></span>
  </div>
  <div id="mapwrap">
    <div class="tooltip" id="tooltip"></div>
    <div class="overlay layers">
      <div class="row"><span>Colour by</span>
        <select id="colorBy"><option value="price">DAM price</option><option value="net">Net position</option></select></div>
      <div class="row"><span>Scale</span>
        <select id="scaleMode"><option value="period">Whole period</option><option value="hour">Current hour</option></select></div>
      <div class="row"><span>Labels</span>
        <select id="labelMode"><option value="auto">Auto</option><option value="codes">Codes</option><option value="values">Codes + values</option><option value="off">Hidden</option></select></div>
      <label class="cb"><input type="checkbox" id="chkArrows" checked/> Exchange arrows</label>
      <label class="cb" id="capRow"><input type="checkbox" id="chkCaps" checked/> Capacity (NTC) bands</label>
    </div>
    <div class="zoomctl">
      <button id="zoomIn" title="Zoom in">+</button>
      <button id="zoomOut" title="Zoom out">&minus;</button>
      <button id="zoomReset" title="Reset view">&#8634;</button>
    </div>
    <div class="overlay panel hidden" id="panel"></div>
    <div class="overlay legend">
      <span class="lt" id="gradTitle">DAM price (&euro;/MWh)</span>
      <div class="gradwrap">
        <div id="grad"></div>
        <div id="gradTicks"><span id="gMin"></span><span id="gMid"></span><span id="gMax"></span></div>
      </div>
      <span><i class="swatch"></i>N/A</span>
      <span id="capLegend"><i class="swatch capsw"></i>NTC <i class="swatch usesw"></i>flow <i class="swatch warnsw"></i>&ge;75% <i class="swatch congsw"></i>&ge;95%</span>
      <span class="gen">generated __GENERATED__ &middot; polygons: entsoe-py &middot; D3.js v7</span>
    </div>
  </div>
</div>
<script>
"use strict";
const DATA = __DATA_JSON__;

/* ------------------------------------------------------------- basics ---- */
const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const zoneBy = {}; DATA.zones.forEach(z => zoneBy[z.code] = z);
const hasLL = z => !!z && z.lon != null && z.lat != null && isFinite(z.lon) && isFinite(z.lat);
const codeOf = d => d.properties ? d.properties.code : d.code;
const times = DATA.times, N = times.length;
const SIGN = DATA.flow_sign || 1;
const caps = DATA.caps || {};
const HAS_CAPS = Object.keys(caps).length > 0;
let tIdx = N - 1;
let showCaps = HAS_CAPS;
if (!HAS_CAPS) { $("capRow").classList.add("hidden"); $("capLegend").classList.add("hidden"); }

const DAYS = ["Sun","Mon","Tue","Wed","Thu","Fri","Sat"];
const MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
const nf0 = new Intl.NumberFormat("en-GB", {maximumFractionDigits: 0});
const fmtMW = v => nf0.format(Math.round(v));
const fmtSigned = v => (v > 0 ? "+" : v < 0 ? "\u2212" : "") + nf0.format(Math.abs(Math.round(v)));
const fmtPrice = v => v.toFixed(2);
const fmtShort = v => Math.abs(v) >= 100 ? nf0.format(Math.round(v)) : (Math.round(v * 10) / 10).toString();
const fmtPct = u => Math.round(u * 100) + "%";

/* ---------------------------------------------------------- flow tables -- */
// borders = every border that has a flow series OR a capacity series
const borderKeys = Array.from(new Set(Object.keys(DATA.flows).concat(Object.keys(caps))));
const flowPairs = borderKeys.map(key => {
  const i = key.lastIndexOf("-");
  return {key: key, a: key.slice(0, i), b: key.slice(i + 1)};
});
const flowAt = (fp, t) => { const a = DATA.flows[fp.key]; return a ? a[t] : null; };
// NTC (MW) per direction: A -> B and B -> A (A < B, same as the flow key)
const ntcAB = (fp, t) => { const c = caps[fp.key]; return c && c.f ? c.f[t] : null; };
const ntcBA = (fp, t) => { const c = caps[fp.key]; return c && c.b ? c.b[t] : null; };

// reference NTC for the band thickness: 95th percentile of all positive values
const capRef = (() => {
  const vals = [];
  for (const key in caps) [caps[key].f, caps[key].b].forEach(arr => {
    if (arr) for (const v of arr) if (v != null && v > 0) vals.push(v);
  });
  vals.sort((x, y) => x - y);
  return Math.max(1, vals.length ? d3.quantile(vals, 0.95) : 1);
})();

// net position per zone and hour (+ = net export, - = net import)
const netAll = {};
DATA.zones.forEach(z => netAll[z.code] = new Array(N).fill(null));
let maxFlow = 1;
flowPairs.forEach(fp => {
  const arr = DATA.flows[fp.key];
  if (!arr) return;
  for (let t = 0; t < N; t++) {
    const v = arr[t]; if (v == null) continue;
    const s = SIGN * v;
    maxFlow = Math.max(maxFlow, Math.abs(v));
    if (netAll[fp.a]) netAll[fp.a][t] = (netAll[fp.a][t] || 0) + s;
    if (netAll[fp.b]) netAll[fp.b][t] = (netAll[fp.b][t] || 0) - s;
  }
});

function utilColor(u) {            // null = default colours
  if (u == null) return null;
  if (u >= 0.95) return "var(--cong)";
  if (u >= 0.75) return "var(--warn)";
  return null;
}

function partnersAt(code, t) {
  const out = [];
  flowPairs.forEach(fp => {
    if (fp.a !== code && fp.b !== code) return;
    const v = flowAt(fp, t);
    const s = v == null ? 0 : SIGN * v;
    const isA = fp.a === code;
    const mw = isA ? s : -s;                              // mw>0: code exports
    const nOut = isA ? ntcAB(fp, t) : ntcBA(fp, t);       // NTC code -> other
    const nIn = isA ? ntcBA(fp, t) : ntcAB(fp, t);        // NTC other -> code
    if (!mw && !(nOut > 0) && !(nIn > 0)) return;
    out.push({other: isA ? fp.b : fp.a, mw: mw, nOut: nOut, nIn: nIn});
  });
  out.sort((x, y) => (Math.abs(y.mw) - Math.abs(x.mw)) ||
    (Math.max(y.nOut || 0, y.nIn || 0) - Math.max(x.nOut || 0, x.nIn || 0)));
  return out;
}

/* --------------------------------------------------------- svg + layers -- */
let W = 1180, H = 760, k = 1;
const wrap = $("mapwrap");
const svg = d3.select("#mapwrap").append("svg");
const sea = svg.append("rect").attr("class", "sea");
const gZoom = svg.append("g");

const projection = d3.geoConicConformal().parallels([40, 62]).rotate([-14, 0]);
const path = d3.geoPath(projection);

const geoFeatures = DATA.zones.filter(z => DATA.geo[z.code])
  .map(z => ({type: "Feature", properties: {code: z.code}, geometry: DATA.geo[z.code]}));
const featBy = {}; geoFeatures.forEach(f => featBy[f.properties.code] = f);
const pointZones = DATA.zones.filter(hasLL);
const fitCol = {type: "FeatureCollection", features: geoFeatures.concat(
  pointZones.map(z => ({type: "Feature", geometry: {type: "Point", coordinates: [z.lon, z.lat]}})))};

const graticule = gZoom.append("path").attr("class", "graticule");
const gZones = gZoom.append("g");
const gFlows = gZoom.append("g");
const gLabels = gZoom.append("g");

gZones.selectAll("path.zone").data(geoFeatures, d => d.properties.code)
  .join("path").attr("class", "zone");

const circleZones = pointZones.filter(z => !DATA.geo[z.code]);
gZones.selectAll("circle.zone").data(circleZones, d => d.code)
  .join("circle").attr("class", "zone");

const labelSel = gLabels.selectAll("g.lab").data(pointZones, d => d.code).join(enter => {
  const g = enter.append("g").attr("class", "lab");
  g.append("text").attr("class", "zlabel zcode").text(d => d.code);
  g.append("text").attr("class", "zlabel zval").attr("dy", "11px");
  return g;
});
const labelXY = d => projection([d.lon, d.lat]);
const labelTransform = d => { const p = labelXY(d); return "translate(" + p[0] + "," + p[1] + ") scale(" + (1 / k) + ")"; };

/* ----------------------------------------------------------- colour scale - */
const PALETTE = ["#313695","#4575b4","#74add1","#abd9e9","#e7f6fa",
                 "#ffffbf","#fee090","#fdae61","#f46d43","#d73027","#a50026"];
const rampPrice = d3.interpolateRgbBasis(PALETTE);
const rampNet = u => d3.interpolateRdBu(1 - u);   // u=1 -> red (export), u=0 -> blue (import)
let colorBy = "price", scaleMode = "period", labelMode = "auto";

const asc = (a, b) => a - b;
const periodPrice = [], periodNet = [];
for (const c in DATA.prices) for (const v of DATA.prices[c]) if (v != null) periodPrice.push(v);
for (const c in netAll) for (const v of netAll[c]) if (v != null) periodNet.push(v);
periodPrice.sort(asc); periodNet.sort(asc);

const valueOf = (code, t) => {
  const a = colorBy === "price" ? DATA.prices[code] : netAll[code];
  return a ? a[t] : null;
};

function domain() {
  let arr;
  if (scaleMode === "hour") {
    arr = DATA.zones.map(z => valueOf(z.code, tIdx)).filter(v => v != null).sort(asc);
  } else {
    arr = colorBy === "price" ? periodPrice : periodNet;
  }
  if (!arr.length) return {lo: 0, hi: 1, min: 0, max: 1};
  const min = arr[0], max = arr[arr.length - 1];
  if (colorBy === "price") {
    let lo = d3.quantile(arr, 0.02), hi = d3.quantile(arr, 0.98);
    if (!(hi - lo > 1e-6)) { lo = min - 1; hi = max + 1; }
    return {lo: lo, hi: hi, min: min, max: max};
  }
  const absArr = arr.map(Math.abs).sort(asc);
  let m = d3.quantile(absArr, 0.98);
  if (!(m > 1e-6)) m = 1;
  return {lo: -m, hi: m, min: min, max: max};
}

let dom = domain();
function colorOf(v) {
  if (v == null) return "var(--nodata)";
  const u = Math.max(0, Math.min(1, (v - dom.lo) / (dom.hi - dom.lo)));
  return (colorBy === "price" ? rampPrice : rampNet)(u);
}

function updateLegend() {
  const ramp = colorBy === "price" ? rampPrice : rampNet;
  const stops = d3.range(0, 1.0001, 0.1).map(u => ramp(u) + " " + Math.round(u * 100) + "%");
  $("grad").style.background = "linear-gradient(to right," + stops.join(",") + ")";
  if (colorBy === "price") {
    $("gradTitle").innerHTML = "DAM price (&euro;/MWh)";
    $("gMin").textContent = (dom.min < dom.lo - 1e-9 ? "\u2264 " : "") + fmtShort(dom.lo);
    $("gMid").textContent = fmtShort((dom.lo + dom.hi) / 2);
    $("gMax").textContent = (dom.max > dom.hi + 1e-9 ? "\u2265 " : "") + fmtShort(dom.hi);
  } else {
    $("gradTitle").innerHTML = "Net position (MW)";
    $("gMin").textContent = "import \u2212" + fmtMW(dom.hi);
    $("gMid").textContent = "0";
    $("gMax").textContent = "export +" + fmtMW(dom.hi);
  }
}

/* ------------------------------------------------------------- tooltip --- */
const tipEl = $("tooltip");
function showTip(html, event) {
  const r = wrap.getBoundingClientRect();
  tipEl.innerHTML = html;
  tipEl.style.display = "block";
  let x = event.clientX - r.left + 16, y = event.clientY - r.top + 12;
  if (x + tipEl.offsetWidth > r.width - 8) x = event.clientX - r.left - tipEl.offsetWidth - 14;
  if (y + tipEl.offsetHeight > r.height - 8) y = event.clientY - r.top - tipEl.offsetHeight - 10;
  tipEl.style.left = Math.max(4, x) + "px";
  tipEl.style.top = Math.max(4, y) + "px";
}
function hideTip() { tipEl.style.display = "none"; }

let rankOf = {}, avgPrice = null, nPriced = 0;
function computeRanks() {
  const list = DATA.zones.map(z => [z.code, (DATA.prices[z.code] || [])[tIdx]]).filter(e => e[1] != null);
  nPriced = list.length;
  avgPrice = list.length ? d3.mean(list, e => e[1]) : null;
  list.sort((x, y) => y[1] - x[1]);
  rankOf = {}; list.forEach((e, i) => rankOf[e[0]] = i + 1);
  return list;
}

function zoneTip(code) {
  const z = zoneBy[code];
  const p = (DATA.prices[code] || [])[tIdx], n = (netAll[code] || [])[tIdx];
  let h = "<b>" + esc(code) + "</b> <span class='mut'>&middot; " + esc(z ? z.name : "") + "</span><br>";
  if (p == null) h += "DAM price: <b>N/A</b><br>";
  else {
    h += "DAM price: <b>" + fmtPrice(p) + "</b> &euro;/MWh";
    if (avgPrice != null) h += " <span class='mut'>(" + (p - avgPrice >= 0 ? "+" : "\u2212") + Math.abs(p - avgPrice).toFixed(1) + " vs avg)</span>";
    h += "<br>";
  }
  if (n != null) h += "Net position: <b>" + fmtSigned(n) + "</b> MW <span class='mut'>" + (n > 0 ? "export" : n < 0 ? "import" : "") + "</span><br>";
  else h += "Net position: <b>N/A</b><br>";
  if (rankOf[code]) h += "<span class='mut'>Price rank " + rankOf[code] + " of " + nPriced + " (1 = most expensive)</span>";
  return h;
}

/* --------------------------------------------------------------- zones ---- */
let selected = null;
gZones.selectAll(".zone")
  .on("mouseenter", function () { d3.select(this).raise(); })
  .on("mousemove", (event, d) => showTip(zoneTip(codeOf(d)), event))
  .on("mouseleave", hideTip)
  .on("click", (event, d) => openPanel(codeOf(d)));

/* --------------------------------------------------------------- arrows --- */
const ARROW = "M -8.5 -4.6 L 10.5 0 L -8.5 4.6 L -3.6 0 Z";
const AMP = 10, DUR = 1400;
const arrowTransform = d => "rotate(" + d.ang + ") scale(" + (1 / k) + ")";

function flowGeom(fp, t) {
  const A = zoneBy[fp.a], B = zoneBy[fp.b];
  if (!hasLL(A) || !hasLL(B)) return null;
  const raw = flowAt(fp, t);
  const hasFlow = raw != null && raw !== 0;
  const v = hasFlow ? SIGN * raw : 0;
  const fwd = v >= 0;                        // true: A -> B (also the default without flow)
  const from = fwd ? A : B, to = fwd ? B : A;
  const nf = ntcAB(fp, t), nb = ntcBA(fp, t);
  let ntc = null;
  if (hasFlow) ntc = fwd ? nf : nb;
  else { const m = Math.max(nf || 0, nb || 0); ntc = m > 0 ? m : null; }
  if (!(ntc > 0)) ntc = null;
  const mag = hasFlow ? Math.abs(raw) : 0;
  const util = hasFlow && ntc ? mag / ntc : null;
  const p0 = projection([from.lon, from.lat]), p1 = projection([to.lon, to.lat]);
  const dx = p1[0] - p0[0], dy = p1[1] - p0[1], L = Math.hypot(dx, dy) || 1;
  const bandW = ntc ? 1.6 + 7 * Math.sqrt(Math.min(1, ntc / capRef)) : 0;
  return {key: fp.key, a: fp.a, b: fp.b, value: raw, hasFlow: hasFlow, mag: mag,
    from: from, to: to, nf: nf, nb: nb, ntc: ntc, util: util,
    x0: p0[0], y0: p0[1], x1: p1[0], y1: p1[1], ux: dx / L, uy: dy / L,
    mx: (p0[0] + p1[0]) / 2, my: (p0[1] + p1[1]) / 2,
    ang: Math.atan2(dy, dx) * 180 / Math.PI,
    bandW: bandW,
    useWCap: ntc ? Math.max(1.2, bandW * Math.min(1, util || 0)) : 0,
    useWOld: 0.8 + 3.4 * Math.sqrt(Math.min(1, mag / maxFlow))};
}

function animate(g, d) {
  g.interrupt("flow");
  const x0 = d.mx - d.ux * AMP, y0 = d.my - d.uy * AMP;
  const x1 = d.mx + d.ux * AMP, y1 = d.my + d.uy * AMP;
  g.attr("transform", "translate(" + x0 + "," + y0 + ")");
  (function loop() {
    if (!g.node() || !g.node().isConnected) return;
    g.transition("flow").duration(DUR).ease(d3.easeLinear)
      .attrTween("transform", () => t =>
        "translate(" + (x0 + (x1 - x0) * t) + "," + (y0 + (y1 - y0) * t) + ")")
      .on("end", loop);
  })();
}

function flowTip(d) {
  const fa = d.a, fb = d.b;
  let h;
  if (d.hasFlow) {
    h = "<b>" + esc(d.from.code) + " &rarr; " + esc(d.to.code) + "</b><br>" +
        "Scheduled exchange: <b>" + fmtMW(d.mag) + "</b> MW<br>";
  } else {
    h = "<b>" + esc(fa) + " &harr; " + esc(fb) + "</b><br><span class='mut'>No scheduled flow</span><br>";
  }
  if (HAS_CAPS && (d.nf != null || d.nb != null)) {
    h += "NTC " + esc(fa) + " &rarr; " + esc(fb) + ": <b>" + (d.nf == null ? "N/A" : fmtMW(d.nf)) + "</b> MW<br>" +
         "NTC " + esc(fb) + " &rarr; " + esc(fa) + ": <b>" + (d.nb == null ? "N/A" : fmtMW(d.nb)) + "</b> MW<br>";
    if (d.util != null) {
      const cls = d.util >= 0.95 ? "cong" : d.util >= 0.75 ? "warn" : "";
      h += "Utilisation: <span class='" + cls + "'>" + fmtPct(d.util) + "</span>" +
           (d.util >= 0.95 ? " <span class='cong'>congested</span>" : "") + "<br>";
    }
    h += "<span class='mut'>NTC = max(coupling, __NTC_FACTOR__ &middot; trader)</span>";
  }
  return h;
}

function updateArrows() {
  const active = [];
  flowPairs.forEach(fp => {
    const g = flowGeom(fp, tIdx);
    if (!g) return;
    if (!g.hasFlow && !(showCaps && g.ntc)) return;
    active.push(g);
  });
  const sel = gFlows.selectAll("g.flow").data(active, d => d.key);
  sel.exit().each(function () { d3.select(this).select(".mover").interrupt("flow"); }).remove();
  sel.enter().append("g").attr("class", "flow")
    .call(g => g.append("line").attr("class", "band"))
    .call(g => g.append("line").attr("class", "use"))
    .call(g => g.append("line").attr("class", "hit"))
    .call(g => g.append("g").attr("class", "mover").append("path").attr("d", ARROW))
    .on("mousemove", (event, d) => showTip(flowTip(d), event))
    .on("mouseleave", hideTip)
    .merge(sel)
    .each(function (d) {
      const g = d3.select(this);
      const useCap = showCaps && !!d.ntc;
      const col = useCap ? utilColor(d.util) : null;
      g.selectAll("line").attr("x1", d.x0).attr("y1", d.y0).attr("x2", d.x1).attr("y2", d.y1);
      g.select("line.band").attr("stroke-width", d.bandW)
        .style("display", useCap ? null : "none");
      g.select("line.use").attr("stroke-width", useCap ? d.useWCap : d.useWOld)
        .style("stroke", col).style("display", d.hasFlow ? null : "none");
      const mover = g.select(".mover");
      mover.style("display", d.hasFlow ? null : "none");
      mover.select("path").attr("transform", arrowTransform(d)).style("fill", col);
      const prev = this.__geom;
      const unchanged = prev && prev.x0 === d.x0 && prev.y0 === d.y0 &&
                        prev.ang === d.ang && prev.hasFlow === d.hasFlow;
      if (!d.hasFlow) mover.interrupt("flow");
      else if (!unchanged) animate(mover, d);
      this.__geom = d;
    });
}

/* -------------------------------------------------------- zoom & pan ----- */
const zoom = d3.zoom().scaleExtent([1, 60]).on("zoom", event => {
  k = event.transform.k;
  gZoom.attr("transform", event.transform);
  restyleForZoom();
});
svg.call(zoom);

function applyLabelMode() {
  gLabels.style("display", labelMode === "off" ? "none" : null);
  const showVals = labelMode === "values" || (labelMode === "auto" && k >= 1.8);
  gLabels.selectAll(".zval").style("display", showVals ? null : "none");
}

function restyleForZoom() {
  gZones.selectAll("circle.zone").attr("r", 9 / k);
  labelSel.attr("transform", labelTransform);
  gFlows.selectAll("g.flow path").attr("transform", d => arrowTransform(d));
  applyLabelMode();
}

d3.select("#zoomIn").on("click", () => svg.transition().duration(250).call(zoom.scaleBy, 1.5));
d3.select("#zoomOut").on("click", () => svg.transition().duration(250).call(zoom.scaleBy, 1 / 1.5));
d3.select("#zoomReset").on("click", () => svg.transition().duration(300).call(zoom.transform, d3.zoomIdentity));

function zoomToZone(code) {
  const z = zoneBy[code]; if (!z) return;
  let cx, cy, s = 4;
  if (featBy[code]) {
    const b = path.bounds(featBy[code]);
    cx = (b[0][0] + b[1][0]) / 2; cy = (b[0][1] + b[1][1]) / 2;
    const dx = b[1][0] - b[0][0], dy = b[1][1] - b[0][1];
    s = Math.max(2, Math.min(10, 0.55 / Math.max(dx / W, dy / H, 1e-6)));
  } else if (hasLL(z)) {
    const p = projection([z.lon, z.lat]); cx = p[0]; cy = p[1];
  } else return;
  svg.transition().duration(700).call(zoom.transform,
    d3.zoomIdentity.translate(W / 2, H / 2).scale(s).translate(-cx, -cy));
}

/* ----------------------------------------------------------- layout ------ */
function layout() {
  const r = wrap.getBoundingClientRect();
  W = Math.max(320, Math.round(r.width));
  H = Math.max(320, Math.round(r.height));
  svg.attr("viewBox", "0 0 " + W + " " + H).attr("width", W).attr("height", H);
  sea.attr("width", W).attr("height", H);
  const PAD = {l: 30, t: 30, r: 30, b: 80};
  projection.fitExtent([[PAD.l, PAD.t], [W - PAD.r, H - PAD.b]], fitCol);
  zoom.extent([[0, 0], [W, H]]).translateExtent([[0, 0], [W, H]]);
  gZones.selectAll("path.zone").attr("d", path);
  gZones.selectAll("circle.zone")
    .attr("cx", d => labelXY(d)[0]).attr("cy", d => labelXY(d)[1]);
  graticule.attr("d", path(d3.geoGraticule().step([10, 10])()));
  svg.call(zoom.transform, d3.zoomIdentity);
  restyleForZoom();
  gFlows.selectAll("g.flow").remove();   // geometry changed: rebuild arrows
  update();
}
window.addEventListener("resize", layout);

/* ----------------------------------------------------------- stats chips - */
function updateStats() {
  const list = computeRanks();           // sorted desc by price
  const box = $("stats");
  let congChip = "";
  if (HAS_CAPS) {
    let nFlow = 0, nCong = 0;
    flowPairs.forEach(fp => {
      const g = flowGeom(fp, tIdx);
      if (g && g.hasFlow) { nFlow++; if (g.util != null && g.util >= 0.95) nCong++; }
    });
    congChip = "<div class='chip' title='Borders with scheduled flow \u2265 95% of NTC'>Congested borders<b>" +
      nCong + " / " + nFlow + "</b></div>";
  }
  if (!list.length) {
    box.innerHTML = "<div class='chip'>Zone average<b>N/A</b></div>" +
      "<div class='chip'>Cheapest<b>N/A</b></div><div class='chip'>Most expensive<b>N/A</b></div>" + congChip;
    return;
  }
  const hi = list[0], lo = list[list.length - 1];
  box.innerHTML =
    "<div class='chip'>Zone average<b>" + fmtPrice(avgPrice) + " &euro;/MWh</b></div>" +
    "<div class='chip' data-z='" + esc(lo[0]) + "' title='Zoom to zone'>Cheapest &middot; " + esc(lo[0]) + "<b>" + fmtPrice(lo[1]) + "</b></div>" +
    "<div class='chip' data-z='" + esc(hi[0]) + "' title='Zoom to zone'>Most expensive &middot; " + esc(hi[0]) + "<b>" + fmtPrice(hi[1]) + "</b></div>" +
    "<div class='chip'>Spread<b>" + (hi[1] - lo[1]).toFixed(1) + "</b></div>" + congChip;
  box.querySelectorAll(".chip[data-z]").forEach(el =>
    el.addEventListener("click", () => { const c = el.dataset.z; openPanel(c); zoomToZone(c); }));
}

/* --------------------------------------------------------- detail panel -- */
const panel = $("panel");
const SW = 300, SH = 64, SP = 4;
const sx = i => SP + i * (SW - 2 * SP) / Math.max(1, N - 1);

function openPanel(code) {
  selected = code;
  const z = zoneBy[code];
  const arr = DATA.prices[code] || [];
  const vals = arr.filter(v => v != null);
  let spark = "<div class='muted' style='margin:8px 0'>No price data for this zone.</div>";
  let ext = null;
  if (vals.length) {
    ext = d3.extent(vals);
    const sy = v => SH - SP - (v - ext[0]) / ((ext[1] - ext[0]) || 1) * (SH - 2 * SP);
    const ln = d3.line().defined(d => d != null).x((d, i) => sx(i)).y(d => sy(d))(arr);
    const ar = d3.area().defined(d => d != null).x((d, i) => sx(i)).y0(SH - SP).y1(d => sy(d))(arr);
    spark = "<svg class='spark' id='spark' viewBox='0 0 " + SW + " " + SH + "'>" +
      "<path class='area' d='" + ar + "'/><path class='ln' d='" + ln + "'/>" +
      "<line class='mk' id='sparkMk' y1='0' y2='" + SH + "'/><circle class='dot' id='sparkDot' r='3.5'/></svg>" +
      "<div class='sparkrange'><span>min " + fmtShort(ext[0]) + "</span><span>" + esc(times[0].slice(0, 10)) +
      " &rarr; " + esc(times[N - 1].slice(0, 10)) + "</span><span>max " + fmtShort(ext[1]) + "</span></div>";
  }
  panel.innerHTML =
    "<div class='ph'><div><b>" + esc(code) + "</b><span class='nm'>" + esc(z ? z.name : "") + "</span></div>" +
    "<div class='btns'><button class='ibtn' id='pZoom' title='Zoom to zone'>&#8982;</button>" +
    "<button class='ibtn' id='pClose' title='Close'>&times;</button></div></div>" +
    "<div class='pgrid'><div><label>DAM price</label><big id='pPrice'>-</big><small> &euro;/MWh</small></div>" +
    "<div><label>Net position</label><big id='pNet'>-</big><small id='pNetLbl'> MW</small></div></div>" +
    spark +
    "<div class='ptitle'>Scheduled exchange" + (HAS_CAPS ? " / NTC" : "") + " this hour</div><div id='pList'></div>";
  panel.classList.remove("hidden");
  $("pClose").onclick = closePanel;
  $("pZoom").onclick = () => zoomToZone(code);
  const sp = $("spark");
  if (sp) {
    const scrub = e => {
      const r = sp.getBoundingClientRect();
      const x = (e.clientX - r.left) / r.width * SW;
      setTime(Math.round((x - SP) / (SW - 2 * SP) * (N - 1)));
    };
    sp.addEventListener("pointerdown", e => { sp.setPointerCapture(e.pointerId); scrub(e); });
    sp.addEventListener("pointermove", e => { if (e.buttons === 1) scrub(e); });
  }
  gZones.selectAll(".zone").classed("sel", d => codeOf(d) === code);
  updatePanel();
}

function closePanel() {
  selected = null;
  panel.classList.add("hidden");
  gZones.selectAll(".zone").classed("sel", false);
}

function updatePanel() {
  if (!selected) return;
  const code = selected;
  const p = (DATA.prices[code] || [])[tIdx], n = (netAll[code] || [])[tIdx];
  $("pPrice").textContent = p == null ? "N/A" : fmtPrice(p);
  $("pNet").textContent = n == null ? "N/A" : fmtSigned(n);
  $("pNetLbl").textContent = n == null ? " MW" : " MW " + (n > 0 ? "export" : n < 0 ? "import" : "");
  const mk = $("sparkMk"), dot = $("sparkDot");
  if (mk) {
    const x = sx(tIdx);
    mk.setAttribute("x1", x); mk.setAttribute("x2", x);
    const vals = (DATA.prices[code] || []).filter(v => v != null);
    if (p != null && vals.length) {
      const e = d3.extent(vals);
      dot.setAttribute("cx", x);
      dot.setAttribute("cy", SH - SP - (p - e[0]) / ((e[1] - e[0]) || 1) * (SH - 2 * SP));
      dot.style.display = null;
    } else dot.style.display = "none";
  }
  const parts = partnersAt(code, tIdx);
  const mx = parts.length ? Math.max(1, Math.abs(parts[0].mw)) : 1;
  $("pList").innerHTML = parts.length ? parts.slice(0, 12).map(r => {
    const out = r.mw > 0, zero = r.mw === 0;
    // NTC in the direction of the flow (both directions' max when idle)
    const ntc = zero ? Math.max(r.nOut || 0, r.nIn || 0) : (out ? r.nOut : r.nIn);
    const util = ntc > 0 ? Math.abs(r.mw) / ntc : null;
    const cls = util != null && util >= 0.95 ? "cong" : util != null && util >= 0.75 ? "warn" : (out ? "out" : "in");
    const width = util != null ? Math.min(100, util * 100) : Math.abs(r.mw) / mx * 100;
    return "<div class='prow'><span class='dir " + (zero ? "zero" : out ? "out" : "in") + "'>" +
      (zero ? "&middot;" : out ? "&rarr;" : "&larr;") + "</span>" +
      "<b>" + esc(r.other) + "</b><span class='bar'><i class='" + cls + "' style='width:" +
      (zero ? 0 : Math.max(3, width)) + "%'></i></span><span class='mw'>" + fmtMW(Math.abs(r.mw)) +
      (ntc > 0 ? " <span class='cap'>/ " + fmtMW(ntc) + "</span>" : "") + " MW</span></div>";
  }).join("") : "<div class='muted'>No scheduled exchanges.</div>";
}

/* ------------------------------------------------------------- update ---- */
function updateLabelValues() {
  labelSel.select(".zval").text(d => {
    const v = valueOf(d.code, tIdx);
    if (v == null) return "N/A";
    return colorBy === "price" ? fmtShort(v) : fmtSigned(v);
  });
}

function update() {
  const t = times[tIdx];
  $("clockHour").textContent = t.slice(11, 16);
  const d = new Date(t.slice(0, 10) + "T" + t.slice(11, 16));
  $("clockDate").textContent = DAYS[d.getDay()] + " " + d.getDate() + " " +
    MONTHS[d.getMonth()] + " " + d.getFullYear() + " \u00b7 delivery hour";
  const hasP = DATA.zones.some(z => ((DATA.prices[z.code] || [])[tIdx]) != null);
  const hasF = flowPairs.some(fp => flowAt(fp, tIdx) != null);
  $("clockNote").textContent = (!hasP && !hasF) ? "Prices N/A \u00b7 Exchanges N/A"
    : !hasP ? "Prices N/A" : !hasF ? "Exchanges N/A" : "";

  dom = domain();
  gZones.selectAll(".zone").style("fill", d => colorOf(valueOf(codeOf(d), tIdx)));
  updateLegend();
  updateStats();
  updateLabelValues();
  if ($("chkArrows").checked) updateArrows();
  updatePanel();
}

/* ------------------------------------------------------ time controls ---- */
const slider = $("timeSlider");
slider.min = 0; slider.max = N - 1; slider.value = tIdx;
$("rangeLabel").textContent = times[0] + "  \u2192  " + times[N - 1];

(function buildTicks() {
  // With all available data the range can span months/years: label midnights
  // for short ranges, first-of-month for long ones, and cap the label count.
  const ticks = $("ticks");
  const spanDays = N / 24;
  const monthly = spanDays > 90;
  const idxs = [];
  times.forEach((t, i) => {
    if (i === 0 || i === N - 1) return;
    if (monthly ? t.slice(8) === "01 00:00" : t.endsWith("00:00")) idxs.push(i);
  });
  const every = Math.max(1, Math.ceil(idxs.length / 12));
  idxs.forEach((i, j) => {
    if (j % every) return;
    const d = new Date(times[i].slice(0, 10) + "T00:00");
    const el = document.createElement("span");
    el.style.left = (i / Math.max(1, N - 1) * 100) + "%";
    el.textContent = monthly ? MONTHS[d.getMonth()] + " " + String(d.getFullYear()).slice(2)
                             : DAYS[d.getDay()] + " " + d.getDate();
    ticks.appendChild(el);
  });
})();

function setTime(i) {
  stopPlay();
  tIdx = Math.max(0, Math.min(N - 1, i));
  slider.value = tIdx;
  update();
}
slider.addEventListener("input", () => setTime(+slider.value));

let timer = null;
const playBtn = $("playBtn");
function stopPlay() {
  if (timer) { clearInterval(timer); timer = null; playBtn.innerHTML = "&#9654; Play"; }
}
function startPlay() {
  stopPlay();
  playBtn.innerHTML = "&#10074;&#10074; Pause";
  timer = setInterval(() => {
    tIdx = (tIdx + 1) % N;
    slider.value = tIdx;
    update();
  }, 650 / +$("speed").value);
}
playBtn.addEventListener("click", () => timer ? stopPlay() : startPlay());
$("speed").addEventListener("change", () => { if (timer) startPlay(); });
$("prevBtn").addEventListener("click", () => setTime(tIdx - 1));
$("nextBtn").addEventListener("click", () => setTime(tIdx + 1));
$("nowBtn").addEventListener("click", () => setTime(N - 1));

document.addEventListener("keydown", e => {
  const tag = e.target.tagName;
  if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA") return;
  if (e.key === "ArrowLeft") { setTime(tIdx - 1); e.preventDefault(); }
  else if (e.key === "ArrowRight") { setTime(tIdx + 1); e.preventDefault(); }
  else if (e.key === "Home") { setTime(0); e.preventDefault(); }
  else if (e.key === "End") { setTime(N - 1); e.preventDefault(); }
  else if (e.key === " " && tag !== "BUTTON") { timer ? stopPlay() : startPlay(); e.preventDefault(); }
  else if (e.key === "Escape") closePanel();
});

/* ------------------------------------------------------- option wiring --- */
$("colorBy").addEventListener("change", e => { colorBy = e.target.value; update(); });
$("scaleMode").addEventListener("change", e => { scaleMode = e.target.value; update(); });
$("labelMode").addEventListener("change", e => { labelMode = e.target.value; applyLabelMode(); });
$("chkArrows").addEventListener("change", e => {
  gFlows.style("display", e.target.checked ? null : "none");
  if (e.target.checked) updateArrows();
});
$("chkCaps").addEventListener("change", e => {
  showCaps = e.target.checked;
  $("capLegend").classList.toggle("hidden", !showCaps);
  if ($("chkArrows").checked) updateArrows();
});

/* --------------------------------------------------------------- theme --- */
function setTheme(th) {
  document.documentElement.dataset.theme = th;
  try { localStorage.setItem("dam-map-theme", th); } catch (e) {}
}
(function initTheme() {
  let th = null;
  try { th = localStorage.getItem("dam-map-theme"); } catch (e) {}
  if (!th) th = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  document.documentElement.dataset.theme = th;
})();
$("themeBtn").addEventListener("click", () =>
  setTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark"));

layout();
</script>
</body>
</html>
"""


def build_html(data):
    payload = json.dumps(data, ensure_ascii=False, allow_nan=False)
    payload = payload.replace("</", "<\\/")
    return (HTML_TEMPLATE
            .replace("__D3_TAG__", d3_tag())
            .replace("__DATA_JSON__", payload)
            .replace("__NTC_FACTOR__", f"{NTC_TRADER_FACTOR:g}")
            .replace("__GENERATED__", data["generated"]))

def generate_map_html(download=False, refresh=False, open_browser=False):
    prt("=" * 68, color="cyan")
    prt("Generating the ENTSO-E-style DAM price map ...", color="cyan")
    data = build_data(download=download, refresh=refresh)

    prt(f"zones: {len(data['zones'])} | with polygons: {len(data['geo'])} "
        f"| with prices: {len(data['prices'])} | borders with flows: {len(data['flows'])} "
        f"| borders with NTC: {len(data['caps'])} | hours: {len(data['times'])}",
        color="gray")
    if data["times"]:
        prt(f"time window: {data['times'][0]}  ->  {data['times'][-1]}", color="gray")

    html = build_html(data)
    check_offline(html)
    OUTPUT_HTML.write_text(html, encoding="utf-8")
    prt(f"Done -> {OUTPUT_HTML} ({len(html) / 1e6:.2f} MB). Open it in any browser.",
        color="green")
    prt("=" * 68, color="cyan")

    if open_browser:
        import webbrowser
        webbrowser.open(OUTPUT_HTML.as_uri())


if __name__ == "__main__":
    generate_map_html(download=False, refresh=False, open_browser=True)