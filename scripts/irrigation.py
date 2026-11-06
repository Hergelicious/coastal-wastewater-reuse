"""Irrigation substitution matched locally. Ocean-outfall plants in the same country whose buffers of radius R overlap
(outfalls no more than 2R apart) form a cluster. In each cluster the replaceable freshwater withdrawal is the smaller of
(a) the cluster's irrigation-grade supply and (b) the irrigated area (GMIA v5 area equipped for irrigation) inside the
union of the cluster's buffers times the gross irrigation requirement. Supply cannot move between clusters.
Requirement per plant: Hargreaves radiation formula ET0 = 0.0135 (T + 17.8) Rs, with Rs converted from NASA POWER
kWh m-2 d-1 to MJ m-2 d-1 (x 3.6) and then to mm d-1 of evaporation (/ 2.45 MJ kg-1); net = max(Kc ET0 - Peff, 0) per
month, Peff by the USDA SCS method; gross = net / 0.65. Kc 1.0 central, 0.8 and 1.2 as low and high (span of FAO-56 mid-season
coefficients from tree crops to cereals), R 25 km central, 10 and 50 km as sensitivity.
Writes out/irrigation_summary.csv and out/irrigation_countries.csv."""
import sys, json, pathlib, numpy as np, pandas as pd
ROOT = pathlib.Path(sys.argv[1]); D = ROOT / "data"; OUT = ROOT / "out"
MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
DAYS = np.array([31, 28.25, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]); EFF = 0.65
KCS = (0.8, 1.0, 1.2); RADII = (10, 25, 50)

p = pd.read_csv(OUT / "plants_model.csv")
o = p[p.ocean_outfall & (p.irr_m3d > 0)].reset_index(drop=True)

# ---- requirement at each Kc, from the cached NASA POWER cells ----
def req(lat, lon):
    f = D / "power_cells" / f"{lat:.1f}_{lon:.1f}.json"
    try:
        par = json.load(open(f))["properties"]["parameter"]
        g = np.array([par["ALLSKY_SFC_SW_DWN"][m] for m in MONTHS], float)
        t = np.array([par["T2M"][m] for m in MONTHS], float); pr = np.array([par["PRECTOTCORR"][m] for m in MONTHS], float) * DAYS
    except Exception:
        return [np.nan] * len(KCS)
    if (g < 0).any() or (t < -90).any() or (pr < 0).any(): return [np.nan] * len(KCS)
    et0 = 0.0135 * (t + 17.8) * g * 3.6 / 2.45 * DAYS                     # mm per month
    peff = np.where(pr <= 250, pr * (125 - 0.2 * pr) / 125, 125 + 0.1 * pr)
    return [np.clip(k * et0 - peff, 0, None).sum() * 10 / EFF for k in KCS]   # m3 per ha per yr
cells = o[["cell_lat", "cell_lon"]].drop_duplicates()
R = {(a, b): req(a, b) for a, b in zip(cells.cell_lat, cells.cell_lon)}
for i, k in enumerate(KCS):
    o[f"req_{k}"] = [R[(a, b)][i] for a, b in zip(o.cell_lat, o.cell_lon)]
for k in KCS:   # cells without climate: scale the plant's filled Kc 1.0 requirement by the median ratio
    ratio = (o[f"req_{k}"] / o["req_1.0"]).median()
    o[f"req_{k}"] = o[f"req_{k}"].fillna(o.irr_req_m3_ha * ratio)
o["req_1.0"] = o.irr_req_m3_ha   # identical to build_plants.py where climate exists; keeps its filled values elsewhere

# ---- GMIA v5 irrigated area ----
def read_asc(path):
    hdr = {}
    with open(path) as fh:
        for _ in range(6):
            k, v = fh.readline().split(); hdr[k.lower()] = float(v)
    a = np.loadtxt(path, skiprows=6, dtype=np.float32); a[a == hdr["nodata_value"]] = np.nan
    return a, hdr
aei, h = read_asc(D / "gmia/gmia_v5_aei_ha.asc"); cs = h["cellsize"]; nr, nc = aei.shape
lat_c = 90 - (np.arange(nr) + 0.5) * cs; lon_c = -180 + (np.arange(nc) + 0.5) * cs
def hav(lat, lon, la, lo):
    return 2 * 6371 * np.arcsin(np.sqrt(np.sin(np.radians(la - lat) / 2) ** 2 + np.cos(np.radians(lat)) * np.cos(np.radians(la)) * np.sin(np.radians(lo - lon) / 2) ** 2))
def window(lat, lon, km):
    dl = km / 111.0; dlo = km / (111.0 * max(np.cos(np.radians(lat)), 0.05))
    r0 = max(int((90 - lat - dl) / cs), 0); r1 = min(int((90 - lat + dl) / cs), nr - 1)
    c0 = max(int((lon - dlo + 180) / cs), 0); c1 = min(int((lon + dlo + 180) / cs), nc - 1)
    return r0, c0, hav(lat, lon, lat_c[r0:r1 + 1][:, None], lon_c[c0:c1 + 1][None, :]) <= km

def clusters(radius):
    """Union-find over outfalls of the same country no more than 2 x radius apart."""
    par = np.arange(len(o))
    def find(i):
        while par[i] != i: par[i] = par[par[i]]; i = par[i]
        return i
    for _, g in o.groupby("CNTRY_ISO"):
        idx = g.index.values; la = g.LAT_OUT.values; lo = g.LON_OUT.values
        for j in range(len(idx)):
            near = idx[j + 1:][hav(la[j], lo[j], la[j + 1:], lo[j + 1:]) <= 2 * radius]
            for q in near:
                a, b = find(idx[j]), find(q)
                if a != b: par[b] = a
    return np.array([find(i) for i in range(len(o))])

rows, ctry = [], None
for radius in RADII:
    o["cl"] = clusters(radius)
    owner = {}
    for r in o.itertuples():
        r0, c0, m = window(r.LAT_OUT, r.LON_OUT, radius)
        for ri, ci in zip(*np.where(m)): owner.setdefault((r0 + ri, c0 + ci), r.cl)   # each grid cell counted once
    area = {}
    for (ri, ci), cl in owner.items():
        v = aei[ri, ci]
        if np.isfinite(v): area[cl] = area.get(cl, 0.0) + float(v)
    o["sup"] = o.irr_m3d * 365; o["sup_s"] = np.where(o.stressed, o.sup, 0.0)
    for k in KCS:
        g = o.groupby("cl").apply(lambda d: pd.Series(dict(
            iso=d.CNTRY_ISO.iloc[0], sup=d.sup.sum(), sup_s=d.sup_s.sum(),
            req=np.average(d[f"req_{k}"], weights=d.sup.clip(lower=1)))), include_groups=False)
        g["aei_ha"] = pd.Series(area).reindex(g.index).fillna(0)
        g["dem"] = g.aei_ha * g.req
        g["rep"] = np.minimum(g.sup, g.dem); g["rep_s"] = np.minimum(g.sup_s, g.dem)
        rows.append(dict(radius_km=radius, kc=k, n_clusters=len(g), supply_km3=g.sup.sum() / 1e9, demand_km3=g.dem.sum() / 1e9,
                         aei_Mha=g.aei_ha.sum() / 1e6, replaceable_km3=g.rep.sum() / 1e9, replaceable_stressed_km3=g.rep_s.sum() / 1e9,
                         share_supply_used=g.rep.sum() / g.sup.sum()))
        if radius == 25 and k == 1.0:
            ctry = g.groupby("iso")[["sup", "sup_s", "dem", "rep", "rep_s", "aei_ha"]].sum() / [1e9, 1e9, 1e9, 1e9, 1e9, 1]
            ctry.columns = ["supply_km3", "supply_stressed_km3", "demand_km3", "replaceable_km3", "replaceable_stressed_km3", "aei_25km_ha"]
S = pd.DataFrame(rows)
# national pooling (the previous method) for comparison, at R 25 km and Kc 1.0
cs_ = pd.read_csv(OUT / "country_summary.csv", index_col=0)
S = pd.concat([S, pd.DataFrame([dict(radius_km=25, kc=1.0, n_clusters=np.nan, supply_km3=cs_.irr_supply_km3.sum(),
                                     demand_km3=cs_.irr_demand_km3.sum(), replaceable_km3=cs_.irr_replaceable_km3.sum(),
                                     replaceable_stressed_km3=cs_.irr_replaceable_stressed_km3.sum(), method="national pooling")])])
S["method"] = S.get("method").fillna("cluster")
S.to_csv(OUT / "irrigation_summary.csv", index=False)
ctry.join(cs_[["country"]]).sort_values("replaceable_km3", ascending=False).to_csv(OUT / "irrigation_countries.csv")
pd.set_option("display.width", 200)
print(S.round(2).to_string()); print(ctry.sort_values("replaceable_km3", ascending=False).head(12).round(2))
