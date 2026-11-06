"""Uncertainty of the city substitution ratio (net potable product of eligible ocean-outfall plants / seawater
desalination capacity) and an approximate route distance for the cities with ratio >= 0.25.
Monte Carlo, N = 20,000 (seed 11):
  feed fraction and potable recovery: triangular, as model.py;
  plant flow: reported or population-based flows (tiers A and B) times a factor drawn from the ratios of utility-reported
    average flow or design capacity to HydroWASTE flow for the validated plants that exist (inputs/validation_top25.csv; Psyttalia excluded because its
    flow was replaced); one draw per city, applied to all its plants (fully correlated within a city);
    modelled flows (tier C) times one country ratio drawn, weighted by reported flow, from out/ml_country_ratios.csv (r_centred);
  existing reuse: national reuse fraction times a factor uniform 0.5 to 1.5, capped at 1.
Fixed scenarios: conservative (feed 0.80, recovery 0.70, flow x 0.79 = median ratio of average to reported flow in the
validation, reuse x 1.5); central; optimistic (feed 0.95, recovery 0.85, flow x 1, reuse x 0.5).
Route distance: from each eligible plant to the nearest desalination plant serving the city (the point where product
enters the network), great-circle distance times a detour factor of 1.3 (1.2 to 1.6), compared with the break-even
distance at the urban pipeline cost (out/breakeven.csv).
Writes out/cities_uncertainty.csv and out/cities_route.csv."""
import sys, pathlib, numpy as np, pandas as pd
ROOT = pathlib.Path(sys.argv[1]); OUT = ROOT / "out"; N = 20000; rng = np.random.default_rng(11)
RADIUS, SIZE_MIN = 50, 10000
T = pd.read_csv(OUT / "cities.csv"); p = pd.read_csv(OUT / "plants_model.csv")
v = pd.read_csv(ROOT / "inputs/validation_top25.csv")
v["r"] = pd.to_numeric(v.reported_m3d, errors="coerce") / v.hydrowaste_m3d
VR = v[(v.plant_exists == "yes") & v.r.notna() & (v.WASTE_ID != 14326)].r.values
_cr = pd.read_csv(OUT / "ml_country_ratios.csv"); CR = _cr.r_centred.values; CW = (_cr.obs / _cr.obs.sum()).values   # flow-weighted, as ml_validate.py
def tri(lo, c, hi): return rng.triangular(lo, c, hi, N)
f, rec = tri(0.80, 0.90, 0.95), tri(0.70, 0.80, 0.85)
def hav(lat, lon, la, lo):
    return 2 * 6371 * np.arcsin(np.sqrt(np.sin(np.radians(la - lat) / 2) ** 2 + np.cos(np.radians(lat)) * np.cos(np.radians(la)) * np.sin(np.radians(lo - lon) / 2) ** 2))
def eligible(r):
    d = hav(r.lat, r.lon, p.LAT_OUT.values, p.LON_OUT.values)
    return p[(d <= RADIUS) & (p.CNTRY_ISO == r.iso) & p.ocean_outfall & (p.LEVEL != "Primary") & (p.WASTE_DIS >= SIZE_MIN)]
rows = []
for r in T[~T.feedwater.str.startswith("brackish")].itertuples():
    e = eligible(r); q = r.reuse_frac if np.isfinite(r.reuse_frac) else 0.0
    qA = e[e.tier != "C"].WASTE_DIS.sum(); qC = e[e.tier == "C"].WASTE_DIS.sum()
    fl = qA * rng.choice(VR, N) + qC * rng.choice(CR, N, p=CW)
    reuse = np.minimum(q * rng.uniform(0.5, 1.5, N), 1.0)
    ratio = fl * f * rec * (1 - reuse) / r.desal_m3d
    sc = lambda ff, rr, fx, rm: (qA + qC) * fx * ff * rr * (1 - min(q * rm, 1.0)) / r.desal_m3d
    rows.append(dict(city=r.city, desal_m3d=r.desal_m3d, n_eligible=len(e), ratio_central=r.ratio_net,
                     ratio_p05=np.percentile(ratio, 5), ratio_p50=np.median(ratio), ratio_p95=np.percentile(ratio, 95),
                     p_ratio_ge1=(ratio >= 1).mean(), p_ratio_ge025=(ratio >= 0.25).mean(),
                     ratio_conservative=sc(0.80, 0.70, 0.79, 1.5), ratio_optimistic=sc(0.95, 0.85, 1.0, 0.5),
                     tierA_B_share=qA / max(qA + qC, 1)))
U = pd.DataFrame(rows)
cls = lambda x: np.select([x >= 1, x >= 0.25], ["Reclamation-first", "Complementary"], "Desalination-first")
for k in ["conservative", "central", "optimistic"]: U[f"class_{k}"] = cls(U[f"ratio_{k}"])
U.sort_values("ratio_central", ascending=False).to_csv(OUT / "cities_uncertainty.csv", index=False)

# ---- approximate route distance for cities with central ratio >= 0.25 ----
DESAL_XY = {  # desalination plant locations (operator pages, Wikipedia and OpenStreetMap)
    "Tampa": [(27.796, -82.404)], "Barcelona": [(41.297, 2.124)], "San Diego": [(33.139, -117.335)],
    "Chennai": [(13.317, 80.338), (12.704, 80.226)], "Sydney": [(-34.025, 151.205)], "Gold Coast": [(-28.157, 153.497)],
    "Singapore": [(1.290, 103.626), (1.302, 103.623), (1.293, 103.876), (1.276, 103.666)]}
be = pd.read_csv(OUT / "breakeven.csv", index_col=0)
d_u, d_2 = be.loc["breakeven_km_urban_pipeline", "p50"], be.loc["breakeven_km_twice_urban", "p50"]
R2 = []
for r in T[T.city.isin(DESAL_XY)].itertuples():
    e = eligible(r).copy(); w = e.pot_net_m3d.values
    gc = np.min([hav(la, lo, e.LAT_WWTP.values, e.LON_WWTP.values) for la, lo in DESAL_XY[r.city]], axis=0)
    gcc = hav(r.lat, r.lon, e.LAT_WWTP.values, e.LON_WWTP.values)
    for det in (1.2, 1.3, 1.6):
        rd = gc * det
        R2.append(dict(city=r.city, detour=det, wmean_route_km=np.average(rd, weights=w) if w.sum() else np.nan,
                       max_route_km=rd.max() if len(rd) else np.nan, centre_wmean_km=np.average(gcc, weights=w) if w.sum() else np.nan,
                       share_within_urban_breakeven=w[rd <= d_u].sum() / max(w.sum(), 1),
                       share_within_twice_urban_breakeven=w[rd <= d_2].sum() / max(w.sum(), 1),
                       ratio_within_urban_breakeven=w[rd <= d_u].sum() / r.desal_m3d,
                       ratio_within_twice_urban_breakeven=w[rd <= d_2].sum() / r.desal_m3d))
R2 = pd.DataFrame(R2); R2.to_csv(OUT / "cities_route.csv", index=False)
pd.set_option("display.width", 250)
print(U.round(2).to_string()); print(R2.round(2).to_string()); print("validation flow ratios:", np.round(np.sort(VR), 2))
