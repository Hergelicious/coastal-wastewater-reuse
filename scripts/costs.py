"""Break-even conveyance distance between potable reclamation and seawater reverse osmosis (SWRO).

D* = (LCOW_swro - LCOW_reuse) / t, with LCOW = non-energy cost + specific energy x electricity price,
and t the cost of moving 1 m3 of water over 1 km. Reclamation is cheaper than SWRO for any delivery distance
below D* (screening estimate; pumping head, storage and permitting are not included). Independent draws (N = 20,000, seed 42):
  SWRO non-energy cost      triangular 0.35, 0.55, 0.85 $/m3   (Ghaffour et al. 2013: total 0.5-1.2 $/m3, energy 19-40%)
  reuse non-energy cost     triangular 0.25, 0.37, 0.50 $/m3   (Giammar et al. 2022: LCOW 0.40-0.61 $/m3 incl. 0.13-0.17 electricity)
  SWRO energy               triangular 3.0, 3.5, 4.0 kWh/m3    (as model.py)
  reuse energy              triangular 0.8, 1.0, 1.5 kWh/m3    (as model.py)
  electricity price p       uniform 0.05 to 0.20 $/kWh         (EIA US industrial 0.08; Eurostat non-household 0.19 EUR; low-tariff case 0.05)
  conveyance cost t         log-uniform 0.06 to 2.0 $/m3 per 100 km
     bulk: 0.06 $/m3 per 100 km horizontal transport (Zhou and Tol 2005, Water Resour. Res. 41:W03003)
     urban pipeline: 1.0 $/m3 per 100 km, derived from the Carlsbad 10-mile (16 km) pipeline, 163 M$ capital for
       189,000 m3/d (Cooley and Ajami 2012), annualized at 5% over 30 years plus 1% of capital per year O&M; 2.0 = twice that
Also reports the flow-weighted distance between eligible ocean-outfall plants and the centres of the matched cities.
Writes out/breakeven.csv, out/breakeven_by_price.csv, out/breakeven_cities.csv."""
import sys, pathlib, numpy as np, pandas as pd
ROOT = pathlib.Path(sys.argv[1]); OUT = ROOT / "out"
rng = np.random.default_rng(42); N = 20000
tri = lambda a, m, b: rng.triangular(a, m, b, N)
# urban pipeline cost per m3 per km from the Carlsbad conveyance pipeline
CAPEX, KM, FLOW, R, YRS, OM = 163e6, 16.1, 189000, 0.05, 30, 0.01
crf = R * (1 + R) ** YRS / ((1 + R) ** YRS - 1)
T_URBAN = CAPEX * (crf + OM) / (FLOW * 365) / KM          # $/m3/km
T_BULK, T_HIGH = 0.0006, 2 * T_URBAN
sw_ne, re_ne = tri(0.35, 0.55, 0.85), tri(0.25, 0.37, 0.50)
e_sw, e_re = tri(3.0, 3.5, 4.0), tri(0.8, 1.0, 1.5)
p = rng.uniform(0.05, 0.20, N)
t = np.exp(rng.uniform(np.log(T_BULK), np.log(T_HIGH), N))
d_lcow = (sw_ne + e_sw * p) - (re_ne + e_re * p)
D = np.where(d_lcow > 0, d_lcow / t, 0.0)
q = lambda x: np.percentile(x, [5, 50, 95])
rows = [("urban_pipeline_cost_usd_m3_per_100km", T_URBAN * 100, np.nan, np.nan), ("delta_lcow_usd_m3", *q(d_lcow)), ("breakeven_km", *q(D)),
        ("breakeven_km_urban_pipeline", *q(np.where(d_lcow > 0, d_lcow / T_URBAN, 0))),
        ("breakeven_km_twice_urban", *q(np.where(d_lcow > 0, d_lcow / T_HIGH, 0))),
        ("share_draws_reuse_cheaper_at_0km", (d_lcow > 0).mean(), np.nan, np.nan)]
for km in (10, 25, 50, 100):
    rows.append((f"P(D*>{km}km)", (D > km).mean(), np.nan, np.nan))
pd.DataFrame(rows, columns=["metric", "p05", "p50", "p95"]).to_csv(OUT / "breakeven.csv", index=False)
# by electricity price: urban pipeline and twice the urban pipeline cost
by = []
for pp in np.round(np.arange(0.05, 0.201, 0.025), 3):
    dl = (sw_ne + e_sw * pp) - (re_ne + e_re * pp)
    by.append((pp, *q(dl / T_URBAN), *q(dl / T_HIGH)))
pd.DataFrame(by, columns=["price", "urban_p05", "urban_p50", "urban_p95", "high_p05", "high_p50", "high_p95"]).to_csv(OUT / "breakeven_by_price.csv", index=False)
# distance of matched ocean-outfall plants to the city centre
pl = pd.read_csv(OUT / "plants_model.csv"); cities = pd.read_csv(OUT / "cities.csv")
def hav(lat, lon, la, lo):
    return 2 * 6371 * np.arcsin(np.sqrt(np.sin(np.radians(la - lat) / 2) ** 2 + np.cos(np.radians(lat)) * np.cos(np.radians(la)) * np.sin(np.radians(lo - lon) / 2) ** 2))
o = pl[pl.ocean_outfall & (pl.LEVEL != "Primary") & (pl.WASTE_DIS >= 10000)]; rec = []   # eligible plants, as cities.py
for c in cities.itertuples():
    if c.feedwater.startswith("brackish"): continue                            # London: not compared
    sel = hav(c.lat, c.lon, o.LAT_OUT.values, o.LON_OUT.values) <= 50          # same plant set as cities.py
    d = hav(c.lat, c.lon, o.LAT_WWTP.values, o.LON_WWTP.values); m = sel & (o.CNTRY_ISO.values == c.iso)   # distance from the plant
    if m.sum() == 0: continue
    w = o.pot_m3d.values[m]; dd = d[m]; srt = np.argsort(dd); cw = np.cumsum(w[srt]) / w.sum()
    rec.append((c.city, float(dd[srt][np.searchsorted(cw, 0.5)]), float((w * dd).sum() / w.sum()), float(w[dd <= 25].sum() / w.sum())))
R = pd.DataFrame(rec, columns=["city", "wmedian_km", "wmean_km", "share_within_25km"])
R.to_csv(OUT / "breakeven_cities.csv", index=False)
print(pd.read_csv(OUT / "breakeven.csv").round(3).to_string()); print(pd.read_csv(OUT / "breakeven_by_price.csv").round(1).to_string())
print(R.round(2).to_string()); print("all-city flow-weighted median km", R.wmedian_km.median().round(1))
