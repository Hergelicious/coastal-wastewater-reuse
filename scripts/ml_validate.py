"""Flow-estimate validation and country-correlated bootstrap.

A gradient-boosted model of log10 per-capita flow is trained on HydroWASTE plants with reported flow (QUAL_WASTE = 1)
and tested under three cross-validation designs: random 5-fold, grouped by country (GroupKFold, 5 folds), and grouped
by HydroWASTE source dataset. Grouped designs use a parsimonious feature set (log population, treatment level, log GDP
per capita), because a held-out country has no country median and source or location features do not transfer, so
the test mimics prediction for a country without reported flows.
Skill: R2 of log10 plant flow, median absolute percentage error of plant flow, ratio of summed predicted to summed
reported flow, 80% prediction-interval coverage; a country-median per-capita baseline is reported alongside.

Country-correlated bootstrap: country-grouped CV gives, for every held-out country, the ratio of summed reported to
summed predicted flow (r_c). Each of B = 2,000 draws assigns every country an r drawn from that empirical set
(weighted by country flow, rescaled so that the flow-weighted mean ratio is 1) and scales estimated flows (QUAL_WASTE = 4) by it; reported flows are kept. The ratio of
the scaled to the original scope total is written to out/flow_bootstrap.csv for model.py.
Also writes out/ml_skill.csv, out/ml_country_ratios.csv and out/flow_worstcase.csv (fully correlated stress test:
every estimated flow scaled by the same 5th or 95th percentile country ratio). Run before model.py."""
import sys, pathlib, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold, GroupKFold
ROOT = pathlib.Path(sys.argv[1]); D = ROOT / "data"; OUT = ROOT / "out"
d = pd.read_csv(D / "hw/HydroWASTE_v10.csv", encoding="latin-1")
ow = pd.read_csv(D / "owid/owid-energy-data.csv", usecols=["iso_code", "year", "gdp", "population"]).dropna(subset=["gdp"])
ow = ow.sort_values("year").groupby("iso_code").tail(1).set_index("iso_code")
d["gdp_pc"] = np.log10((ow.gdp / ow.population).reindex(d.CNTRY_ISO).values); d["gdp_pc"] = d.gdp_pc.fillna(d.gdp_pc.median())
d = d[d.POP_SERVED > 0].copy()
d["lpop"] = np.log10(d.POP_SERVED); d["lvl"] = d.LEVEL.map({"Primary": 0, "Secondary": 1, "Advanced": 2}).fillna(1)
d["alat"] = d.LAT_WWTP.abs(); d["y"] = np.log10(d.WASTE_DIS / d.POP_SERVED)
tr_all = d[(d.QUAL_WASTE == 1) & np.isfinite(d.y)].reset_index(drop=True)
feats = ["lpop", "QUAL_POP", "lvl", "gdp_pc", "alat", "LON_WWTP", "SOURCE", "cmed"]
def model(q=None, fs=feats):
    kw = dict(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=1.0,
              categorical_features=[fs.index("SOURCE")] if "SOURCE" in fs else None, random_state=0)
    return HistGradientBoostingRegressor(loss="quantile", quantile=q, **kw) if q else HistGradientBoostingRegressor(**kw)

def run(name, splits, fs=feats):
    y = tr_all.y.values; oof = np.full(len(y), np.nan); lo = oof.copy(); hi = oof.copy(); base = oof.copy()
    for tr, te in splits:
        a, b = tr_all.iloc[tr].copy(), tr_all.iloc[te].copy()
        med = a.groupby("CNTRY_ISO").y.median(); gm = a.y.median()
        a["cmed"] = np.nan   # out-of-fold country medians inside the training set, so the feature is not leaked
        for i, j in KFold(5, shuffle=True, random_state=1).split(a):
            a.iloc[j, a.columns.get_loc("cmed")] = a.iloc[j].CNTRY_ISO.map(a.iloc[i].groupby("CNTRY_ISO").y.median()).values
        a["cmed"] = a.cmed.fillna(gm)
        b["cmed"] = b.CNTRY_ISO.map(med).fillna(gm)
        X, Xt = a[fs].values, b[fs].values
        oof[te] = model(None, fs).fit(X, a.y).predict(Xt); lo[te] = model(0.1, fs).fit(X, a.y).predict(Xt); hi[te] = model(0.9, fs).fit(X, a.y).predict(Xt)
        base[te] = b.cmed.values
    f, pf, bf = tr_all.WASTE_DIS.values, 10 ** oof * tr_all.POP_SERVED.values, 10 ** base * tr_all.POP_SERVED.values
    r2 = lambda pr: 1 - np.sum((np.log10(f) - np.log10(pr)) ** 2) / np.sum((np.log10(f) - np.log10(f).mean()) ** 2)
    out = dict(design=name, n=len(f), r2_log_flow=r2(pf), r2_log_flow_baseline=r2(bf),
               mdape_pct=np.median(np.abs(pf / f - 1)) * 100, mdape_baseline_pct=np.median(np.abs(bf / f - 1)) * 100,
               sum_ratio=pf.sum() / f.sum(), pi80_coverage=np.mean((y >= lo) & (y <= hi)))
    return out, pf

res = []; o1, _ = run("random", KFold(5, shuffle=True, random_state=0).split(tr_all)); res.append(o1)
fs_g = ["lpop", "lvl", "gdp_pc"]   # parsimonious set: held-out groups have no country median, and source or location
# features do not transfer to unseen countries (R2 of log flow 0.21 with the full set, 0.84 with the parsimonious one)
o2, pf_c = run("country", GroupKFold(5).split(tr_all, groups=tr_all.CNTRY_ISO), fs_g); res.append(o2)
o3, _ = run("source", GroupKFold(5).split(tr_all, groups=tr_all.SOURCE), fs_g); res.append(o3)
S = pd.DataFrame(res); S.to_csv(OUT / "ml_skill.csv", index=False); print(S.round(3).to_string())

# country-level aggregate ratios from country-grouped CV
cr = pd.DataFrame(dict(iso=tr_all.CNTRY_ISO, obs=tr_all.WASTE_DIS, pred=pf_c)).groupby("iso").sum()
cr = cr[cr.obs > 0]; cr["r"] = cr.obs / cr.pred
w0 = cr.obs / cr.obs.sum(); lr = np.log(cr.r.values); o_ = np.argsort(lr)
cr["r_centred"] = cr.r / (w0 * cr.r).sum()   # flow-weighted mean ratio 1: no systematic volume bias assumed
cr.to_csv(OUT / "ml_country_ratios.csv")
w = cr.obs / cr.obs.sum()
print(f"country ratios: n={len(cr)}, flow-weighted median {np.interp(0.5, np.cumsum(w.values[np.argsort(cr.r.values)]), np.sort(cr.r.values)):.3f}; "
      f"p05 {cr.r.quantile(.05):.2f} p95 {cr.r.quantile(.95):.2f}")
pl = pd.read_csv(OUT / "plants.csv")
from common import scopes as _scopes
scopes = {k: np.asarray(m) for k, m in _scopes(pl).items()}
rng = np.random.default_rng(7); B = 2000
countries = pl.CNTRY_ISO.unique(); est = (pl.QUAL_WASTE == 4).values
draws = {k: [] for k in scopes}
for _ in range(B):
    rmap = pd.Series(rng.choice(cr.r_centred.values, len(countries), p=w.values), index=countries)
    scale = np.where(est, pl.CNTRY_ISO.map(rmap).values, 1.0)
    for k, m in scopes.items():
        q = pl.WASTE_DIS.values[m]; draws[k].append((q * scale[m]).sum() / q.sum())
Bt = pd.DataFrame(draws); Bt.to_csv(OUT / "flow_bootstrap.csv", index=False)
print("bootstrap scope ratio p05/p50/p95:\n", Bt.quantile([.05, .5, .95]).round(3).to_string())
# fully correlated stress test: every estimated flow scaled by the same low or high country ratio (5th, 95th percentile)
lo_r, hi_r = np.percentile(cr.r_centred.values, [5, 95])
wc = []
for k, m in scopes.items():
    q = pl.WASTE_DIS.values[m]; e = est[m]
    rep, es = q[~e].sum(), q[e].sum()
    wc.append(dict(scope=k, total_Mm3d=(rep + es) / 1e6, estimated_share=es / (rep + es), ratio_low=lo_r, ratio_high=hi_r,
                   low_Mm3d=(rep + lo_r * es) / 1e6, high_Mm3d=(rep + hi_r * es) / 1e6))
pd.DataFrame(wc).to_csv(OUT / "flow_worstcase.csv", index=False)
print(pd.DataFrame(wc).round(2).to_string())

