"""Stage 2: reclamation model, Monte Carlo uncertainty, cascade, and desalination comparison.

Mass balance (per plant, m3/d):
  feed       = effluent * f_avail                 (share that can be diverted; flow variability, minimum discharge)
  potable    = feed * r_pot                       (overall product recovery of MF/UF + RO + UV-AOP)
  irrigation = feed * r_irr                       (filtration + disinfection, no RO)
  concentrate (potable route) = feed - potable
Seawater RO producing the same potable volume: brine = potable * (1 - rr_swro) / rr_swro.
Electricity (kWh/d): reuse = e_pot * potable + u_prim * feed (u_prim only for plants with primary treatment);
  seawater RO = e_swro * potable.  PV kWp = daily kWh / (GHI * PR).  Land = kWp * module area / GCR.
Uncertainty: N independent triangular draws (low, mode = central, high) of every parameter; reported as the median
and the 5th to 95th percentiles. Corner bounds (all combinations of extremes) are reported for electricity saved.
If out/flow_bootstrap.csv exists, each draw also scales flows by a bootstrap ratio from the imputation model.
"""
import sys, itertools, pathlib
import numpy as np, pandas as pd

ROOT = pathlib.Path(sys.argv[1]); OUT = ROOT / "out"
N = 20000; RNG = np.random.default_rng(42)
P = dict(  # central, low, high
    f_avail=(0.90, 0.80, 0.95),
    r_pot=(0.80, 0.70, 0.85),
    r_irr=(0.95, 0.90, 0.98),
    e_pot=(1.00, 0.80, 1.50),      # kWh per m3 of potable product
    e_irr=(0.30, 0.20, 0.50),      # kWh per m3 of irrigation-grade product
    e_swro=(3.50, 3.00, 4.00),     # kWh per m3, incl. intake and pretreatment, excl. conveyance
    rr_swro=(0.42, 0.40, 0.50),
    u_prim=(0.30, 0.20, 0.50),     # kWh per m3 of feed to bring primary-treated effluent to secondary standard
    pr=(0.80, 0.75, 0.85), m2_kwp=(5.0, 4.5, 6.0), gcr=(0.40, 0.33, 0.50),
)
SIZE_MIN = 10000   # m3/d, cascade step 3
c = pd.read_csv(OUT / "plants.csv")
c["prim"] = (c.LEVEL == "Primary").astype(float)

# ---------------- central plant-level values ----------------
k = {a: b[0] for a, b in P.items()}
Q = c.WASTE_DIS
c["feed_m3d"] = Q * k["f_avail"]
c["pot_m3d"] = c.feed_m3d * k["r_pot"]
c["irr_m3d"] = c.feed_m3d * k["r_irr"]
c["conc_m3d"] = c.feed_m3d - c.pot_m3d
c["e_reuse_kwh_d"] = k["e_pot"] * c.pot_m3d + k["u_prim"] * c.feed_m3d * c.prim
c["e_swro_kwh_d"] = k["e_swro"] * c.pot_m3d
c["e_saved_kwh_d"] = c.e_swro_kwh_d - c.e_reuse_kwh_d
c["kwp_reuse"] = c.e_reuse_kwh_d / (c.ghi * k["pr"])
c["land_ha_reuse"] = c.kwp_reuse * k["m2_kwp"] / k["gcr"] / 1e4
c["kwp_swro"] = c.e_swro_kwh_d / (c.ghi * k["pr"])
c["brine_swro_m3d"] = c.pot_m3d * (1 - k["rr_swro"]) / k["rr_swro"]
c["brine_avoided_m3d"] = c.brine_swro_m3d - c.conc_m3d
c["co2_saved_tpy"] = c.e_saved_kwh_d * 365 * c.grid_g_kwh / 1e6
c["pot_net_m3d"] = c.pot_m3d * (1 - c.reuse_frac)

# ---------------- Monte Carlo on aggregates ----------------
draw = {a: RNG.triangular(b[1], b[0], b[2], N) for a, b in P.items()}
fb = OUT / "flow_bootstrap.csv"
boot = pd.read_csv(fb) if fb.exists() else None
def sums(g):
    q = g.WASTE_DIS; qp = q * g.prim
    return dict(SQ=q.sum(), SQp=qp.sum(), SQg=(q / g.ghi).sum(), SQpg=(qp / g.ghi).sum(), SQc=(q * g.grid_g_kwh).sum(),
                SQpc=(qp * g.grid_g_kwh).sum(), SQn=(q * (1 - g.reuse_frac)).sum())
def mc(g, scope):
    s = sums(g); x = draw
    scale = 1.0
    if boot is not None and scope in boot.columns: scale = RNG.choice(boot[scope].values, N)
    f, r = x["f_avail"], x["r_pot"]
    pot = f * r * s["SQ"] * scale
    e_re = (x["e_pot"] * f * r * s["SQ"] + x["u_prim"] * f * s["SQp"]) * scale
    e_sw = x["e_swro"] * pot
    kwp = (x["e_pot"] * f * r * s["SQg"] + x["u_prim"] * f * s["SQpg"]) * scale / x["pr"]
    out = dict(
        effluent_Mm3d=np.full(N, s["SQ"]) * scale / 1e6,
        potable_Mm3d=pot / 1e6, irrigation_Mm3d=f * x["r_irr"] * s["SQ"] * scale / 1e6,
        potable_net_reuse_Mm3d=f * r * s["SQn"] * scale / 1e6,
        e_reuse_TWh=e_re * 365 / 1e9, e_swro_TWh=e_sw * 365 / 1e9, e_saved_TWh=(e_sw - e_re) * 365 / 1e9,
        e_ratio=e_re / e_sw,
        GWp_reuse=kwp / 1e6, land_km2_reuse=kwp * x["m2_kwp"] / x["gcr"] / 1e6,
        GWp_swro=x["e_swro"] * f * r * s["SQg"] * scale / x["pr"] / 1e6,
        brine_swro_Mm3d=pot * (1 - x["rr_swro"]) / x["rr_swro"] / 1e6,
        conc_Mm3d=f * (1 - r) * s["SQ"] * scale / 1e6,
        co2_Mt=((x["e_swro"] - x["e_pot"]) * f * r * s["SQc"] - x["u_prim"] * f * s["SQpc"]) * scale * 365 / 1e12,
    )
    out["brine_avoided_Mm3d"] = out["brine_swro_Mm3d"] - out["conc_Mm3d"]
    res = {}
    for kk, v in out.items():
        res[kk] = np.median(v); res[kk + "_p05"] = np.percentile(v, 5); res[kk + "_p95"] = np.percentile(v, 95)
    # corner bounds for electricity saved (parameter extremes only, no flow scaling)
    keys = ["f_avail", "r_pot", "e_pot", "e_swro", "u_prim"]; vals = []
    for combo in itertools.product(*[(P[kk][1], P[kk][2]) for kk in keys]):
        z = dict(zip(keys, combo))
        vals.append(((z["e_swro"] - z["e_pot"]) * z["f_avail"] * z["r_pot"] * s["SQ"] - z["u_prim"] * z["f_avail"] * s["SQp"]) * 365 / 1e9)
    res["e_saved_TWh_corner_min"] = min(vals); res["e_saved_TWh_corner_max"] = max(vals)
    res["n_plants"] = len(g)
    return res

oc = c[c.ocean_outfall]
from common import MED, GULF_RED
scopes = {"coastal_50km": c, "ocean_outfalls": oc, "river_outfalls_50km": c[~c.ocean_outfall],
          "ocean_stressed": oc[oc.stressed], "ocean_tierA": oc[oc.tier == "A"], "coastal_tierA": c[c.tier == "A"],
          "ocean_Mediterranean": oc[MED(oc)], "ocean_Gulf_RedSea": oc[GULF_RED(oc)]}
G = pd.DataFrame({k_: mc(g, k_) for k_, g in scopes.items()}).T
G.to_csv(OUT / "global_mc.csv")

# ---------------- country summaries (central values, ocean outfalls and all coastal) ----------------
def agg(g):
    o = g[g.ocean_outfall]
    return pd.Series(dict(
        n=len(g), n_ocean=len(o), effluent_Mm3d=g.WASTE_DIS.sum() / 1e6, ocean_effluent_Mm3d=o.WASTE_DIS.sum() / 1e6,
        ocean_potable_Mm3d=o.pot_m3d.sum() / 1e6, ocean_potable_net_Mm3d=o.pot_net_m3d.sum() / 1e6,
        ocean_irr_Mm3d=o.irr_m3d.sum() / 1e6, ocean_irr_stressed_Mm3d=o[o.stressed].irr_m3d.sum() / 1e6,
        ocean_potable_stressed_Mm3d=o[o.stressed].pot_m3d.sum() / 1e6,
        ocean_stressed_share=o[o.stressed].pot_m3d.sum() / max(o.pot_m3d.sum(), 1e-9),
        ocean_e_saved_TWh=o.e_saved_kwh_d.sum() * 365 / 1e9, ocean_GWp=o.kwp_reuse.sum() / 1e6,
        ocean_co2_Mt=o.co2_saved_tpy.sum() / 1e6, ocean_tierC_share=o[o.tier == "C"].WASTE_DIS.sum() / max(o.WASTE_DIS.sum(), 1e-9),
        coastal_potable_Mm3d=g.pot_m3d.sum() / 1e6, ghi_median=g.ghi.median(), grid_g_kwh=g.grid_g_kwh.median(),
        reuse_frac=g.reuse_frac.iloc[0]))
C = c.groupby("CNTRY_ISO").apply(agg, include_groups=False)
ctx = pd.read_csv(OUT / "country_context.csv", index_col=0)
C = C.join(ctx[["country", "jmp_sm_pct", "pop_without_sm_M", "aei_25km_union_ha", "aei_25km_union_ocean_ha"]])
# irrigation: plant-specific gross irrigation requirement (m3/ha/yr, from build_plants.py) instead of a flat rate
oc_ = c[c.ocean_outfall].copy()
oc_["irr_ha"] = oc_.irr_m3d * 365 / oc_.irr_req_m3_ha.clip(lower=500)
g_ = oc_.groupby("CNTRY_ISO")
C["irr_supply_ha"] = g_.irr_ha.sum().reindex(C.index).fillna(0)
C["irr_supply_stressed_ha"] = oc_[oc_.stressed].groupby("CNTRY_ISO").irr_ha.sum().reindex(C.index).fillna(0)
C["irr_req_m3_ha"] = g_.apply(lambda d: np.average(d.irr_req_m3_ha, weights=d.WASTE_DIS.clip(lower=1)), include_groups=False).reindex(C.index)
C["irr_cover_25km"] = C.irr_supply_ha / C.aei_25km_union_ocean_ha.replace(0, np.nan)
C["irr_cover_25km_stressed"] = C.irr_supply_stressed_ha / C.aei_25km_union_ocean_ha.replace(0, np.nan)
# freshwater irrigation withdrawals replaceable: the smaller of irrigation-grade supply and the gross requirement of the
# irrigated area within 25 km of ocean outfalls (country union of buffers), in km3/yr
C["irr_demand_km3"] = C.aei_25km_union_ocean_ha * C.irr_req_m3_ha / 1e9
C["irr_supply_km3"] = C.ocean_irr_Mm3d * 365 / 1e3
C["irr_replaceable_km3"] = np.minimum(C.irr_supply_km3, C.irr_demand_km3.fillna(0))
C["irr_supply_stressed_km3"] = C.ocean_irr_stressed_Mm3d * 365 / 1e3
C["irr_replaceable_stressed_km3"] = np.minimum(C.irr_supply_stressed_km3, C.irr_demand_km3.fillna(0))
C = C.sort_values("ocean_potable_Mm3d", ascending=False)
C.to_csv(OUT / "country_summary.csv")

# ---------------- desalination comparison ----------------
DC = pd.read_csv(OUT / "desal_countries.csv").set_index("iso")
DC["ocean_potable_Mm3d"] = C.ocean_potable_Mm3d.reindex(DC.index)
DC["ocean_potable_net_Mm3d"] = C.ocean_potable_net_Mm3d.reindex(DC.index)
DC["coastal_potable_Mm3d"] = C.coastal_potable_Mm3d.reindex(DC.index)
DC["ratio"] = DC.ocean_potable_Mm3d / DC.desal_Mm3d
DC["ratio_net"] = DC.ocean_potable_net_Mm3d / DC.desal_Mm3d
DC["coverable_Mm3d"] = np.minimum(DC.ocean_potable_Mm3d, DC.desal_Mm3d)
DC["e_saved_if_replaced_TWh"] = DC.coverable_Mm3d * 1e6 * 365 * (P["e_swro"][0] - P["e_pot"][0]) / 1e9
DC.sort_values("desal_Mm3d", ascending=False).to_csv(OUT / "desal_comparison_countries.csv")
REG = {}
for i in "DZA BHR EGY IRN IRQ ISR JOR KWT LBN LBY MAR OMN QAT SAU SYR TUN ARE YEM PSE MLT DJI".split(): REG[i] = "Middle East and North Africa"
for i in "CHN JPN KOR IDN MYS PHL THA VNM AUS NZL SGP HKG TWN KHM MMR PNG FJI BRN PRK MAC TLS WSM TON VUT SLB KIR FSM MHL PLW NCL PYF GUM MNP ASM".split(): REG[i] = "East Asia and Pacific"
for i in "USA CAN BMU GRL PRI VIR".split(): REG[i] = "North America"
for i in "GBR IRL FRA DEU NLD BEL LUX ESP PRT ITA GRC AUT CHE DNK SWE NOR FIN ISL CYP AND MCO GIB FRO IMN JEY GGY".split(): REG[i] = "Western Europe"
for i in ("MEX BRA ARG CHL PER COL VEN ECU URY PRY BOL GUY SUR CUB DOM HTI JAM TTO BHS BRB BLZ CRI PAN NIC HND SLV GTM LCA VCT GRD ATG "
          "DMA KNA ABW CUW CYM TCA VGB AIA MSR SXM BES MTQ GLP GUF MAF BLM").split(): REG[i] = "Latin America and Caribbean"
for i in "IND PAK BGD LKA NPL BTN MDV AFG".split(): REG[i] = "Southern Asia"
for i in "RUS UKR POL ROU BGR HRV SVN HUN CZE SVK SRB BIH MNE MKD ALB EST LVA LTU BLR MDA GEO ARM AZE KAZ UZB TKM KGZ TJK TUR XKX".split(): REG[i] = "Eastern Europe and Central Asia"
C["region"] = [REG.get(i, "Sub-Saharan Africa") for i in C.index]
DR = pd.read_csv(OUT / "desal_regions.csv").set_index("region")
DR["ocean_potable_Mm3d"] = C.groupby("region").ocean_potable_Mm3d.sum()
DR["coastal_potable_Mm3d"] = C.groupby("region").coastal_potable_Mm3d.sum()
DR["ocean_potable_net_Mm3d"] = C.groupby("region").ocean_potable_net_Mm3d.sum()
DR["ratio"] = DR.ocean_potable_Mm3d / DR.desal_Mm3d
DR["ratio_net"] = DR.ocean_potable_net_Mm3d / DR.desal_Mm3d
DR["e_saved_if_replaced_TWh"] = np.minimum(DR.ocean_potable_Mm3d, DR.desal_Mm3d) * 1e6 * 365 * (P["e_swro"][0] - P["e_pot"][0]) / 1e9
DR.to_csv(OUT / "desal_comparison_regions.csv")

# ---------------- cascade ----------------
DW = pd.read_csv(OUT / "desal_world.csv").set_index("key").value
desal_iso = set(DC[DC.desal_Mm3d >= 0.1].index)
steps = []
def step(name, unit, df, col):
    steps.append(dict(step=name, unit=unit, n_plants=len(df), value_Mm3d=df[col].sum() / 1e6,
                      tierA_Mm3d=df[df.tier == "A"][col].sum() / 1e6))
step("Treated effluent within 50 km of the sea", "effluent", c, "WASTE_DIS")
step("Discharged at ocean outfalls", "effluent", oc, "WASTE_DIS")
step("Potable product after diversion and treatment losses", "product", oc, "pot_m3d")
s3 = oc[(oc.LEVEL != "Primary") & (oc.WASTE_DIS >= SIZE_MIN)]
step(f"From plants with secondary or advanced treatment and at least {SIZE_MIN:,} m3/d", "product", s3, "pot_m3d")
step("Net of existing reuse", "product", s3, "pot_net_m3d")
s5 = s3[s3.stressed]
step("In basins of high or extremely high water stress", "product", s5, "pot_net_m3d")
s6 = s5[s5.CNTRY_ISO.isin(desal_iso)]
step("In countries with at least 0.1 Mm3/d of desalination", "product", s6, "pot_net_m3d")
K = pd.DataFrame(steps)
# Monte Carlo: product steps scale with f_avail * r_pot; all steps scale with the flow bootstrap (ocean scope)
fs_ = RNG.choice(boot["ocean_outfalls"].values, N) if boot is not None else np.ones(N)
fr = draw["f_avail"] * draw["r_pot"] / (P["f_avail"][0] * P["r_pot"][0]) * fs_
K["central_Mm3d"] = K.value_Mm3d
for qn, qq in (("median", 50), ("p05", 5), ("p95", 95)):
    K[qn] = np.where(K.unit == "product", K.central_Mm3d * np.percentile(fr, qq), K.central_Mm3d * np.percentile(fs_, qq))
K["value_Mm3d"] = K["median"]; K = K.drop(columns="median")
K.to_csv(OUT / "cascade.csv", index=False)
# size threshold sensitivity
sens = []
for t in (0, 5000, 10000, 20000, 50000):
    sub = oc[(oc.LEVEL != "Primary") & (oc.WASTE_DIS >= t)]
    sens.append(dict(threshold_m3d=t, n=len(sub), product_net_Mm3d=sub.pot_net_m3d.sum() / 1e6,
                     final_Mm3d=sub[sub.stressed & sub.CNTRY_ISO.isin(desal_iso)].pot_net_m3d.sum() / 1e6))
pd.DataFrame(sens).to_csv(OUT / "cascade_size_sensitivity.csv", index=False)
# existing-reuse sensitivity: coastal reuse fraction = national fraction x m, capped at 1 (m = 1 is the main analysis)
rs = []
for m_ in (0.0, 0.5, 1.0, 1.5):
    net = s3.pot_m3d * (1 - (s3.reuse_frac.fillna(0) * m_).clip(upper=1))
    fin = s3.stressed & s3.CNTRY_ISO.isin(desal_iso)
    rs.append(dict(reuse_multiplier=m_, net_product_Mm3d=net.sum() / 1e6, screened_potential_Mm3d=net[fin].sum() / 1e6,
                   screened_n_plants=int((fin & (net > 0)).sum())))
pd.DataFrame(rs).to_csv(OUT / "cascade_reuse_sensitivity.csv", index=False)

# ---------------- future water stress (Aqueduct 4.0 projections; current flows held fixed) ----------------
FS = []
el = oc[(oc.LEVEL != "Primary") & (oc.WASTE_DIS >= SIZE_MIN)]
for sc in ("opt", "bau", "pes"):
    for y in (30, 50, 80):
        col = f"fut_{sc}{y}_c"; s_o = oc[col] >= 3; s_e = el[col] >= 3
        FS.append(dict(scenario=sc, year=2000 + y, effluent_Mm3d=oc[s_o].WASTE_DIS.sum() / 1e6, share=oc[s_o].WASTE_DIS.sum() / oc.WASTE_DIS.sum(),
                       potable_net_eligible_Mm3d=el[s_e].pot_net_m3d.sum() / 1e6,
                       potable_net_eligible_desal_Mm3d=el[s_e & el.CNTRY_ISO.isin(desal_iso)].pot_net_m3d.sum() / 1e6,
                       newly_stressed_effluent_Mm3d=oc[s_o & ~oc.stressed].WASTE_DIS.sum() / 1e6))
FS.insert(0, dict(scenario="baseline", year=2019, effluent_Mm3d=oc[oc.stressed].WASTE_DIS.sum() / 1e6, share=oc[oc.stressed].WASTE_DIS.sum() / oc.WASTE_DIS.sum(),
                  potable_net_eligible_Mm3d=el[el.stressed].pot_net_m3d.sum() / 1e6,
                  potable_net_eligible_desal_Mm3d=el[el.stressed & el.CNTRY_ISO.isin(desal_iso)].pot_net_m3d.sum() / 1e6, newly_stressed_effluent_Mm3d=0.0))
pd.DataFrame(FS).to_csv(OUT / "future_stress.csv", index=False)
FC = oc.assign(now=oc.stressed, f50=oc.fut_bau50_c >= 3).groupby("CNTRY_ISO").apply(
    lambda d: pd.Series(dict(now=d[d.now].pot_m3d.sum() / 1e6, bau2050=d[d.f50].pot_m3d.sum() / 1e6)), include_groups=False)
FC.join(C[["country"]]).to_csv(OUT / "future_stress_countries.csv")

# ---------------- seasonal check: monthly water stress (Aqueduct) against monthly irradiance and irrigation need ----------------
bm = oc[[f"bws_m{i:02d}" for i in range(1, 13)]].where(lambda x: x < 9999).values
gm = oc[[f"ghi_m{i:02d}" for i in range(1, 13)]].values
nm = oc[[f"nir_m{i:02d}" for i in range(1, 13)]].values
hi = bm >= 0.4                                   # high or extremely high monthly stress (raw ratio >= 0.4)
anyhi = hi.any(1); w_ = oc.WASTE_DIS.values
sun_hi = np.array([np.nanmean(g[h]) >= np.nanmean(g) if h.any() else False for g, h in zip(gm, hi)])
nir_tot = np.nansum(nm, 1); nir_hi = np.array([np.nansum(n[h]) for n, h in zip(nm, hi)])
SE = dict(effluent_with_stressed_month_Mm3d=w_[anyhi].sum() / 1e6, share_with_stressed_month=w_[anyhi].sum() / w_.sum(),
          median_stressed_months=float(np.median(hi.sum(1)[anyhi])),
          share_sunnier_in_stressed_months=w_[anyhi & sun_hi].sum() / max(w_[anyhi].sum(), 1),
          share_irrigation_need_in_stressed_months=np.nansum(nir_hi * w_) / max(np.nansum(nir_tot * w_), 1))
pd.Series(SE).to_csv(OUT / "seasonal.csv")

# ---------------- treatment-level sensitivity (ocean outfalls, central parameters) ----------------
o = oc.copy(); o_prim = o.LEVEL == "Primary"; tot = o.WASTE_DIS.sum()
def esaved(u, mask): return (o.e_swro_kwh_d - (k["e_pot"] * o.pot_m3d + u * o.feed_m3d * mask)).sum() * 365 / 1e9
TS = [("primary_share_of_effluent", o[o_prim].WASTE_DIS.sum() / tot), ("advanced_share_of_effluent", o[o.LEVEL == "Advanced"].WASTE_DIS.sum() / tot),
      ("potable_Mm3d_central", o.pot_m3d.sum() / 1e6), ("potable_Mm3d_excl_primary", o[~o_prim].pot_m3d.sum() / 1e6),
      ("e_saved_TWh_central", esaved(k["u_prim"], o_prim))]
TS += [(f"e_saved_TWh_u_prim_{u}", esaved(u, o_prim)) for u in (0.2, 0.5, 1.0)]
TS += [("e_saved_TWh_tierC_as_primary_u0.5", esaved(0.5, o_prim | (o.tier == "C")))]
pd.DataFrame(TS, columns=["metric", "value"]).to_csv(OUT / "treatment_sensitivity.csv", index=False)

# ---------------- one-at-a-time sensitivity of electricity saved (ocean outfalls) ----------------
so = sums(oc)
def es(z): return ((z["e_swro"] - z["e_pot"]) * z["f_avail"] * z["r_pot"] * so["SQ"] - z["u_prim"] * z["f_avail"] * so["SQp"]) * 365 / 1e9
tor = []
for kk in ["f_avail", "r_pot", "e_pot", "e_swro", "u_prim"]:
    lo_, hi_ = dict(k), dict(k); lo_[kk] = P[kk][1]; hi_[kk] = P[kk][2]
    tor.append(dict(param=kk, low=P[kk][1], high=P[kk][2], at_low=es(lo_), at_high=es(hi_), central=es(k)))
bq = np.percentile(boot["ocean_outfalls"], [5, 95]) if boot is not None else (1, 1)
tor.append(dict(param="flow_bootstrap", low=bq[0], high=bq[1], at_low=es(k) * bq[0], at_high=es(k) * bq[1], central=es(k)))
pd.DataFrame(tor).to_csv(OUT / "tornado_e_saved.csv", index=False)

c.to_csv(OUT / "plants_model.csv", index=False)
pd.set_option("display.width", 220)
cols = ["n_plants", "effluent_Mm3d", "potable_Mm3d", "potable_Mm3d_p05", "potable_Mm3d_p95", "e_saved_TWh", "e_saved_TWh_p05", "e_saved_TWh_p95",
        "e_saved_TWh_corner_min", "e_saved_TWh_corner_max", "e_ratio", "GWp_reuse", "land_km2_reuse", "brine_avoided_Mm3d", "co2_Mt", "co2_Mt_p05", "co2_Mt_p95"]
print(G[cols].round(2).to_string()); print(K.round(2).to_string()); print(pd.DataFrame(sens).round(2).to_string())
print("world refs:", DW.to_dict())
