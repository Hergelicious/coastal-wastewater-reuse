"""Stage 1: coastal plant table with all context layers (no modelling parameters here).

Plant set: HydroWASTE v1.0 entries whose status is not Closed, Decommissioned, Non-Operational, Projected,
Proposed or Under Construction. Coastal: COAST_50KM = 1 and estimated outfall within 50 km of the Natural Earth
1:10 m ocean coastline (removes large-lake plants that the COAST flag also counts).
Outfall type (HydroWASTE estimate): ocean when RIVER_DIS is empty (README: "empty if estimated outfall location is
the ocean"), otherwise river.
Data tiers: A reported flow (QUAL_WASTE 1-3); B flow estimated from reported population served or population
equivalent (QUAL_WASTE 4, QUAL_POP 1-2); C population and flow both modelled (QUAL_WASTE 4, QUAL_POP 3-4).
Validation corrections from inputs/validation_corrections.csv are applied (plants found not to exist removed, flows,
outfall type and treatment level corrected for the 25 largest ocean outfalls). Writes out/plants.csv, out/country_context.csv
and out/validation_effect.csv.
"""
import sys, json, pathlib
import numpy as np, pandas as pd, pycountry
from scipy.spatial import cKDTree

ROOT = pathlib.Path(sys.argv[1]); D = ROOT / "data"; OUT = ROOT / "out"; OUT.mkdir(exist_ok=True)

# ---------------- plants and coastal classification ----------------
d = pd.read_csv(D / "hw/HydroWASTE_v10.csv", encoding="latin-1")
d = d[~d.STATUS.isin(["Closed", "Decommissioned", "Non-Operational", "Projected", "Proposed", "Under Construction"])].copy()
coast = json.load(open(D / "ne/ne_10m_coastline.geojson"))
pts = []
for f in coast["features"]:
    g = f["geometry"]; lines = g["coordinates"] if g["type"] == "MultiLineString" else [g["coordinates"]]
    for ln in lines: pts.extend(ln)
pts = np.array(pts, float)
def unit(lon, lat):
    lo, la = np.radians(lon), np.radians(lat)
    return np.c_[np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]
tree = cKDTree(unit(pts[:, 0], pts[:, 1]))
ch, _ = tree.query(unit(d.LON_OUT.values, d.LAT_OUT.values))
d["sea_km"] = 2 * 6371.0 * np.arcsin(np.clip(ch / 2, 0, 1))
all_flow = d.WASTE_DIS.sum()
c = d[(d.COAST_50KM == 1) & (d.sea_km <= 50)].copy()
c["ocean_outfall"] = c.RIVER_DIS.isna()
c["tier"] = np.select([c.QUAL_WASTE.isin([1, 2, 3]), c.QUAL_POP.isin([1, 2])], ["A", "B"], "C")

# ---------------- validation corrections (inputs/validation_corrections.csv; checks of the 25 largest ocean outfalls) ----------------
before = dict(n=len(c), coastal=c.WASTE_DIS.sum(), ocean=c[c.ocean_outfall].WASTE_DIS.sum())
corr = pd.read_csv(ROOT / "inputs/validation_corrections.csv")
c["validated_change"] = ""
for r in corr.itertuples():
    if r.field == "drop_country_synthetic":
        m = (c.CNTRY_ISO == r.new_value) & (c.SOURCE == 12); c = c[~m]; continue
    w = int(r.WASTE_ID); m = c.WASTE_ID == w
    if not m.any(): continue
    if r.field == "drop": c = c[~m]; continue
    if r.field == "WASTE_DIS": c.loc[m, "WASTE_DIS"] = float(r.new_value)
    if r.field == "ocean_outfall": c.loc[m, "ocean_outfall"] = str(r.new_value) == "True"
    if r.field == "LEVEL": c.loc[m, "LEVEL"] = r.new_value
    c.loc[m, "validated_change"] = r.field
after = dict(n=len(c), coastal=c.WASTE_DIS.sum(), ocean=c[c.ocean_outfall].WASTE_DIS.sum())
pd.DataFrame([dict(state="HydroWASTE as published", **before), dict(state="after validation corrections", **after)]).to_csv(OUT / "validation_effect.csv", index=False)
c = c.reset_index(drop=True)

# ---------------- solar and climate (NASA POWER climatology, 0.5 degree cells) ----------------
# Irrigation requirement per cell: reference evapotranspiration by the Hargreaves radiation formula,
# ET0 = 0.0135 (T + 17.8) Rs (Rs in mm/d of evaporation equivalent), crop coefficient 1.0, effective rainfall by the
# USDA SCS method; net requirement summed over months, gross requirement = net / irrigation efficiency (0.65).
MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
DAYS = np.array([31, 28.25, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
KC, EFF = 1.0, 0.65
c["cell_lat"] = (c.LAT_OUT * 2).round() / 2; c["cell_lon"] = (c.LON_OUT * 2).round() / 2
def climate(lat, lon):
    f = D / "power_cells" / f"{lat:.1f}_{lon:.1f}.json"
    out = dict(ghi=np.nan, irr_req_m3_ha=np.nan, irr_peak_month=np.nan, **{f"ghi_m{m + 1:02d}": np.nan for m in range(12)},
               **{f"nir_m{m + 1:02d}": np.nan for m in range(12)})
    if not f.exists(): return out
    try: par = json.load(open(f))["properties"]["parameter"]
    except Exception: return out
    g = par["ALLSKY_SFC_SW_DWN"]; out["ghi"] = g["ANN"]
    gm = np.array([g[m] for m in MONTHS], float)
    for m in range(12): out[f"ghi_m{m + 1:02d}"] = gm[m]
    if "T2M" in par and "PRECTOTCORR" in par:
        t = np.array([par["T2M"][m] for m in MONTHS], float); pr = np.array([par["PRECTOTCORR"][m] for m in MONTHS], float) * DAYS
        if (gm < 0).any() or (t < -90).any() or (pr < 0).any(): return out
        et0 = 0.0135 * (t + 17.8) * gm * 3.6 / 2.45 * DAYS                     # mm per month
        peff = np.where(pr <= 250, pr * (125 - 0.2 * pr) / 125, 125 + 0.1 * pr)
        nir = np.clip(KC * et0 - peff, 0, None)
        for m in range(12): out[f"nir_m{m + 1:02d}"] = nir[m]
        out["irr_req_m3_ha"] = nir.sum() * 10 / EFF; out["irr_peak_month"] = int(np.argmax(nir)) + 1 if nir.sum() > 0 else np.nan
    return out
cells = c[["cell_lat", "cell_lon"]].drop_duplicates().reset_index(drop=True)
cells = pd.concat([cells, pd.DataFrame([climate(a, b) for a, b in zip(cells.cell_lat, cells.cell_lon)])], axis=1)
c = c.merge(cells, on=["cell_lat", "cell_lon"], how="left")
c.loc[c.ghi < 0, "ghi"] = np.nan
for col in ["ghi", "irr_req_m3_ha"]:
    c[col] = c.groupby("CNTRY_ISO")[col].transform(lambda s: s.fillna(s.median()))
    c[col] = c[col].fillna(c[col].median())

# ---------------- water stress (Aqueduct 4.0, from join_aqueduct.py) ----------------
aqc = pd.read_csv(OUT / "plant_aqueduct.csv")
aq = aqc[["WASTE_ID", "bws_raw", "bws_cat", "bws_label"] + [x for x in aqc.columns if x.startswith("fut_") or x.startswith("bws_m")]]
c = c.merge(aq, on="WASTE_ID", how="left")
c["stressed"] = c.bws_cat >= 3

# ---------------- grid carbon intensity (Ember via OWID, latest year) ----------------
ow = pd.read_csv(D / "owid/owid-energy-data.csv", usecols=["iso_code", "year", "carbon_intensity_elec"]).dropna()
ow = ow.sort_values("year").groupby("iso_code").tail(1).set_index("iso_code")
c["grid_g_kwh"] = c.CNTRY_ISO.map(ow.carbon_intensity_elec).fillna(ow.carbon_intensity_elec.median())

# ---------------- irrigated land near each outfall (GMIA v5) ----------------
def read_asc(p):
    with open(p) as fh:
        hdr = {}
        for _ in range(6):
            k, v = fh.readline().split(); hdr[k.lower()] = float(v)
    a = np.loadtxt(p, skiprows=6, dtype=np.float32); a[a == hdr["nodata_value"]] = np.nan
    return a, hdr
aei, h = read_asc(D / "gmia/gmia_v5_aei_ha.asc")
cs = h["cellsize"]; nr, nc = aei.shape
lat_c = 90 - (np.arange(nr) + 0.5) * cs; lon_c = -180 + (np.arange(nc) + 0.5) * cs
def rc(lat, lon): return min(max(int((90 - lat) / cs), 0), nr - 1), min(max(int((lon + 180) / cs), 0), nc - 1)
def window(lat, lon, km):
    dlat = km / 111.0; dlon = km / (111.0 * max(np.cos(np.radians(lat)), 0.1))
    r0, c0 = rc(lat + dlat, lon - dlon); r1, c1 = rc(lat - dlat, lon + dlon)
    r0, r1 = min(r0, r1), max(r0, r1); c0, c1 = min(c0, c1), max(c0, c1)
    la = lat_c[r0:r1 + 1][:, None]; lo = lon_c[c0:c1 + 1][None, :]
    dist = 2 * 6371 * np.arcsin(np.sqrt(np.sin(np.radians(la - lat) / 2) ** 2 + np.cos(np.radians(lat)) * np.cos(np.radians(la)) * np.sin(np.radians(lo - lon) / 2) ** 2))
    return r0, c0, dist <= km
a25 = []; owner = {}; owner_oc = {}   # union of 25 km buffers per country: all coastal outfalls, and ocean outfalls only
for r in c.itertuples():
    r0, c0, m = window(r.LAT_OUT, r.LON_OUT, 25)
    a25.append(float(np.nansum(np.where(m, aei[r0:r0 + m.shape[0], c0:c0 + m.shape[1]], 0))))
    for ri, ci in zip(*np.where(m)):
        owner.setdefault((r0 + ri, c0 + ci), r.CNTRY_ISO)
        if r.ocean_outfall: owner_oc.setdefault((r0 + ri, c0 + ci), r.CNTRY_ISO)
c["aei_25km_ha"] = a25
def unite(own):
    u = {}
    for (ri, ci), iso in own.items():
        v = aei[ri, ci]
        if np.isfinite(v): u[iso] = u.get(iso, 0.0) + float(v)
    return u
union, union_oc = unite(owner), unite(owner_oc)

# ---------------- existing reuse (Jones et al. 2021, country level) ----------------
j = pd.read_csv(D / "jones21/Wastewater_production/Country_WWpctr_FlowRate.txt", sep="\t")
FIX = {"Channel Islands": "JEY", "Congo Dem Republic": "COD", "Congo Rep": "COG", "Cote d Ivoire": "CIV", "Korea Demo": "PRK",
       "Korea Rep": "KOR", "Lao Peoples Democratic Republic": "LAO", "Macedonia FYR": "MKD", "Micronesia (Fed. States of)": "FSM",
       "St. Kitts and Nevis": "KNA", "St. Lucia": "LCA", "St. Vincent and the Grenadines": "VCT", "Taiwan China": "TWN",
       "Virgin Islands (U.S.)": "VIR", "West Bank & Gaza": "PSE", "Turkey": "TUR", "Iran (Islamic Republic of)": "IRN",
       "Bolivia (Plurinational State of)": "BOL", "Curacao": "CUW", "Hong Kong SAR": "HKG", "Macao SAR": "MAC", "Swaziland": "SWZ",
       "Venezuela (Bolivarian Republic of)": "VEN"}
def iso3(n):   # exact names only; no fuzzy matching (fuzzy matching mis-assigned the Netherlands)
    if n in FIX: return FIX[n]
    try: return pycountry.countries.lookup(n).alpha_3
    except Exception: return None
j["iso"] = j.Country.map(iso3)
j = j.dropna(subset=["iso"]).drop_duplicates("iso").set_index("iso")
j["reuse_frac"] = (j.WWr_million_m3_yr / j.WWt_million_m3_yr.replace(0, np.nan)).clip(0, 1)
c["reuse_frac"] = c.CNTRY_ISO.map(j.reuse_frac).fillna(0.0)
c.to_csv(OUT / "plants.csv", index=False)

# ---------------- country context ----------------
w = pd.read_excel(D / "jmp/JMP_2025_WLD.xlsx", sheet_name="wat", usecols=["iso3", "year", "wat_sm_t", "pop_t"]).dropna(subset=["wat_sm_t"])
w = w.sort_values("year").groupby("iso3").tail(1).set_index("iso3")
ctx = pd.DataFrame(index=sorted(c.CNTRY_ISO.unique()))
ctx["country"] = c.groupby("CNTRY_ISO").COUNTRY.first()
ctx["jmp_sm_pct"] = w.wat_sm_t.reindex(ctx.index)
ctx["jmp_year"] = w.year.reindex(ctx.index)
ctx["pop_without_sm_M"] = (w.pop_t * (100 - w.wat_sm_t) / 100 / 1e3).reindex(ctx.index)
ctx["aei_25km_union_ha"] = pd.Series(union).reindex(ctx.index).fillna(0)
ctx["aei_25km_union_ocean_ha"] = pd.Series(union_oc).reindex(ctx.index).fillna(0)
ctx["reuse_frac"] = j.reuse_frac.reindex(ctx.index)
ctx["treated_jones21_Mm3d"] = (j.WWt_million_m3_yr / 365).reindex(ctx.index)
ctx["reused_jones21_Mm3d"] = (j.WWr_million_m3_yr / 365).reindex(ctx.index)
ctx.index.name = "iso"
ctx.to_csv(OUT / "country_context.csv")
pd.Series({"all_treated_Mm3d": all_flow / 1e6, "n_operational": len(d)}).to_csv(OUT / "world_totals.csv")
print(f"operational {len(d)}; coastal {len(c)} ({c.WASTE_DIS.sum()/1e6:.1f} Mm3/d); ocean outfalls {c.ocean_outfall.sum()} "
      f"({c[c.ocean_outfall].WASTE_DIS.sum()/1e6:.1f}); tiers {c.tier.value_counts().to_dict()}")
