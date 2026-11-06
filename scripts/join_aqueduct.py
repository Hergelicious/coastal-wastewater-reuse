"""Join every HydroWASTE outfall to its WRI Aqueduct 4.0 baseline-annual sub-basin, then attach the projected
water-stress categories (future_annual: business as usual, optimistic and pessimistic scenarios for 2030, 2050 and 2080)
and the monthly baseline water stress (baseline_monthly) of the same HydroBASINS level-6 basin (pfaf_id).
Writes out/plant_aqueduct.csv (WASTE_ID, bws_raw, bws_score, bws_cat, bws_label, fut_*, bws_m01..bws_m12, ...)."""
import sys, pathlib, pyogrio, geopandas as gpd, pandas as pd
ROOT = pathlib.Path(sys.argv[1]); D = ROOT / "data"; OUT = ROOT / "out"; OUT.mkdir(exist_ok=True)
gdb = D / "aqueduct/Aqueduct40_waterrisk_download_Y2023M07D05/GDB/Aq40_Y2023D07M05.gdb"
cols = ["string_id", "pfaf_id", "gid_0", "name_0", "bws_raw", "bws_score", "bws_cat", "bws_label", "w_awr_def_tot_cat", "w_awr_def_tot_label"]
aq = pyogrio.read_dataframe(gdb, layer="baseline_annual", columns=cols)
d = pd.read_csv(D / "hw/HydroWASTE_v10.csv", encoding="latin-1")
p = gpd.GeoDataFrame(d[["WASTE_ID", "LAT_OUT", "LON_OUT"]], geometry=gpd.points_from_xy(d.LON_OUT, d.LAT_OUT), crs="EPSG:4326")
j = gpd.sjoin(p, aq, how="left", predicate="within").drop_duplicates("WASTE_ID")
un = j[j.bws_cat.isna()][["WASTE_ID", "geometry"]]
if len(un):   # outfalls just offshore: nearest sub-basin within 0.2 degrees
    nn = gpd.sjoin_nearest(gpd.GeoDataFrame(un, geometry="geometry", crs="EPSG:4326"), aq, how="left", max_distance=0.2).drop_duplicates("WASTE_ID").set_index("WASTE_ID")
    j = j.set_index("WASTE_ID"); j.loc[nn.index, cols] = nn[cols].values; j = j.reset_index()
fc = [f"{s}{y}_ws_x_{k}" for s in ("bau", "opt", "pes") for y in (30, 50, 80) for k in ("r", "c")]
fut = pyogrio.read_dataframe(gdb, layer="future_annual", columns=["pfaf_id"] + fc, read_geometry=False).drop_duplicates("pfaf_id")
fut = fut.rename(columns={c: "fut_" + c.replace("_ws_x", "") for c in fc})
mc = [f"bws_{m:02d}_raw" for m in range(1, 13)]
mon = pyogrio.read_dataframe(gdb, layer="baseline_monthly", columns=["pfaf_id"] + mc, read_geometry=False).drop_duplicates("pfaf_id")
mon = mon.rename(columns={f"bws_{m:02d}_raw": f"bws_m{m:02d}" for m in range(1, 13)})
j = j.merge(fut, on="pfaf_id", how="left").merge(mon, on="pfaf_id", how="left")
j.drop(columns=["geometry", "index_right"], errors="ignore").to_csv(OUT / "plant_aqueduct.csv", index=False)
print("future matched", j.fut_bau50_c.notna().sum(), "monthly matched", j.bws_m01.notna().sum())
print("matched", j.bws_cat.notna().sum(), "of", len(j))
