"""City-level matching: reclaimable effluent of plants within RADIUS km of a city centre against the seawater
desalination capacity that serves the city. Capacities of operating or standby seawater plants from operator, utility or
government sources (inputs/city_desal_sources.csv). Cities: those served by seawater desalination plants of at least about
50,000 m3/d, chosen to cover each region with large desalination capacity (Australia, United States, Europe, North Africa,
Middle East, South and East Asia). Effluent: ocean-outfall plants passing the cascade filters (secondary or advanced
treatment, at least 10,000 m3/d).
Plants must lie in the same country as the city. Classes (net potable product from ocean outfalls / desalination): >= 1 reclamation-first; 0.25 to 1 complementary;
< 0.25 desalination-first. Writes out/cities.csv."""
import sys, pathlib, numpy as np, pandas as pd
ROOT = pathlib.Path(sys.argv[1]); OUT = ROOT / "out"; RADIUS = 50
CITIES = [  # city, iso, lat, lon, desal m3/d, plants included
    ("Perth", "AUS", -31.95, 115.86, 400000, "Kwinana 130 ML/d; Southern Seawater (Binningup) 270 ML/d"),
    ("Sydney", "AUS", -33.87, 151.21, 250000, "Kurnell 250 ML/d"),
    ("Melbourne", "AUS", -37.81, 144.96, 410000, "Victorian Desalination Plant 410 ML/d"),
    ("Adelaide", "AUS", -34.93, 138.60, 300000, "Port Stanvac 300 ML/d"),
    ("Gold Coast", "AUS", -28.02, 153.40, 125000, "Tugun 125 ML/d"),
    ("San Diego", "USA", 32.72, -117.16, 189000, "Carlsbad 50 MGD"),
    ("Tampa", "USA", 27.95, -82.46, 95000, "Tampa Bay 25 MGD"),
    ("London", "GBR", 51.51, -0.13, 150000, "Beckton 150,000 m3/d"),
    ("Barcelona", "ESP", 41.39, 2.17, 200000, "Llobregat 200,000 m3/d"),
    ("Alicante", "ESP", 38.35, -0.48, 355000, "Alicante II 65,000; Torrevieja 240,000; Mutxamel 50,000 m3/d"),
    ("Algiers", "DZA", 36.75, 3.06, 320000, "Hamma 200,000; Fouka 120,000 m3/d"),
    ("Oran", "DZA", 35.70, -0.63, 590000, "Magtaa 500,000; Arzew 90,000 m3/d"),
    ("Tel Aviv", "ISR", 32.08, 34.78, 1493000, "Sorek 625,000; Palmachim 246,000; Ashdod 274,000; Hadera 348,000 m3/d"),
    ("Chennai", "IND", 13.08, 80.27, 200000, "Minjur 100,000; Nemmeli 100,000 m3/d"),
    ("Tianjin", "CHN", 39.13, 117.20, 200000, "Beijiang 200,000 m3/d"),
    ("Singapore", "SGP", 1.35, 103.82, 864000, "Five plants, 190 MGD (PUB)"),
    ("Dubai", "ARE", 25.20, 55.27, 2227600, "Jebel Ali 2,227,600 m3/d"),
    ("Abu Dhabi", "ARE", 24.45, 54.38, 2691000, "Taweelah RO, A1, A2, B; Umm Al Nar"),
    ("Kuwait City", "KWT", 29.37, 47.98, 2238000, "Shuaiba N and S, Az-Zour, Shuwaikh, Doha E and W, Subiya"),
    ("Jeddah", "SAU", 21.49, 39.19, 364000, "Jeddah (SWCC) 364,000 m3/d"),
    ("Agadir", "MAR", 30.43, -9.60, 400000, "Chtouka Ait Baha 400,000 m3/d"),
    ("Larnaca", "CYP", 34.92, 33.62, 64000, "Larnaca 64,000 m3/d"),
    ("Malta", "MLT", 35.90, 14.51, 124000, "Pembroke 54,000; Ghar Lapsi II 50,000; Ghar Lapsi 20,000 m3/d"),
]
# capacities from operator, utility or government sources (inputs/city_desal_sources.csv): operating or standby plants
src = pd.read_csv(ROOT / "inputs/city_desal_sources.csv")
src = src[src.include].assign(capacity_m3d=lambda d: d.capacity_m3d.astype(float))
src["city"] = src.city.replace({"Alicante area": "Alicante", "Tel Aviv area": "Tel Aviv"})
CAP = src.groupby("city").capacity_m3d.sum(); PLANTS = src.groupby("city").plant.apply(lambda s: "; ".join(s))
FEED = {"London": "brackish (tidal Thames)", "Tampa": "seawater (Tampa Bay estuary)"}
CITIES = [(n, i, la, lo, float(CAP.get(n, d)), PLANTS.get(n, pl)) for n, i, la, lo, d, pl in CITIES]
p = pd.read_csv(OUT / "plants_model.csv")
def hav(lat, lon, la, lo):
    return 2 * 6371 * np.arcsin(np.sqrt(np.sin(np.radians(la - lat) / 2) ** 2 + np.cos(np.radians(lat)) * np.cos(np.radians(la)) * np.sin(np.radians(lo - lon) / 2) ** 2))
SIZE_MIN = 10000   # as the substitution cascade (model.py)
rows = []
for name, iso, lat, lon, desal, plants in CITIES:
    dist = hav(lat, lon, p.LAT_OUT.values, p.LON_OUT.values)
    g = p[(dist <= RADIUS) & (p.CNTRY_ISO == iso)]; o = g[g.ocean_outfall]   # same country only (Singapore: excludes Johor and Batam)
    e = o[(o.LEVEL != "Primary") & (o.WASTE_DIS >= SIZE_MIN)]                  # eligible: cascade treatment and size filters
    rows.append(dict(city=name, iso=iso, lat=lat, lon=lon, desal_m3d=desal, desal_plants=plants, feedwater=FEED.get(name, "seawater"),
                     n_plants=len(g), n_ocean=len(o), n_eligible=len(e),
                     effluent_m3d=g.WASTE_DIS.sum(), ocean_effluent_m3d=o.WASTE_DIS.sum(), ocean_potable_m3d=o.pot_m3d.sum(),
                     ocean_potable_net_m3d=o.pot_net_m3d.sum(), eligible_potable_m3d=e.pot_m3d.sum(), eligible_potable_net_m3d=e.pot_net_m3d.sum(),
                     tierC_share=o[o.tier == "C"].WASTE_DIS.sum() / max(o.WASTE_DIS.sum(), 1), stressed_share=o[o.stressed].WASTE_DIS.sum() / max(o.WASTE_DIS.sum(), 1),
                     reuse_frac=g.reuse_frac.iloc[0] if len(g) else np.nan))
T = pd.DataFrame(rows)
T["ratio_all"] = T.ocean_potable_m3d / T.desal_m3d                # all ocean-outfall plants, gross
T["ratio_gross"] = T.eligible_potable_m3d / T.desal_m3d          # eligible plants, gross
T["ratio_net_all"] = T.ocean_potable_net_m3d / T.desal_m3d
T["ratio_net"] = T.eligible_potable_net_m3d / T.desal_m3d      # headline: eligible plants, net of reuse
T["class"] = np.select([T.feedwater.str.startswith("brackish"), T.ratio_net >= 1, T.ratio_net >= 0.25],
                       ["Not compared (brackish feed)", "Reclamation-first", "Complementary"], "Desalination-first")
T = T.sort_values("ratio_net", ascending=False)
T.to_csv(OUT / "cities.csv", index=False)
pd.set_option("display.width", 250)
print(T[["city", "desal_m3d", "feedwater", "n_ocean", "n_eligible", "ocean_potable_m3d", "eligible_potable_net_m3d", "ratio_gross", "ratio_net_all", "ratio_net", "class", "tierC_share", "reuse_frac"]].round(2).to_string())
print(T["class"].value_counts().to_dict())
