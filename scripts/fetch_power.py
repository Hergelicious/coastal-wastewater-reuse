"""Fetch NASA POWER climatology (ALLSKY_SFC_SW_DWN, T2M, PRECTOTCORR) for 0.5-degree cells holding coastal plants."""
import sys, json, time, pathlib, urllib.request
import numpy as np, pandas as pd
from concurrent.futures import ThreadPoolExecutor
ROOT = pathlib.Path(sys.argv[1]); D = ROOT / "data"; C = D / "power_cells"; C.mkdir(exist_ok=True)
d = pd.read_csv(D / "hw/HydroWASTE_v10.csv", encoding="latin-1")
d = d[d.COAST_50KM == 1]
cells = sorted(set(zip((d.LAT_OUT * 2).round() / 2, (d.LON_OUT * 2).round() / 2)))
print("cells", len(cells), flush=True)
def get(c):
    lat, lon = c; f = C / f"{lat:.1f}_{lon:.1f}.json"
    if f.exists():
        try:
            if "PRECTOTCORR" in json.load(open(f))["properties"]["parameter"]: return 1
        except Exception: pass
    url = ("https://power.larc.nasa.gov/api/temporal/climatology/point?parameters=ALLSKY_SFC_SW_DWN,T2M,PRECTOTCORR"
           f"&community=RE&longitude={lon}&latitude={lat}&format=JSON")
    for i in range(5):
        try:
            with urllib.request.urlopen(url, timeout=90) as r: j = json.load(r)
            json.dump(j, open(f, "w")); return 1
        except Exception as e:
            time.sleep(4 * (i + 1))
    return 0
done = 0
with ThreadPoolExecutor(5) as ex:
    for i, ok in enumerate(ex.map(get, cells)):
        done += ok
        if i % 200 == 0: print(i, done, flush=True)
print("finished", done, "of", len(cells))
