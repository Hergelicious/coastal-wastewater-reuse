"""Desalination reference values (world, regions, 30 countries), each with year and source.
Writes out/desal_world.csv, out/desal_regions.csv, out/desal_countries.csv."""
import sys, pathlib, pandas as pd
from docx import Document
ROOT = pathlib.Path(sys.argv[1]); D = ROOT / "data"; OUT = ROOT / "out"; OUT.mkdir(exist_ok=True)

world = pd.DataFrame([
    ["operational_2018", 95.37, 2018, "Jones et al. 2019, Table 2 (GWI DesalData 2018)"],
    ["installed_2020", 97.2, 2020, "Eke et al. 2020 (GWI DesalData, February 2020; 16,876 installed plants)"],
    ["installed_2024", 91.5, 2024, "IDRA figure as reported by the EU Blue Economy Observatory (2025); primary IDRA document not seen"],
    ["contracted_2024", 105.0, 2024, "IDRA, installed plus contracted capacity 2024"],
    ["operating_2024_upper", 135.0, 2024, "EU Blue Economy Observatory, desalination sector page (2025); unsourced operating-capacity estimate, used as an upper bound"],
    ["seawater_share", 0.57, 2020, "Eke et al. 2020: seawater is 57% of global installed capacity (February 2020)"],
], columns=["key", "value", "year", "source"])
world.to_csv(OUT / "desal_world.csv", index=False)

regions = pd.DataFrame([
    ["Middle East and North Africa", 45.32], ["East Asia and Pacific", 17.52], ["North America", 11.34], ["Western Europe", 8.75],
    ["Latin America and Caribbean", 5.46], ["Southern Asia", 2.94], ["Eastern Europe and Central Asia", 2.26], ["Sub-Saharan Africa", 1.78]],
    columns=["region", "desal_Mm3d"])
regions["year"] = 2018; regions["source"] = "Jones et al. 2019, Table 2"
regions.to_csv(OUT / "desal_regions.csv", index=False)

rows = []
t = Document(D / "frontiers/table_1.docx").tables[5]
mena = {"Bahrain": "BHR", "Djibouti": "DJI", "Egypt": "EGY", "Eritrea": "ERI", "Iran": "IRN", "Iraq": "IRQ", "Israel": "ISR", "Jordan": "JOR",
        "Kuwait": "KWT", "Oman": "OMN", "Palestine": "PSE", "Qatar": "QAT", "Saudi Arabia": "SAU", "Somalia": "SOM", "Sudan": "SDN", "UAE": "ARE", "Yemen": "YEM"}
for r in t.rows[1:]:
    cl = [x.text.strip() for x in r.cells]
    rows.append([mena[cl[0]], cl[0], float(cl[2]), 2022, "Frontiers in Water 7:1672360 (2025), Table S6; GWI DesalData 2022"])
for iso, name, sh in [("USA", "United States", 11.2), ("CHN", "China", 7.5), ("ESP", "Spain", 5.7)]:
    rows.append([iso, name, sh / 100 * 95.37, 2018, "Jones et al. 2019, share of 95.37 Mm3/d; GWI DesalData 2018"])
for iso, name, v in [("IND", "India", 2.87), ("DZA", "Algeria", 2.76), ("MAR", "Morocco", 0.27), ("ZAF", "South Africa", 0.45),
                     ("PAK", "Pakistan", 0.1905), ("NGA", "Nigeria", 0.1289)]:
    rows.append([iso, name, v, 2020, "Dhakal et al. 2022, Membranes 12:381; DesalData 2020"])
rows += [["AUS", "Australia", 1.605, 2024, "Sum of operating or standby plants from utility sources (inputs/city_desal_sources.csv): Kwinana 123, Southern Seawater 274, Kurnell 250, Victorian 411, Adelaide 274, Tugun 133 ML/d, plus the industrial Cape Preston plant 140 ML/d"],
         ["SGP", "Singapore", 0.864, 2025, "Sum of the five operating plants from PUB and operator sources (inputs/city_desal_sources.csv)"],
         ["CHL", "Chile", 0.708, 2023, "Consejo Minero and ACADES national inventory, April 2023: 22 plants, 8,200 L/s"],
         ["MEX", "Mexico", 0.750, 2022, "Energies 15:8434 (2022): 749,751 m3/d in 351 plants; GWI DesalData 2022"]]
pd.DataFrame(rows, columns=["iso", "country", "desal_Mm3d", "year", "source"]).drop_duplicates("iso").to_csv(OUT / "desal_countries.csv", index=False)
print("ok", len(rows))
