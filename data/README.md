# Input data (not redistributed)

Place each dataset in the folder named below before running the scripts.

| Folder | Dataset | Source |
|---|---|---|
| data/hw/HydroWASTE_v10.csv | HydroWASTE v1.0 | https://doi.org/10.6084/m9.figshare.14847786 (CC BY 4.0) |
| data/aqueduct/Aqueduct40_waterrisk_download_Y2023M07D05/GDB/ | WRI Aqueduct 4.0 geodatabase | https://www.wri.org/data/aqueduct-global-maps-40-data |
| data/gmia/gmia_v5_aei_ha.asc, gmia_v5_aeinc_pct_aei.asc, gmia_v5_aai_pct_aei.asc | FAO GMIA v5 (5 arc-min ASCII grids) | https://www.fao.org/aquastat/en/geospatial-information/global-maps-irrigated-areas/latest-version |
| data/jmp/JMP_2025_WLD.xlsx | WHO/UNICEF JMP 2025 world file | https://washdata.org/data |
| data/jones21/Wastewater_production/Country_WWpctr_FlowRate.txt | Jones et al. 2021 country wastewater table | https://doi.org/10.1594/PANGAEA.918731 |
| data/owid/owid-energy-data.csv | Our World in Data energy dataset (Ember grid intensity, GDP, population) | https://github.com/owid/energy-data |
| data/ne/ne_10m_coastline.geojson, ne_110m_countries.geojson | Natural Earth | https://github.com/nvkelso/natural-earth-vector (public domain) |
| data/frontiers/table_1.docx | Supplementary material of Frontiers in Water 7:1672360 (2025), Table S6 | https://doi.org/10.3389/frwa.2025.1672360 |
| data/power_cells/ | NASA POWER climatology, fetched by scripts/fetch_power.py | https://power.larc.nasa.gov |

City desalination capacities (scripts/cities.py) and national values for Australia, Singapore, Chile and Mexico (scripts/desal_inputs.py) are written in the scripts with their sources; city capacities are read from inputs/city_desal_sources.csv.
