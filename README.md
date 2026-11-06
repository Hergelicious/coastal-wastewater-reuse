# coastal-reclaim

Analysis code for *Coastal cities discharge as much treated wastewater to the sea as the world can desalinate*.

The code takes every wastewater treatment plant in HydroWASTE v1.0 that discharges within 50 km of the sea, applies corrections from checks against utility sources, separates sea or estuary outfalls from river outfalls, converts effluent to potable product with a closed mass balance, and compares it with desalination capacity at world, regional, national and city scale. It estimates electricity, photovoltaic capacity, brine and CO₂ of reclamation against seawater reverse osmosis, applies a substitution cascade of practical constraints, matches irrigation supply and demand locally, computes a break-even conveyance distance, quantifies uncertainty in city ratios, and tests how far plant flow estimates transfer between countries.


## Layout

```
run_all.sh                 full pipeline, in order
scripts/
  fetch_power.py           NASA POWER climatology (irradiance, temperature, precipitation) for each 0.5° cell with a coastal outfall
  join_aqueduct.py         Aqueduct 4.0 baseline, monthly and projected (2030, 2050, 2080) water stress at each outfall
  build_plants.py          coastal plant set, validation corrections, outfall type, data tier, irradiance, irrigation
                           requirement, water stress, grid intensity, irrigated area, national reuse fraction
  desal_inputs.py          desalination reference values: world, 8 regions, 30 countries (year and source per row)
  common.py                sea-region boxes and plant scopes shared by ml_validate.py and model.py
  ml_validate.py           flow validation (random, country-grouped and source-grouped CV) and country-correlated bootstrap
  model.py                 mass balance, energy, PV, brine, CO2; Monte Carlo; cascade with size and reuse sensitivity;
                           country and regional comparison; future and seasonal water stress; sensitivity
  irrigation.py            irrigation substitution matched within clusters of outfalls (radius and crop coefficient sensitivity)
  reuse_audit.py           audit of the existing-reuse subtraction
  cities.py                city matching of seawater desalination capacity and nearby eligible effluent (23 cities)
  costs.py                 break-even conveyance distance (Monte Carlo) and plant-to-city distances
  city_uncertainty.py      Monte Carlo intervals and conservative or optimistic cases for city ratios; route distances
inputs/
  validation_corrections.csv   corrections from checks of the 25 largest ocean outfalls against utility sources
  validation_top25.csv         the checks themselves: reported flows, receiving water, treatment, sources
  city_desal_sources.csv       desalination plants serving 23 cities: capacity, feedwater, status, source URL
data/README.md             where to obtain each input dataset (not redistributed)
out/                       created by the pipeline
```
The repository also includes `Supplementary_Software_1_atlas.html`, 
an interactive atlas for exploring the coastal wastewater treatment 
plants and their geographic distribution.
## Run

```
pip install -r requirements.txt
bash run_all.sh
```

Every script takes the repository root as its only argument, reads `data/` and `inputs/`, and writes `out/`. After the NASA POWER cells are cached (the first run makes about 3,000 API calls), a full run takes a few minutes. Random seeds are fixed (Monte Carlo 42, cost model 42, bootstrap 7, city Monte Carlo 11), so outputs are reproducible. Tested with Python 3.13.

## Key definitions

- Coastal plant: HydroWASTE status not Closed, Decommissioned, Non-Operational, Projected, Proposed or Under Construction; `COAST_50KM = 1` and estimated outfall within 50 km of the Natural Earth 1:10 m ocean coastline.
- Ocean outfall: `RIVER_DIS` empty (estimated outfall is the sea or an estuary), after the corrections in `inputs/validation_corrections.csv`.
- Data tiers: A reported flow (`QUAL_WASTE` 1 to 3); B flow from reported population (`QUAL_WASTE` 4, `QUAL_POP` 1 or 2); C modelled population and flow.
- Mass balance: feed = 0.90 × effluent (0.80 to 0.95); potable = 0.80 × feed (0.70 to 0.85); irrigation grade = 0.95 × feed.
- Electricity: potable reuse 1.0 kWh/m³ (0.8 to 1.5) plus 0.3 kWh/m³ feed for primary-only plants; seawater RO 3.5 kWh/m³ (3.0 to 4.0).
- Irrigation requirement: Hargreaves radiation ET0 = 0.0135 (T + 17.8) Rs, with Rs = POWER irradiance (kWh m⁻² d⁻¹) × 3.6 / 2.45 in mm d⁻¹; crop coefficient 1.0 (0.8 to 1.2); USDA SCS effective rainfall; efficiency 0.65.
- Parameters are defined at the top of `model.py` (process), `costs.py` (costs) and `city_uncertainty.py` (city Monte Carlo) as (central, low, high).

## Main outputs (`out/`)

| File | Content |
|---|---|
| global_mc.csv | totals by scope, medians and 90% intervals |
| cascade.csv, cascade_size_sensitivity.csv, cascade_reuse_sensitivity.csv | substitution cascade and its sensitivity |
| desal_comparison_countries.csv, desal_comparison_regions.csv | comparison with desalination capacity |
| country_summary.csv | country results at central parameters |
| irrigation_summary.csv, irrigation_countries.csv | locally matched irrigation substitution |
| cities.csv, cities_uncertainty.csv, cities_route.csv | city matching, intervals and route distances |
| breakeven.csv, breakeven_by_price.csv, breakeven_cities.csv | break-even conveyance distance |
| ml_skill.csv, ml_country_ratios.csv, flow_bootstrap.csv, flow_worstcase.csv | flow validation and bootstrap |
| validation_effect.csv, reuse_audit.csv | effect of corrections; reuse audit |
| future_stress.csv, future_stress_countries.csv, seasonal.csv | projected and monthly water stress |
| treatment_sensitivity.csv, tornado_e_saved.csv | sensitivity |
| plants_model.csv | plant-level results |

