#!/usr/bin/env bash
# Full pipeline. Run from the repository root after placing the inputs listed in data/README.md.
set -euo pipefail
R=.
mkdir -p out
python scripts/fetch_power.py $R       # NASA POWER cells (cached in data/power_cells; skipped where present)
python scripts/join_aqueduct.py $R     # out/plant_aqueduct.csv
python scripts/build_plants.py $R      # out/plants.csv, country_context.csv, world_totals.csv, validation_effect.csv
python scripts/desal_inputs.py $R      # out/desal_world.csv, desal_regions.csv, desal_countries.csv
python scripts/ml_validate.py $R       # out/ml_skill.csv, ml_country_ratios.csv, flow_bootstrap.csv, flow_worstcase.csv
python scripts/model.py $R             # out/global_mc.csv, cascade*.csv, country_summary.csv, desal_comparison_*.csv, plants_model.csv, ...
python scripts/irrigation.py $R        # out/irrigation_summary.csv, irrigation_countries.csv
python scripts/reuse_audit.py $R       # out/reuse_audit.csv
python scripts/cities.py $R            # out/cities.csv
python scripts/costs.py $R             # out/breakeven*.csv
python scripts/city_uncertainty.py $R  # out/cities_uncertainty.csv, cities_route.csv
