"""Audit of the existing-reuse subtraction (Jones et al. 2021 country fractions) for countries with ocean outfalls.
Flags countries whose reported reuse equals or exceeds reported treated volume (fraction capped at 1).
Writes out/reuse_audit.csv."""
import sys, pathlib, numpy as np, pandas as pd
OUT = pathlib.Path(sys.argv[1]) / "out"
ctx = pd.read_csv(OUT / "country_context.csv", index_col=0); C = pd.read_csv(OUT / "country_summary.csv", index_col=0)
A = ctx[["country", "treated_jones21_Mm3d", "reused_jones21_Mm3d"]].join(C[["ocean_effluent_Mm3d", "ocean_potable_Mm3d", "ocean_potable_net_Mm3d"]])
A = A[A.ocean_effluent_Mm3d > 0]
A["raw_ratio"] = A.reused_jones21_Mm3d / A.treated_jones21_Mm3d
A["reuse_frac_used"] = A.raw_ratio.clip(0, 1).fillna(0)
A["flag"] = np.select([A.treated_jones21_Mm3d.isna(), A.raw_ratio >= 0.999], ["no Jones 2021 entry; zero reuse assumed", "reuse >= treated; capped at 1"], "")
A["hydrowaste_vs_jones_treated"] = A.ocean_effluent_Mm3d / A.treated_jones21_Mm3d
A.sort_values("ocean_potable_Mm3d", ascending=False).to_csv(OUT / "reuse_audit.csv")
print(A[A.flag != ""].sort_values("ocean_potable_Mm3d", ascending=False).round(3).to_string())
print("potable removed by reuse subtraction:", round(A.ocean_potable_Mm3d.sum() - A.ocean_potable_net_Mm3d.sum(), 2),
      "of which capped countries:", round(A[A.flag.str.startswith("reuse")].ocean_potable_Mm3d.sum(), 2))
