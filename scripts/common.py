"""Shared definitions: sea-region boxes and plant scopes used by ml_validate.py and model.py."""
# Mediterranean: 30-46 N, 5.6 W (Gibraltar) to 37 E, excluding the Black Sea (north of 41 N and east of 27.5 E)
MED = lambda g: g.LAT_OUT.between(30, 46) & g.LON_OUT.between(-5.6, 37) & ~((g.LAT_OUT > 41) & (g.LON_OUT > 27.5))
# Red Sea (12.5-29.9 N, 32.3-43.5 E, south of the Suez Canal) and Persian Gulf with Gulf of Oman (22-31 N, 47-60 E)
GULF_RED = lambda g: (g.LAT_OUT.between(12.5, 29.9) & g.LON_OUT.between(32.3, 43.5)) | (g.LAT_OUT.between(22, 31) & g.LON_OUT.between(47, 60))

def scopes(c):
    """Boolean masks over the coastal plant table for every reported scope."""
    oc = c.ocean_outfall
    return {"coastal_50km": c.index == c.index, "ocean_outfalls": oc, "river_outfalls_50km": ~oc,
            "ocean_stressed": oc & c.stressed, "ocean_tierA": oc & (c.tier == "A"), "coastal_tierA": c.tier == "A",
            "ocean_Mediterranean": oc & MED(c), "ocean_Gulf_RedSea": oc & GULF_RED(c)}
