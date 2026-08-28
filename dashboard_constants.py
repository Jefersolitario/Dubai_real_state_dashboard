"""Shared dashboard constants and pure helpers.

Kept free of Streamlit/Plotly imports so pure modules (fair_value_model,
optimize_fair_value, smoke tests) can reuse them without pulling UI
dependencies or triggering the Streamlit app module.
"""

from __future__ import annotations

import polars as pl

# DLD `AREA_EN` values are official district names (e.g. "MARSA DUBAI"),
# not the community names buyers know. Map them to friendly display names;
# districts not listed here fall back to the raw AREA_EN value.
AREA_DISPLAY: dict[str, str] = {
    "MARSA DUBAI": "Dubai Marina",
    "AL BARSHA SOUTH FOURTH": "Jumeirah Village Circle (JVC)",
    "AL BARSHA SOUTH FIFTH": "Jumeirah Village Triangle (JVT)",
    "AL BARSHAA SOUTH THIRD": "Arjan",
    "AL JADAF": "Al Jadaf",
    "AL THANYAH FIFTH": "Jumeirah Lakes Towers",
    "AL THANYAH THIRD": "The Greens",
    "AL THANYAH FIRST": "Barsha Heights (TECOM)",
    "AL KHAIRAN FIRST": "Dubai Creek Harbour",
    "AL MERKADH": "Sobha Hartland (MBR City)",
    "BUKADRA": "Sobha Hartland II",
    "HADAEQ SHEIKH MOHAMMED BIN RASHID": "Dubai Hills Estate",
    "AL HEBIAH FOURTH": "Dubai Sports City",
    "AL HEBIAH FIRST": "Motor City",
    "AL HEBIAH THIRD": "DAMAC Hills",
    "MADINAT HIND 4": "DAMAC Hills 2",
    "NADD HESSA": "Dubai Silicon Oasis",
    "ME'AISEM FIRST": "Dubai Production City",
    "AL WARSAN FIRST": "International City Phase 1",
    "WARSAN FOURTH": "International City (Warsan 4)",
    "MADINAT AL MATAAR": "Dubai South (Expo City)",
    "AL YELAYISS 2": "Town Square",
    "PALM DEIRA": "Dubai Islands",
    "MADINAT DUBAI ALMELAHEYAH": "Dubai Maritime City",
    "JABAL ALI FIRST": "Al Furjan / Discovery Gardens",
    "ZAABEEL SECOND": "Za'abeel",
    "NAD AL SHIBA FIRST": "Nad Al Sheba",
    "BUSINESS BAY": "Business Bay",
    "BURJ KHALIFA": "Downtown Dubai (Burj Khalifa)",
    "PALM JUMEIRAH": "Palm Jumeirah",
    # Verified against the live snapshot + Ejari artifacts (note the DLD
    # spellings: AL SAFOUH, AL BARSHAA SOUTH SECOND, TRADE CENTER).
    "AL SAFOUH FIRST": "Al Sufouh (Internet City)",
    "AL SAFOUH SECOND": "Al Sufouh (Internet City)",
    "AL BARSHA FIRST": "Al Barsha",
    "AL BARSHAA SOUTH SECOND": "Dubai Science Park (Al Barsha South 2)",
    "WADI AL SAFA 2": "Liwan",
    "WADI AL SAFA 3": "Majan (Dubailand)",
    "WADI AL SAFA 4": "Dubailand (Wadi Al Safa 4)",
    "WADI AL SAFA 5": "Dubai Residence Complex (Dubailand)",
    "AL WASL": "City Walk (Al Wasl)",
    "AL SATWA": "Al Satwa / Jumeirah Garden City",
    "AL KIFAF": "Al Kifaf (Wasl1)",
    "UM SUQAIM THIRD": "Madinat Jumeirah Living (Umm Suqeim 3)",
    "JUMEIRAH FIRST": "La Mer / Jumeirah 1",
    "JUMEIRAH SECOND": "Jumeirah Bay Island (Jumeirah 2)",
    "TRADE CENTER FIRST": "Trade Centre / DIFC",
    "TRADE CENTER SECOND": "Trade Centre / DIFC",
    "ZAABEEL FIRST": "Za'abeel",
    "RAS AL KHOR INDUSTRIAL FIRST": "Ras Al Khor",
    "AL HEBIAH SECOND": "Dubai Studio City",
    "SAIH SHUAIB 2": "Dubai Industrial City",
    "MIRDIF": "Mirdif",
    "PALM JABAL ALI": "Palm Jebel Ali",
    "DUBAI INVESTMENT PARK FIRST": "Dubai Investment Park (DIP)",
    "DUBAI INVESTMENT PARK SECOND": "Dubai Investment Park (DIP)",
    "AL GOZE FOURTH": "Al Quoz 4",
    "NAD AL HAMAR": "Nad Al Hamar",
    # Rental-market districts: huge in Ejari, negligible apartment sales.
    # Industrial/labor districts are deliberately unmapped — shared-housing
    # rents would skew zone medians.
    "AL NAHDA FIRST": "Al Nahda",
    "AL NAHDA SECOND": "Al Nahda",
    "AL KARAMA": "Al Karama",
    "AL MURQABAT": "Al Muraqqabat (Deira)",
    "AL WARQA FIRST": "Al Warqaa 1",
    "MANKHOOL": "Mankhool (Bur Dubai)",
    "AL QUSAIS": "Al Qusais",
    "MUHAISANAH FOURTH": "Muhaisnah 4",
    "AL RAFFA": "Al Raffa (Bur Dubai)",
    "AL SUQ AL KABEER": "Al Souk Al Kabeer (Bur Dubai)",
    "NAIF": "Naif (Deira)",
    "AL MUTEENA": "Al Muteena (Deira)",
    "HOR AL ANZ": "Hor Al Anz",
    "HOR AL ANZ EAST": "Hor Al Anz",
    "AL HAMRIYA": "Al Hamriya (Bur Dubai)",
    "OUD METHA": "Oud Metha",
    "AL MAMZER": "Al Mamzar",
    "AL BARAHA": "Al Baraha (Deira)",
    "AL BADA": "Al Bada'a",
    "PORT SAEED": "Port Saeed (Deira)",
}

# Sidebar options — every area with a friendly display name, A-Z.
# Derived from AREA_DISPLAY so the picker can never miss a mapped area.
NEIGHBORHOODS: list[str] = sorted(set(AREA_DISPLAY.values()))

# Tiers cover areas with meaningful sales volume, calibrated to each tier's
# 12-month median sales PSF (Ultra ~3,150 / Premium ~2,500 / Mid ~1,770 /
# Value ~1,500 / Budget ~1,100 AED/sqft). Rental-only districts (Al Karama,
# Al Nahda, ...) carry no tier — tiers feed sales charts and the model only.
TIER_MAP: dict[str, str] = {}
TIER_AREAS: dict[str, list[str]] = {
    "Ultra-premium": [
        "Palm Jumeirah", "Downtown Dubai (Burj Khalifa)", "Za'abeel",
        "City Walk (Al Wasl)", "La Mer / Jumeirah 1",
        "Jumeirah Bay Island (Jumeirah 2)", "Trade Centre / DIFC",
        "Palm Jebel Ali",
    ],
    "Premium": [
        "Dubai Marina", "Dubai Creek Harbour", "Dubai Hills Estate",
        "Sobha Hartland (MBR City)", "Sobha Hartland II", "Business Bay",
        "Dubai Maritime City", "Dubai Islands",
        "Madinat Jumeirah Living (Umm Suqeim 3)",
        "Al Satwa / Jumeirah Garden City", "Al Kifaf (Wasl1)", "Ras Al Khor",
    ],
    "Mid-market": [
        "Jumeirah Lakes Towers", "Jumeirah Village Triangle (JVT)",
        "The Greens", "Al Jadaf", "Nad Al Sheba",
        "Town Square", "Al Furjan / Discovery Gardens", "DAMAC Hills",
        "Al Barsha", "Al Sufouh (Internet City)",
        "Dubai Science Park (Al Barsha South 2)", "Dubailand (Wadi Al Safa 4)",
    ],
    "Value": [
        "Jumeirah Village Circle (JVC)", "Arjan", "Motor City",
        "Dubai Silicon Oasis", "Dubai Production City", "Dubai Sports City",
        "Dubai South (Expo City)",
        "Barsha Heights (TECOM)", "Majan (Dubailand)",
        "Dubai Residence Complex (Dubailand)", "Dubai Studio City",
        "Dubai Industrial City", "Liwan", "Mirdif",
    ],
    "Budget": [
        "International City Phase 1", "International City (Warsan 4)",
        "DAMAC Hills 2", "Dubai Investment Park (DIP)",
        "Al Quoz 4", "Nad Al Hamar",
    ],
}
for tier, areas in TIER_AREAS.items():
    for a in areas:
        TIER_MAP[a] = tier
TIER_ORDER = ["Ultra-premium", "Premium", "Mid-market", "Value", "Budget"]
TIER_COLORS = {
    "Ultra-premium": "#e377c2",
    "Premium":       "#ff7f0e",
    "Mid-market":    "#636efa",
    "Value":         "#00cc96",
    "Budget":        "#ffa15a",
}

# Official DLD district name -> market tier. Built from the display-name
# mapping, plus TIER_AREAS entries that are already official district names
# (e.g. DUBAI INVESTMENT PARK FIRST has no friendly display name).
DISTRICT_TIER: dict[str, str] = {
    district: TIER_MAP[display]
    for district, display in AREA_DISPLAY.items()
    if display in TIER_MAP
}
for _area, _tier in TIER_MAP.items():
    if _area == _area.upper() and _area not in DISTRICT_TIER:
        DISTRICT_TIER[_area] = _tier

# DLD reports ACTUAL_AREA in square metres; the dashboard displays sqft.
SQM_TO_SQFT = 10.7639


def area_display_expr() -> pl.Expr:
    """AREA_EN mapped to its friendly display name (unmapped pass through)."""
    return pl.col("AREA_EN").replace(AREA_DISPLAY)


def bedroom_type_expr() -> pl.Expr:
    """ROOMS_EN normalized to the dashboard bedroom labels (1 B/R -> 1BR)."""
    return pl.col("ROOMS_EN").cast(pl.Utf8).str.replace(" B/R", "BR")


def y_cap_range(values: pl.Series, pad: float = 1.1) -> list[float] | None:
    """[0, p99*pad] axis range when extreme outliers would squash the chart.

    Returns None (Plotly autorange) unless the max exceeds 1.5x the 99th
    percentile, so ordinary data keeps the automatic axis. "lower"
    interpolation keeps the quantile below the max even for small samples,
    where "nearest" would land on the max itself and never trigger.
    """
    if values.len() == 0:
        return None
    cap = values.quantile(0.99, interpolation="lower")
    vmax = values.max()
    if cap is None or vmax is None or vmax <= cap * 1.5:
        return None
    return [0.0, cap * pad]


def layout_defaults(title: str) -> dict:
    """Shared dark-theme Plotly layout. Must not include margin —
    individual charts set margins themselves (see CLAUDE.md)."""
    return dict(
        title=dict(text=title, font=dict(size=13), x=0.01),
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font=dict(family="Segoe UI, Arial, sans-serif", size=11, color="#fafafa"),
    )
