"""Shared dashboard constants.

Kept free of Streamlit/Plotly imports so pure modules (fair_value_model,
optimize_fair_value, smoke tests) can reuse them without pulling UI
dependencies or triggering the Streamlit app module.
"""

from __future__ import annotations

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
    "AL THANYAH THIRD": "The Greens / Barsha Heights",
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
}

# Sidebar options — display names of the highest-volume districts in the data.
NEIGHBORHOODS: list[str] = [
    # High-volume
    "Dubai South (Expo City)",
    "Jumeirah Village Circle (JVC)",
    "Al Furjan / Discovery Gardens",
    "Business Bay",
    "Dubai Islands",
    "Dubai Marina",
    "Arjan",
    "Dubai Creek Harbour",
    "Dubai Production City",
    "Jumeirah Lakes Towers",
    # Mid-volume
    "Jumeirah Village Triangle (JVT)",
    "Sobha Hartland (MBR City)",
    "Dubai Sports City",
    "Motor City",
    "Town Square",
    "Downtown Dubai (Burj Khalifa)",
    "Dubai Hills Estate",
    "Al Jadaf",
    "Dubai Silicon Oasis",
    "Palm Jumeirah",
]

TIER_MAP: dict[str, str] = {}
TIER_AREAS: dict[str, list[str]] = {
    "Ultra-premium": [
        "Palm Jumeirah", "Downtown Dubai (Burj Khalifa)", "Za'abeel",
    ],
    "Premium": [
        "Dubai Marina", "Dubai Creek Harbour", "Dubai Hills Estate",
        "Sobha Hartland (MBR City)", "Sobha Hartland II", "Business Bay",
        "Dubai Maritime City", "Dubai Islands",
    ],
    "Mid-market": [
        "Jumeirah Lakes Towers", "Jumeirah Village Triangle (JVT)",
        "The Greens / Barsha Heights", "Al Jadaf", "Nad Al Sheba",
        "Town Square", "Al Furjan / Discovery Gardens", "DAMAC Hills",
    ],
    "Value": [
        "Jumeirah Village Circle (JVC)", "Arjan", "Motor City",
        "Dubai Silicon Oasis", "Dubai Production City", "Dubai Sports City",
        "Dubai South (Expo City)",
    ],
    "Budget": [
        "International City Phase 1", "International City (Warsan 4)",
        "DAMAC Hills 2", "DUBAI INVESTMENT PARK FIRST",
        "DUBAI INVESTMENT PARK SECOND",
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

# Official DLD district name -> market tier (via its display name).
DISTRICT_TIER: dict[str, str] = {
    district: TIER_MAP[display]
    for district, display in AREA_DISPLAY.items()
    if display in TIER_MAP
}

# DLD reports ACTUAL_AREA in square metres; the dashboard displays sqft.
SQM_TO_SQFT = 10.7639
