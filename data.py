from pathlib import Path
import json
import time
import zipfile

import numpy as np
import pandas as pd
import requests

DATA = Path(__file__).parent / "data"
HEADERS = {"User-Agent": "helvegrid/0.1"}

OSM_GRID = "https://zenodo.org/api/records/23896564/files/{}/content"
SWISSGRID = "https://www.swissgrid.ch/dam/dataimport/energy-statistic/EnergieUebersichtCH-{}.xlsx"
BFE_PLANTS = ("https://data.geo.admin.ch/ch.bfe.elektrizitaetsproduktionsanlagen/"
              "elektrizitaetsproduktionsanlagen/elektrizitaetsproduktionsanlagen_2056.csv.zip")
ENERGY_CHARTS = "https://api.energy-charts.info/{}"
GEO_ADMIN = "https://api3.geo.admin.ch/rest/services/api/MapServer/identify"


def download(url, name, params=None):
    path = DATA / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        for wait in (0, 10, 30, 60):
            time.sleep(wait)
            r = requests.get(url, params=params, headers=HEADERS, timeout=600)
            if r.status_code != 429:
                break
        r.raise_for_status()
        path.write_bytes(r.content)
    return path


def osm_grid():
    """Buses, lines and transformers of the European grid from OpenStreetMap (PyPSA-Eur, Zenodo)."""
    return {name: pd.read_csv(download(OSM_GRID.format(f"{name}.csv"), f"osm/{name}.csv"), index_col=0, quotechar="'")
            for name in ("buses", "lines", "transformers")}


def swissgrid(year):
    """Swissgrid energy overview in MW per hour."""
    cache = DATA / f"swissgrid_{year}.pkl"
    if cache.exists():
        return pd.read_pickle(cache)
    df = pd.read_excel(download(SWISSGRID.format(year), f"swissgrid_{year}.xlsx"),
                       sheet_name="Zeitreihen0h15", skiprows=[1], index_col=0)
    start = pd.to_datetime(df.index[0], format="%d.%m.%Y %H:%M").tz_localize("Europe/Zurich")
    df.index = pd.date_range(start, periods=len(df), freq="15min")
    df.columns = [c.split("\n")[-1].strip() for c in df.columns]
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.resample("h").sum() / 1000
    df = df[df.index.year == year]
    df.to_pickle(cache)
    return df


def _energy_charts(endpoint, name, **params):
    d = json.loads(download(ENERGY_CHARTS.format(endpoint), f"energy_charts/{name}.json", params).read_text())
    index = pd.to_datetime(d["unix_seconds"], unit="s", utc=True).tz_convert("Europe/Zurich")
    return d, index


def generation(year):
    """Swiss production per type in MW (Energy-Charts)."""
    d, index = _energy_charts("public_power", f"generation_ch_{year}", country="ch",
                              start=f"{year}-01-01", end=f"{year}-12-31")
    df = pd.DataFrame({s["name"]: s["data"] for s in d["production_types"]}, index=index)
    df = df.resample("h").mean()
    return df[df.index.year == year]


def plants():
    """Every Swiss power plant from the federal register (BFE), with type, MW and LV95 coordinates."""
    z = zipfile.ZipFile(download(BFE_PLANTS, "bfe_plants.zip"))
    p = pd.read_csv(z.open("ElectricityProductionPlant.csv"), encoding="utf-8-sig", low_memory=False)
    kind = p["SubCategory"].map({"subcat_6": "Nuclear", "subcat_2": "Solar", "subcat_3": "Wind"}).fillna("Others")
    hydro = p["SubCategory"] == "subcat_1"
    storage = p["PlantCategory"].isin(["plantcat_6", "plantcat_7"])
    kind[hydro] = np.where(storage[hydro], "Hydro storage", "Hydro run-of-river")
    return pd.DataFrame({"type": kind, "mw": p["TotalPower"] / 1000, "canton": p["Canton"],
                         "x": p["_x"], "y": p["_y"]})


def cantons(points):
    """Canton of each point {name: (lon, lat)} from the swisstopo map service."""
    path = DATA / "cantons.json"
    known = json.loads(path.read_text()) if path.exists() else {}
    for name, (lon, lat) in points.items():
        if name in known:
            continue
        r = requests.get(GEO_ADMIN, headers=HEADERS, timeout=60, params={
            "geometryType": "esriGeometryPoint", "geometry": f"{lon},{lat}", "sr": 4326, "tolerance": 0,
            "layers": "all:ch.swisstopo.swissboundaries3d-kanton-flaeche.fill",
            "returnGeometry": "false", "imageDisplay": "1,1,96", "mapExtent": "0,0,1,1"})
        found = r.json().get("results", [])
        known[name] = found[0]["attributes"]["ak"] if found else None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(known))
    return {name: known[name] for name in points}


def lv95(lon, lat):
    """WGS84 to Swiss LV95 coordinates (swisstopo approximate formulas, about 1 m)."""
    p = (np.asarray(lat) * 3600 - 169028.66) / 10000
    l = (np.asarray(lon) * 3600 - 26782.5) / 10000
    e = 2600072.37 + 211455.93 * l - 10938.51 * l * p - 0.36 * l * p**2 - 44.54 * l**3
    n = 1200147.07 + 308807.95 * p + 3745.25 * l**2 + 76.63 * p**2 - 194.56 * l**2 * p + 119.79 * p**3
    return e, n
