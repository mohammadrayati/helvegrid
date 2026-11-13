from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
import os

import numpy as np
import pandas as pd
import pandapower as pp
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

import data

TRAFO_VK_PERCENT = 12.0
LOAD_Q_PER_P = 0.2
COST = {"Solar": 0.0, "Wind": 0.0, "Hydro run-of-river": 1.0, "Nuclear": 5.0, "Hydro storage": 30.0, "Others": 60.0}
ENERGY_CHARTS_TYPES = {"Nuclear": ["Nuclear"], "Hydro run-of-river": ["Hydro Run-of-River"],
                       "Hydro storage": ["Hydro water reservoir", "Hydro pumped storage"],
                       "Solar": ["Solar"], "Wind": ["Wind onshore"], "Others": ["Others"]}
NEIGHBOURS = ["AT", "DE", "FR", "IT"]
SLACK_MW, SLACK_COST = 10000.0, 10.0


def build():
    osm = data.osm_grid()
    buses, lines, trafos = osm["buses"], osm["lines"], osm["transformers"]
    swiss = buses.index[buses.country == "CH"]
    lines = lines[lines.bus0.isin(swiss) | lines.bus1.isin(swiss)]
    keep = _main_island(pd.concat([lines, trafos[trafos.bus0.isin(swiss) & trafos.bus1.isin(swiss)]]))
    lines = lines[lines.bus0.isin(keep)]
    trafos = trafos[trafos.bus0.isin(keep) & trafos.bus1.isin(keep)]
    buses = buses.loc[keep]

    net = pp.create_empty_network(name="Swiss transmission grid")
    pp.create_buses(net, len(buses), vn_kv=buses.voltage.values, name=buses.index.values, zone=buses.country.values)
    net.bus["lon"], net.bus["lat"] = buses.x.values, buses.y.values
    inside = net.bus.zone == "CH"
    points = dict(zip(net.bus.name[inside], zip(net.bus.lon[inside], net.bus.lat[inside])))
    net.bus.loc[inside, "canton"] = list(data.cantons(points).values())
    bus = pd.Series(net.bus.index, index=net.bus.name)

    km = lines.length.values / 1000
    pp.create_lines_from_parameters(
        net, bus[lines.bus0].values, bus[lines.bus1].values, length_km=km, name=lines.index.values,
        r_ohm_per_km=lines.r.values / km, x_ohm_per_km=lines.x.values / km,
        c_nf_per_km=lines.b.values / (2 * np.pi * 50) * 1e9 / km,
        max_i_ka=lines.s_nom.values / (np.sqrt(3) * lines.voltage.values), max_loading_percent=100.0)
    net.line["coords"] = lines.geometry.map(_coords).values

    flip = trafos.voltage_bus0 < trafos.voltage_bus1
    hv = np.where(flip, trafos.bus1, trafos.bus0)
    lv = np.where(flip, trafos.bus0, trafos.bus1)
    pp.create_transformers_from_parameters(
        net, bus[hv].values, bus[lv].values, sn_mva=trafos.s_nom.values, name=trafos.index.values,
        vn_hv_kv=trafos[["voltage_bus0", "voltage_bus1"]].max(axis=1).values,
        vn_lv_kv=trafos[["voltage_bus0", "voltage_bus1"]].min(axis=1).values,
        vkr_percent=0.3, vk_percent=TRAFO_VK_PERCENT, pfe_kw=0.0, i0_percent=0.0, max_loading_percent=100.0)

    pp.create_loads(net, net.bus.index[inside], p_mw=0.0, controllable=False)
    _add_plants(net)
    _add_borders(net)
    return net


def _main_island(branches):
    names = pd.Index(branches.bus0).union(branches.bus1).unique()
    pos = pd.Series(np.arange(len(names)), index=names)
    a, b = pos[branches.bus0].to_numpy(), pos[branches.bus1].to_numpy()
    graph = coo_matrix((np.ones(len(a)), (a, b)), shape=(len(names), len(names)))
    island = connected_components(graph, directed=False)[1]
    return names[island == np.bincount(island).argmax()]


def _coords(wkt):
    return [tuple(map(float, point.split())) for point in wkt[wkt.index("(") + 1:-1].split(",")]


def _buses_in(net, cantons):
    inside = net.bus.index[net.bus.zone == "CH"]
    found = inside[net.bus.canton[inside].isin(cantons)]
    return found if len(found) else inside


def _add_plants(net):
    plants = data.plants()
    inside = net.bus[net.bus.zone == "CH"]
    located = plants.x.notna()
    nearest = cKDTree(np.column_stack(data.lv95(inside.lon, inside.lat))).query(plants.loc[located, ["x", "y"]])[1]
    parts = [pd.DataFrame({"bus": inside.index[nearest], "type": plants.type[located], "mw": plants.mw[located]})]
    for (canton, kind), mw in plants[~located].groupby(["canton", "type"], dropna=False).mw.sum().items():
        buses = _buses_in(net, [canton])
        parts.append(pd.DataFrame({"bus": buses, "type": kind, "mw": mw / len(buses)}))
    capacity = pd.concat(parts).groupby(["bus", "type"]).mw.sum().reset_index()
    pp.create_sgens(net, capacity.bus.values, p_mw=0.0, type=capacity.type.values, controllable=False)
    net.sgen.loc[net.sgen.index[-len(capacity):], "capacity_mw"] = capacity.mw.values


def _add_borders(net):
    s_line = np.sqrt(3) * net.line.max_i_ka * net.bus.vn_kv[net.line.from_bus].values
    ends = pd.concat([pd.Series(s_line.values, index=net.line.from_bus), pd.Series(s_line.values, index=net.line.to_bus)])
    border = net.bus.index[net.bus.zone != "CH"]
    capacity = ends.groupby(level=0).sum()[border]
    pp.create_sgens(net, border, p_mw=0.0, type="Border", controllable=False)
    net.sgen.loc[net.sgen.index[-len(border):], "capacity_mw"] = capacity.values
    pp.create_ext_grid(net, capacity.idxmax(), min_p_mw=-SLACK_MW, max_p_mw=SLACK_MW)


def set_hour(net, swissgrid, generation, hour):
    """Load per canton, production per type and exchange per neighbour of one hour."""
    s = swissgrid.loc[hour]
    load = pd.Series(0.0, index=net.load.bus)
    for column in s.index[s.index.str.startswith("Consumption Canton")]:
        buses = _buses_in(net, column.split(" ", 2)[2].split(", "))
        load[buses] += s[column] / len(buses)
    load *= s["Total energy consumption Swiss controlblock"] / load.sum()
    net.load["p_mw"] = load.values
    net.load["q_mvar"] = LOAD_Q_PER_P * load.values

    g = generation.loc[hour]
    by_type = pd.Series({kind: g[cols].sum() for kind, cols in ENERGY_CHARTS_TYPES.items()}).clip(lower=0)
    by_type *= s["Total energy production Swiss controlblock"] / by_type.sum()
    plants = net.sgen.type != "Border"
    share = net.sgen.capacity_mw / net.sgen[plants].groupby("type").capacity_mw.transform("sum")
    net.sgen.loc[plants, "p_mw"] = (share * net.sgen.type.map(by_type))[plants]

    border = net.sgen.index[~plants]
    country = net.bus.zone[net.sgen.bus[border]].values
    share = net.sgen.capacity_mw[border] / net.sgen.capacity_mw[border].groupby(country).transform("sum")
    imports = {c: s[f"Cross Border Exchange {c}->CH"] - s[f"Cross Border Exchange CH->{c}"] for c in NEIGHBOURS}
    net.sgen.loc[border, "p_mw"] = share * pd.Series(country, index=border).map(imports).fillna(0.0)
    return net


def run_power_flow(net):
    pp.runpp(net, init="dc", numba=False)
    return net


def run_dc_power_flow(net):
    pp.rundcpp(net, numba=False)
    return net


def run_opf(net):
    """DC optimal power flow: cheapest Swiss dispatch for this hour's load and exchange, within the line limits.

    Storage hydro may run up to its installed capacity, the other plants up to this hour's output.
    The exchange with each neighbour stays as measured; the slack is expensive, so it is used only when nothing else works.
    """
    net.poly_cost = net.poly_cost.iloc[0:0]
    plants = net.sgen.type != "Border"
    dispatchable = net.sgen.type == "Hydro storage"
    net.sgen["controllable"] = plants
    net.sgen.loc[plants, "min_p_mw"] = 0.0
    net.sgen.loc[plants, "max_p_mw"] = np.where(dispatchable, net.sgen.capacity_mw, net.sgen.p_mw)[plants]
    pp.create_poly_costs(net, net.sgen.index[plants], "sgen", cp1_eur_per_mw=net.sgen.type[plants].map(COST).values)
    pp.create_poly_cost(net, net.ext_grid.index[0], "ext_grid", cp1_eur_per_mw=0.0, cp2_eur_per_mw2=SLACK_COST)
    pp.rundcopp(net)
    net.sgen["p_mw"] = net.res_sgen.p_mw
    net.sgen["controllable"] = False
    return net


def summary(net):
    plants = net.sgen.type != "Border"
    production = net.res_sgen.p_mw[plants].groupby(net.sgen.type[plants]).sum()
    exchange = net.res_sgen.p_mw[~plants].groupby(net.bus.zone[net.sgen.bus[~plants]].values).sum()
    exchange["slack"] = net.res_ext_grid.p_mw.sum()
    grid = pd.Series({"load MW": net.load.p_mw.sum(), "losses MW": net.res_line.pl_mw.sum() + net.res_trafo.pl_mw.sum(),
                      "max line loading %": net.res_line.loading_percent.max(), "min voltage pu": net.res_bus.vm_pu.min()})
    return pd.concat({"production MW": production, "import MW": exchange, "grid": grid}).round(2) + 0.0


SOLVERS = {"ac": run_power_flow, "dc": run_dc_power_flow, "opf": run_opf}


def run_hours(net, swissgrid, generation, hours, method="ac", workers=None):
    """Run one solver for many hours in parallel. Returns the summary per hour and the line loading per hour.
    Hours where the solver fails are left out."""
    chunks = np.array_split(pd.DatetimeIndex(hours), 4 * (workers or os.cpu_count()))
    inputs = ([swissgrid.loc[c] for c in chunks], [generation.loc[c] for c in chunks])
    with ProcessPoolExecutor(workers) as pool:
        parts = list(pool.map(_run_chunk, repeat(net), *inputs, repeat(method)))
    return pd.concat([p[0] for p in parts]), pd.concat([p[1] for p in parts])


def _run_chunk(net, swissgrid, generation, method):
    rows, loading = {}, {}
    for hour in swissgrid.index:
        set_hour(net, swissgrid, generation, hour)
        try:
            SOLVERS[method](net)
        except (pp.LoadflowNotConverged, pp.OPFNotConverged):
            continue
        rows[hour] = summary(net)
        loading[hour] = net.res_line.loading_percent.set_axis(net.line.name)
    return pd.DataFrame(rows).T, pd.DataFrame(loading).T
