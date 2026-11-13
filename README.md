# helvegrid

A model of the Swiss 220/380 kV transmission grid built only from public data, with an AC power flow and a DC optimal power flow (OPF) for one hour or every hour of several years.

## Data

Everything is downloaded on the first run and kept in `data/`.

| What | Source |
|---|---|
| Lines, transformers, substations | OpenStreetMap, prepared by PyPSA-Eur ([Zenodo record 23896564](https://zenodo.org/records/23896564)) |
| Power plants with location, type and MW | Federal register of power plants, BFE ([download](https://data.geo.admin.ch/ch.bfe.elektrizitaetsproduktionsanlagen/elektrizitaetsproduktionsanlagen/elektrizitaetsproduktionsanlagen_2056.csv.zip)) |
| Hourly load per canton, total production, cross-border exchange | Swissgrid energy overview ([swissgrid.ch](https://www.swissgrid.ch/en/home/operation/grid-data/generation.html)) |
| Hourly production per type | Energy-Charts, Fraunhofer ISE ([api.energy-charts.info](https://api.energy-charts.info)) |
| Canton of each substation | swisstopo map service ([api3.geo.admin.ch](https://api3.geo.admin.ch)) |

## Run

```bash
cd helvegrid
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/jupyter lab run.ipynb
```

The first run downloads about 60 MB and runs two years hour by hour, which takes about 15 minutes. The results are saved in `data/`, so after that the notebook runs in under a minute.

To use it from Python instead:

```python
import data, network

net = network.build()
network.set_hour(net, data.swissgrid(2025), data.generation(2025), "2025-01-15 18:00")
network.run_power_flow(net)    # AC power flow, results in net.res_line, net.res_bus
network.run_opf(net)           # DC OPF, sets the plant outputs to the optimal dispatch
network.summary(net)
```

For many hours (in parallel on all CPU cores):

```python
swissgrid, generation = data.swissgrid(2025), data.generation(2025)
hours = swissgrid.index.intersection(generation.index)
per_hour, line_loading = network.run_hours(net, swissgrid, generation, hours, method="ac")   # or "dc", "opf"
```

`per_hour` has one row per hour (production per type, import per country, load, losses, highest line loading). `line_loading` has the loading of every line in every hour. Hours where the solver fails are left out.

## How the model works

- **Grid.** The model keeps every OSM line with at least one end in Switzerland. The foreign end of a cross-border line becomes a border substation. Line impedance and rating come from PyPSA-Eur.
- **Load.** Swissgrid gives the load of each canton (or group of cantons). It is split equally over the substations in that canton, then scaled to the total consumption of the Swiss control block.
- **Production.** Each plant from the BFE register is connected to its nearest substation. Plants without coordinates are spread over the substations of their canton. The Energy-Charts output per type is scaled to the Swissgrid total and split over the plants of that type by installed capacity.
- **Borders.** The Swissgrid exchange with each neighbour (AT, DE, FR, IT) is split over that country's border substations by line capacity.
- **Power flow.** Newton-Raphson (pandapower). The slack is the border substation with the most line capacity. It covers the losses and the small gap in the data.
- **OPF.** DC OPF (pandapower, PYPOWER solver). Load and border exchange stay fixed, and the OPF picks the cheapest Swiss dispatch with every line and transformer at or below its limit.
  - Storage hydro can produce up to its installed capacity.
  - The other plants can produce up to their output in that hour.
  - If no dispatch meets the limits, pandapower raises `OPFNotConverged`.

## Assumptions

These values are not public. They are set at the top of `network.py`.

| Name | Value | Meaning |
|---|---|---|
| `TRAFO_VK_PERCENT` | 12 | Transformer short-circuit voltage |
| `LOAD_Q_PER_P` | 0.2 | Reactive load per MW |
| `COST` | 0 to 60 EUR/MWh | Production cost per type for the OPF |

Further simplifications:
- Substations are single busbars.
- The lower-voltage grids (below 220 kV) are not modelled.

## Files

| File | Content |
|---|---|
| `data.py` | Downloads and reads the public data |
| `network.py` | Builds the pandapower grid, sets one hour, runs the power flow and the OPF, for one hour or many |
| `run.ipynb` | Step-by-step example with maps and tables |
