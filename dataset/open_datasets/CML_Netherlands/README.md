# Four-year commercial microwave link dataset for the Netherlands

Nationwide commercial microwave link data, 2011-2015.

- **Official source:** Overeem, Walraven, Leijnse and Uijlenhoet (2024), [doi:10.4121/be252844-b672-471e-8d69-27269a862ec1.v1](https://doi.org/10.4121/be252844-b672-471e-8d69-27269a862ec1.v1), 4TU.ResearchData, CC BY 4.0
- **OpenSense page:** <https://opensenseaction.eu/datasets/four-year-commercial-microwave-link-dataset-for-the-netherlands/>

T-Mobile NL links (Nokia, 1 dB resolution; NEC, 0.1 dB), 15-min minimum and maximum
received power over each interval, no transmitted power (constant within 0.2 dB, no
power control), on average ~3070 sub-links over ~1818 paths, 13 January 2011 to
15 March 2015 with gaps. Polarization is not given; the operator says most links are
vertical. The network shrinks over the period (fibre replaced links): about 4000
sub-links report per 15 minutes in 2011-2012, 3000 in 2013, 2000 in early 2014.

## In FieldSense

```bash
python -m core.opensense.fetch --dataset netherlands     # IDRawCMLdata.zip, 9.5 GB, md5-checked
```

writes `~/data/cml/netherlands/_download/IDRawCMLdata.zip`. `core/opensense/netherlands.py`
streams the RAINLINK-format text out of the zip (never unpacked; the file is sorted by
time, so a period is found without parsing what comes before it) and writes one
OpenSense-format netCDF per month to `~/data/cml/netherlands/monthly/`:

```python
from core.opensense import netherlands as nl

ds = nl.open_months("2012-06-30", "2012-07-02")    # converts the months if missing
sub = nl.usable(nl.sublinks(ds))                    # one entry per sub-link, 0 km / >= 60 GHz dropped
rain = nl.rainlink_retrieval(sub)                   # RAINLINK (Overeem et al. 2016), pycomlink's port
```

- `cml_id` is a path, named by its two endpoints (`lat0_lon0_lat1_lon1`, site 0 the
  south-western end); `sublink_0`, `sublink_1` are its RAINLINK IDs, transmitting from
  site 0 first. `rainlink_id(cml_id, sublink_id)` keeps the original ID.
- `time` is the **end** of the 15-min interval (UTC), as in the file.
- `rsl_min`, `rsl_max` in dBm; `frequency` in GHz per sub-link; `length` in km per path.

Used by [`projects/maps/netherlands`](../../../projects/maps/netherlands/) (summer 2012
against KNMI's hourly gauges) and [`examples/05_netherlands.ipynb`](../../../examples/05_netherlands.ipynb).
