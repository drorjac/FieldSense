# References and links

Every dataset, software package and paper the projects and tutorials rely on, in one place.
DOIs were checked against Crossref or doi.org.

## Datasets

| dataset | what | data | paper |
|---|---|---|---|
| OpenMRG | 364 CMLs, SMHI radar, city and SMHI gauges; Gothenburg, JJA 2015 | [doi:10.5281/zenodo.7107689](https://doi.org/10.5281/zenodo.7107689) | Andersson et al. (2022), *ESSD* 14, 5411-5426, [doi:10.5194/essd-14-5411-2022](https://doi.org/10.5194/essd-14-5411-2022) |
| OpenMRG2 PWS | 30 Netatmo PWS over Gothenburg, JJA 2015 (with OpenMRG) | [OpenSenseAction/OpenMRG2](https://github.com/OpenSenseAction/OpenMRG2) | |
| OpenRainER | 151 CMLs, ARPAE radar and gauges; Emilia-Romagna, 2021-2022 | [doi:10.5281/zenodo.22829808](https://doi.org/10.5281/zenodo.22829808); radar products [doi:10.5281/zenodo.10593848](https://doi.org/10.5281/zenodo.10593848) | Covi & Roversi (data description on Zenodo) |
| OpenMesh | NYC Mesh links, Weather Underground PWS; New York City, 2023-2024 | links [doi:10.5281/zenodo.15287692](https://doi.org/10.5281/zenodo.15287692), PWS [doi:10.5281/zenodo.17508286](https://doi.org/10.5281/zenodo.17508286) | Jacoby et al. (2026), *ESSD* 18, 5817, [doi:10.5194/essd-18-5817-2026](https://doi.org/10.5194/essd-18-5817-2026) |
| Amsterdam PWS | Netatmo stations, Amsterdam, 2016-2018 | [4TU](https://data.4tu.nl/articles/dataset/Rainfall_observations_datasets_from_Personal_Weather_Stations/12703250) | de Vos et al. (2019), *GRL*, [doi:10.1029/2019GL083731](https://doi.org/10.1029/2019GL083731) |
| CML Netherlands | Dutch CML data | [doi:10.4121/be252844-b672-471e-8d69-27269a862ec1.v1](https://doi.org/10.4121/be252844-b672-471e-8d69-27269a862ec1.v1) | Overeem et al. |
| OpenSense example subsets | a few days of each dataset in the OpenSense format | [OpenSenseAction/opensense_example_data](https://github.com/OpenSenseAction/opensense_example_data) | |
| NOAA MRMS | multi-radar multi-sensor QPE, CONUS | [NSSL MRMS](https://www.nssl.noaa.gov/projects/mrms/) | |
| ASOS / METAR | airport gauges, via Iowa Environmental Mesonet | [IEM ASOS download](https://mesonet.agron.iastate.edu/request/download.phtml) | |

Data format: Fencl et al. (2023), *Data formats and standards for opportunistic rainfall
sensors*, Open Research Europe 3:169,
[doi:10.12688/openreseurope.16068.1](https://doi.org/10.12688/openreseurope.16068.1); code and
examples in [OpenSenseAction/OS_data_format_conventions](https://github.com/OpenSenseAction/OS_data_format_conventions).
Where each file lives on disk: [`DATA.md`](../DATA.md); sources and licences:
[`dataset/README.md`](../dataset/README.md).

## Software

| package | used for | links |
|---|---|---|
| pycomlink | CML processing: baselines, wet-antenna models, k-R relation | [github.com/pycomlink/pycomlink](https://github.com/pycomlink/pycomlink) |
| poligrain | point, line and grid geometry; radar along links; validation metrics | [github.com/OpenSenseAction/poligrain](https://github.com/OpenSenseAction/poligrain) |
| mergeplg | interpolation and radar merging: IDW, block kriging, KED, RADOLAN | [github.com/OpenSenseAction/mergeplg](https://github.com/OpenSenseAction/mergeplg) |
| pypwsqc | quality control of personal weather stations | [github.com/OpenSenseAction/pypwsqc](https://github.com/OpenSenseAction/pypwsqc) |
| PyNNcml | neural-network CML retrieval (two-step RNN) | [github.com/haihabi/PyNNcml](https://github.com/haihabi/PyNNcml) |
| pysteps | optical flow, extrapolation and ensemble nowcasting, verification | [pysteps.github.io](https://pysteps.github.io), [github.com/pySTEPS/pysteps](https://github.com/pySTEPS/pysteps) |
| PyKrige | variogram models and kriging | [github.com/GeoStat-Framework/PyKrige](https://github.com/GeoStat-Framework/PyKrige) |
| wradlib | weather radar processing | [wradlib.org](https://wradlib.org) |

## OpenSense repositories used as templates

| repository | what FieldSense took from it |
|---|---|
| [radar_adjustment_intercomparison](https://github.com/OpenSenseAction/radar_adjustment_intercomparison) | the CML chain and 16 adjustment variants; reproduced in `projects/radar_adjustment` |
| [TrainingSchoolMergingApplication](https://github.com/OpenSenseAction/TrainingSchoolMergingApplication) | the merging and pysteps nowcasting sessions; `projects/os_nowcasting`, `tutorials/05-07` |
| [PrePEP_short_course_OS](https://github.com/OpenSenseAction/PrePEP_short_course_OS), [OPENSENSE_sandbox](https://github.com/OpenSenseAction/OPENSENSE_sandbox), [training_school_opensene_2023](https://github.com/OpenSenseAction/training_school_opensene_2023) | processing and interpolation examples; `tutorials/02`, `05` |

## Papers

**CML rainfall**

- Messer, Zinevich & Alpert (2006). Environmental monitoring by wireless communication networks. *Science* 312, 713. [doi:10.1126/science.1120034](https://doi.org/10.1126/science.1120034)
- Overeem, Leijnse & Uijlenhoet (2013). Country-wide rainfall maps from cellular communication networks. *PNAS* 110, 2741-2745. [doi:10.1073/pnas.1217961110](https://doi.org/10.1073/pnas.1217961110)
- ITU-R P.838-3 (2005). Specific attenuation model for rain for use in prediction methods. [itu.int](https://www.itu.int/rec/R-REC-P.838-3-200503-I/en)
- Schleiss & Berne (2010). Identification of dry and rainy periods using telecommunication microwave links. *IEEE GRSL* 7, 611-615. [doi:10.1109/LGRS.2010.2043052](https://doi.org/10.1109/LGRS.2010.2043052)
- Leijnse, Uijlenhoet & Stricker (2008). Microwave link rainfall estimation: effects of link length and frequency, temporal sampling, power resolution, and wet antenna attenuation. *Adv. Water Resour.* 31, 1481-1493. [doi:10.1016/j.advwatres.2008.03.004](https://doi.org/10.1016/j.advwatres.2008.03.004)
- Pastorek et al. (2022). Precipitation estimates from commercial microwave links: practical approaches to wet-antenna correction. *IEEE TGRS* 60. [doi:10.1109/TGRS.2021.3110004](https://doi.org/10.1109/TGRS.2021.3110004) (pycomlink calls the model `waa_pastorek_2021`)

**Rain maps**

- Goldshtein, Messer & Zinevich (2009). Rain rate estimation using measurements from commercial telecommunications links. *IEEE Trans. Signal Process.* 57, 1616-1625. [doi:10.1109/TSP.2009.2012554](https://doi.org/10.1109/TSP.2009.2012554)

**Rain field simulation** (`core/simulation`, `synthetic_testbed`)

- Rodriguez-Iturbe, Cox & Isham (1987). Some models for rainfall based on stochastic point processes. *Proc. R. Soc. Lond. A* 410, 269-288. [doi:10.1098/rspa.1987.0039](https://doi.org/10.1098/rspa.1987.0039)
- Cox & Isham (1988). A simple spatial-temporal model of rainfall. *Proc. R. Soc. Lond. A* 415, 317-328. [doi:10.1098/rspa.1988.0016](https://doi.org/10.1098/rspa.1988.0016)
- Schertzer & Lovejoy (1987). Physical modeling and analysis of rain and clouds by anisotropic scaling multiplicative processes. *JGR* 92, 9693-9714. [doi:10.1029/JD092iD08p09693](https://doi.org/10.1029/JD092iD08p09693)
- Over & Gupta (1996). A space-time theory of mesoscale rainfall using random cascades. *JGR* 101, 26319-26331. [doi:10.1029/96JD02033](https://doi.org/10.1029/96JD02033)
- Venugopal, Foufoula-Georgiou & Sapozhnikov (1999). Evidence of dynamic scaling in space-time rainfall. *JGR* 104, 31599-31610. [doi:10.1029/1999JD900437](https://doi.org/10.1029/1999JD900437)
- Feral, Sauvageot, Castanet & Lemorton (2003). HYCELL - A new hybrid model of the rain horizontal distribution for propagation studies: 1. *Radio Sci.* 38. [doi:10.1029/2002RS002802](https://doi.org/10.1029/2002RS002802)
- Rebora, Ferraris, von Hardenberg & Provenzale (2006). RainFARM: rainfall downscaling by a filtered autoregressive model. *J. Hydrometeor.* 7, 724-738. [doi:10.1175/JHM517.1](https://doi.org/10.1175/JHM517.1)
- Leblois & Creutin (2013). Space-time simulation of intermittent rainfall with prescribed advection field (SAMPO). *Water Resour. Res.* 49, 3375-3387. [doi:10.1002/wrcr.20190](https://doi.org/10.1002/wrcr.20190)
- Paschalis, Molnar, Fatichi & Burlando (2013). A stochastic model for high-resolution space-time precipitation simulation (STREAP). *Water Resour. Res.* 49, 8400-8417. [doi:10.1002/2013WR014437](https://doi.org/10.1002/2013WR014437)
- Kessler (1969). On the distribution and continuity of water substance in atmospheric circulations. *Meteor. Monogr.* 10(32), AMS.

**Nowcasting**

- Germann & Zawadzki (2002). Scale-dependence of the predictability of precipitation from continental radar images. Part I. *Mon. Wea. Rev.* 130, 2859-2873. [doi:10.1175/1520-0493(2002)130<2859:SDOTPO>2.0.CO;2](https://doi.org/10.1175/1520-0493(2002)130%3C2859:SDOTPO%3E2.0.CO;2)
- Seed (2003). A dynamic and spatial scaling approach to advection forecasting (S-PROG). *J. Appl. Meteor.* 42, 381-388. [doi:10.1175/1520-0450(2003)042<0381:ADASSA>2.0.CO;2](https://doi.org/10.1175/1520-0450(2003)042%3C0381:ADASSA%3E2.0.CO;2)
- Bowler, Pierce & Seed (2006). STEPS: a probabilistic precipitation forecasting scheme. *Q. J. R. Meteorol. Soc.* 132, 2127-2155. [doi:10.1256/qj.04.100](https://doi.org/10.1256/qj.04.100)
- Pulkkinen et al. (2019). Pysteps: an open-source Python library for probabilistic precipitation nowcasting (v1.0). *GMD* 12, 4185-4219. [doi:10.5194/gmd-12-4185-2019](https://doi.org/10.5194/gmd-12-4185-2019)
- Pulkkinen, Chandrasekar, von Lerber & Harri (2020). Nowcasting of convective rainfall using volumetric radar observations (ANVIL). *IEEE TGRS* 58, 7845-7859. [doi:10.1109/TGRS.2020.2984594](https://doi.org/10.1109/TGRS.2020.2984594)
- Pulkkinen, Chandrasekar & Niemi (2021). Lagrangian integro-difference equation model for precipitation nowcasting (LINDA). *J. Atmos. Oceanic Technol.* 38, 2125-2145. [doi:10.1175/JTECH-D-21-0013.1](https://doi.org/10.1175/JTECH-D-21-0013.1)
- Ayzel, Scheffer & Heistermann (2020). RainNet v1.0: a convolutional neural network for radar-based precipitation nowcasting. *GMD* 13, 2631-2644. [doi:10.5194/gmd-13-2631-2020](https://doi.org/10.5194/gmd-13-2631-2020)
