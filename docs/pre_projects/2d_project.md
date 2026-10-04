% ============================================================
%  Project Proposal - B.Sc. Final Project
%  2D Rainfall Mapping from CML Attenuation via Machine Learning
%  CellEnMon Lab, School of Electrical Engineering, Tel Aviv University
% ============================================================
\documentclass[11pt,a4paper]{article}

\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
%\usepackage{lmodern}
\usepackage[margin=2.4cm]{geometry}
\usepackage{amsmath,amssymb,amsthm}
\usepackage{booktabs}
\usepackage{array}
\usepackage{tabularx}
\usepackage{enumitem}
\usepackage{graphicx}
%\usepackage{microtype}
\usepackage[table]{xcolor}
\usepackage{titlesec}
\usepackage[colorlinks=true,linkcolor=blue!55!black,citecolor=blue!55!black,urlcolor=blue!55!black]{hyperref}

\newcolumntype{L}[1]{>{\raggedright\arraybackslash}p{#1}}
\newcolumntype{C}[1]{>{\centering\arraybackslash}p{#1}}

\titleformat{\section}{\normalfont\large\bfseries}{\thesection.}{0.6em}{}
\titleformat{\subsection}{\normalfont\normalsize\bfseries}{\thesubsection}{0.6em}{}
\setlist[itemize]{topsep=2pt,itemsep=1pt,leftmargin=1.3em}
\setlist[enumerate]{topsep=2pt,itemsep=1pt,leftmargin=1.6em}

\newcommand{\RR}{\mathbb{R}}

% ------------------------------------------------------------
\begin{document}

\begin{center}
{\LARGE\bfseries Data-Driven 2-D Rainfall Mapping from\\[2pt]
Commercial Microwave Link Attenuation}\\[10pt]
{\large Project Proposal - B.Sc. Final Project}\\[8pt]
{\normalsize CellEnMon Lab (The Cellular Environmental Monitoring Lab)\\
School of Electrical Engineering, Tel Aviv University}\\[8pt]
{\small Students: \rule{4cm}{0.4pt} \quad Supervisor: \rule{4cm}{0.4pt} \quad Date: \rule{2.5cm}{0.4pt}}
\end{center}

\vspace{4pt}
\hrule
\vspace{10pt}

% ============================================================
\section{Subject and Field of the Project}

The project addresses \textbf{two-dimensional rainfall mapping} from attenuation measurements
collected over \textbf{Commercial Microwave Links (CMLs)}, the point-to-point microwave backhaul
links of cellular networks, with optional fusion of complementary environmental sensors
(weather radar, rain gauges, satellite products). The work is carried out at the CellEnMon Lab
of the School of Electrical Engineering and sits at the intersection of three fields:

\begin{center}
\begin{tabular}{@{}lL{11.2cm}@{}}
\toprule
\textbf{Field} & \textbf{Role in the project} \\
\midrule
Signal processing & Extraction of rain-induced attenuation from raw RSL/TSL series, baseline (dry) removal, noise and wet-antenna handling. \\
Opportunistic sensing & Reuse of an existing telecom infrastructure as a dense, wide-coverage rain sensor network with no dedicated deployment. \\
Machine / deep learning & Learned inversion from a set of path-integrated line measurements to a gridded rain-intensity field. \\
\bottomrule
\end{tabular}
\end{center}

% ============================================================
\section{Technical Background}

CMLs are infrastructure links between cellular base stations, transmitting in the microwave band
(typically $5$-$42$\,GHz, with sub-bands in use as well). Rainfall along the link path induces an
attenuation that is measurable as the difference between the transmitted signal level (TSL) and the
received signal level (RSL). Because cellular networks are deployed densely in populated areas,
thousands of existing links can be exploited as an opportunistic rain-sensor network.

For a link $i$ with path $L_i$, the attenuation measurement $Y_i$ is related to the rain-rate field
$X(\cdot)$ by the empirical power law
\begin{equation}
Y_i \;=\; a_i \int_{s \in L_i} X(s)^{\,b_i}\, \mathrm{d}L_i \;+\; \sigma_i Z_i,
\qquad Z_i \sim \mathcal{N}(0,1),
\label{eq:powerlaw}
\end{equation}
where $X(s)$ is the rain rate in mm/h at location $s$, the coefficients $(a_i,b_i)$ depend on the
link frequency, polarization and path length, and $\sigma_i Z_i$ is additive measurement noise
\cite{leijnse2007,eshel2021,moufad2026}.

\subsection*{Problem statement}

Given attenuation measurements $\{Y_i\}_{i=1}^{m}$ from a CML network at time $t$ over a region of
interest $\Omega \subset \RR^2$, together with the network metadata
$\mathcal{M} = \{(\mathbf{p}_i^{\mathrm{tx}}, \mathbf{p}_i^{\mathrm{rx}}, f_i, \text{pol}_i, \ell_i)\}_{i=1}^m$,
estimate the discretized rain field
\begin{equation}
\widehat{R}(x,y,t) \in \RR^{H \times W}, \qquad (x,y) \in \Omega ,
\end{equation}
using the CML measurements as the primary observation source.

The problem is ill-posed for two structural reasons. First, every measurement is a
\emph{path integral}, so the map from field to observation is a lossy projection and $m \ll HW$.
Second, the forward operator in \eqref{eq:powerlaw} is \emph{nonlinear} in $X$ through the exponent
$b_i$. Rain fields are additionally sparse, non-stationary and heavy-tailed, which penalizes
smoothing-based estimators.

% ============================================================
\section{Prior Art: Model-Driven Approaches}

All methods evaluated in the lab so far are model-driven. They are summarized below and serve as
the baselines against which this project is measured.

\begin{center}
\begin{tabular}{@{}L{2.5cm}L{7.6cm}L{4.4cm}@{}}
\toprule
\textbf{Approach} & \textbf{Principle} & \textbf{Main limitation} \\
\midrule
Naive interpolation (IDW, OK) &
Each link is collapsed to a single noisy Virtual Rain Gauge (VRG) at the path midpoint, followed by
spatial interpolation (inverse distance weighting, ordinary Kriging) \cite{eshel2019} &
Discards along-path variability. Degrades under heterogeneous rain. \\
\addlinespace
Tomographic inversion &
The 2-D field is reconstructed as a tomographic problem (CT analogy), exploiting the spatial
intersection of multiple link paths on a grid \cite{giuli1991} &
Requires dense path crossing. Highly sensitive to noise and to regularization choice. \\
\addlinespace
Hybrid (GMZ) &
$K$ VRGs are distributed along each link and constrained to reproduce the measured path integral,
then redistributed iteratively using neighboring links and interpolated \cite{goldshtein2009} &
Still an explicit physical inversion with hand-set priors. \\
\bottomrule
\end{tabular}
\end{center}

A recent line of work replaces the hand-set prior with a learned generative prior and keeps the
exact nonlinear path-integral operator of \eqref{eq:powerlaw}, casting reconstruction as Bayesian
posterior sampling under a diffusion-model prior \cite{moufad2026}. On the OpenMRG benchmark that
family outperforms IDW, OK and GMZ on RMSE, Pearson correlation and cumulative-rainfall error. It
defines the current performance target for this project.

% ============================================================
\section{This Project: A Data-Driven Formulation}

Instead of treating each link as an independent virtual gauge or inverting the physics explicitly,
we train a data-driven estimator
\begin{equation}
g_\theta : \big(\{Y_i\}_{i=1}^m, \mathcal{M}\big) \;\longmapsto\; \widehat{R} \in \RR^{H \times W},
\end{equation}
supervised by radar-derived reference fields available in the lab database. The learned mapping
absorbs the network geometry, the link-dependent power-law coefficients and the spatial statistics
of rainfall into one estimator, rather than imposing them as separate modeling stages.

\subsection*{Candidate model families}

\begin{center}
\begin{tabular}{@{}L{3.3cm}L{7.0cm}L{4.2cm}@{}}
\toprule
\textbf{Family} & \textbf{Representation} & \textbf{Why it is a candidate} \\
\midrule
Kriging with External Drift (KED) &
Geostatistical interpolation with a learned or radar-informed drift term &
Strong, cheap reference point. Interpretable. \\
\addlinespace
Graph neural network (GNN) &
Links as graph nodes or edges, metadata as node features, grid cells as query points &
Native handling of an irregular, time-varying set of links. \\
\addlinespace
CNN / U-Net on a grid &
Links rasterized onto the target grid as sparse input channels &
Mature architectures. Direct image-to-image supervision against radar. \\
\addlinespace
Diffusion / generative prior &
Learned prior over rain fields, conditioned on CML observations &
Preserves rainfall statistics and yields an ensemble rather than one deterministic map \cite{moufad2026}. \\
\bottomrule
\end{tabular}
\end{center}

\noindent
The first two or three families are to be implemented and compared. The generative branch is a
stretch goal, contingent on time and GPU availability.

% ============================================================
\section{Work Packages}

\begin{center}
\begin{tabularx}{\textwidth}{@{}C{0.9cm}L{3.5cm}X@{}}
\toprule
\textbf{WP} & \textbf{Title} & \textbf{Content} \\
\midrule
1 & Theoretical background &
Attenuation-based rainfall estimation from CMLs. Literature review on 2-D rainfall mapping from CML
networks, covering the model-driven baselines and the learned-prior line of work. \\
\addlinespace
2 & Data pipeline &
Retrieval, cleaning and organization of RSL/TSL series and network metadata from the existing lab
database. Missing-value and noise handling. Attenuation computation as $\mathrm{TSL}-\mathrm{RSL}$
with dry-baseline subtraction. Temporal alignment of CML, radar and gauge records. \\
\addlinespace
3 & Methodology definition &
Choice of geographic grid and projection, grid resolution and temporal aggregation window.
Selection of the data-driven mapping method and definition of the train / validation / test split
by storm event rather than by random sampling. \\
\addlinespace
4 & Implementation and training &
Python implementation of the selected models. Supervised training against radar-derived fields.
Hyperparameter tuning. Baseline reimplementation (IDW, OK, GMZ) for a like-for-like comparison. \\
\addlinespace
5 & Validation &
Quantitative comparison against held-out radar fields and independent rain gauges. Error metrics
per Table~\ref{tab:metrics}. Breakdown by rain intensity, link density and storm type. \\
\addlinespace
6 & Visualization &
Tool or dashboard presenting the reconstructed maps, including a time sequence (animation) over a
storm event. \\
\addlinespace
7 & Documentation &
Final report: method, results, performance analysis, limitations and directions for further work
such as multi-sensor fusion. \\
\bottomrule
\end{tabularx}
\end{center}

% ============================================================
\section{Evaluation Protocol}

Reference fields are radar-derived products from the lab database, with independent rain gauges used
as a secondary, fully held-out reference. Evaluation is performed on storm events that are absent
from training.

\begin{table}[h]
\centering
\begin{tabular}{@{}L{4.0cm}L{6.2cm}L{4.3cm}@{}}
\toprule
\textbf{Metric} & \textbf{Definition} & \textbf{What it captures} \\
\midrule
RMSE & Pixel-wise root mean square error against the reference field & Overall magnitude error \\
MAE & Pixel-wise mean absolute error & Error robust to outliers \\
Pearson correlation (PCC) & Linear agreement, per time step and spatially pooled & Spatial pattern fidelity \\
Cumulative rainfall error & Difference in total precipitation over an event & Hydrological bias \\
Detection scores (POD, FAR, CSI) & Wet / dry classification at a rain-rate threshold & Rain-area delineation \\
Gauge-point error & Error at independent gauge locations & Unbiased external check \\
\bottomrule
\end{tabular}
\caption{Validation metrics. All metrics are reported for the proposed models and for the
IDW, OK and GMZ baselines on identical splits.}
\label{tab:metrics}
\end{table}

% ============================================================
\section{Deliverables}

\begin{center}
\begin{tabularx}{\textwidth}{@{}C{0.9cm}L{3.9cm}X@{}}
\toprule
\textbf{\#} & \textbf{Deliverable} & \textbf{Acceptance criterion} \\
\midrule
D1 & Data pipeline &
Reproducible extraction and preprocessing of raw RSL/TSL and network metadata from the lab
database into model-ready tensors, under version control. \\
\addlinespace
D2 & Data-driven mapping model &
Trained model (neural network, ML-based spatial interpolation, or a combination) producing
$\widehat{R}(x,y,t)$ from CML attenuation, with training and inference scripts. \\
\addlinespace
D3 & Visualization tool &
Interactive or scripted display of the reconstructed maps, including a time-series animation over
a storm event. \\
\addlinespace
D4 & Benchmark results &
Metrics of Table~\ref{tab:metrics} for the proposed models and for the three model-driven baselines
on the same held-out events. \\
\addlinespace
D5 & Final report &
Method description, results, validation against independent radar and gauge data, limitations and
proposals for further research. \\
\bottomrule
\end{tabularx}
\end{center}

% ============================================================
\section{Implementation Means}

\begin{center}
\begin{tabular}{@{}L{3.6cm}L{11.4cm}@{}}
\toprule
\textbf{Category} & \textbf{Tools} \\
\midrule
Language & Python \\
Data processing & numpy, pandas, xarray \\
ML / DL & PyTorch or TensorFlow, scikit-learn, PyTorch Geometric (for the GNN branch) \\
Geostatistics & pykrige or gstools (Kriging, KED) \\
Geographic visualization & matplotlib, GeoPandas, folium \\
Version control & Git \\
Data source & Existing lab database: historical RSL/TSL from the Israeli CML network, with
radar and rain-gauge records for labeling \\
Compute & Personal laptops for routine work. Lab GPU for deep-model training. No additional
equipment procurement is required. \\
\bottomrule
\end{tabular}
\end{center}

% ============================================================
\section{Current State and Existing Basis}

The CellEnMon Lab maintains ongoing research on CML-based opportunistic environmental sensing. The
following assets already exist and are the starting point of this project:

\begin{center}
\begin{tabular}{@{}L{5.0cm}L{10.0cm}@{}}
\toprule
\textbf{Asset} & \textbf{Status} \\
\midrule
Acquisition infrastructure & Operational collection and first-stage processing of RSL/TSL data \\
Database & Historical measurements from an Israeli CML network, with radar-based labels \\
Prior methodology & Published work on data-driven 2-D rainfall mapping from CMLs, plus the
model-driven baselines of Section~3 \\
\bottomrule
\end{tabular}
\end{center}

\noindent
The present project is a continuation step. The students build on the existing infrastructure,
data and algorithms, and their contribution is the attenuation-to-2-D-map conversion stage.

% ============================================================
\section{Risks and Mitigations}

\begin{center}
\begin{tabular}{@{}L{5.4cm}L{9.6cm}@{}}
\toprule
\textbf{Risk} & \textbf{Mitigation} \\
\midrule
Radar labels are themselves biased and need calibration &
Treat radar as a noisy reference. Report gauge-point error as an independent check. \\
\addlinespace
Sparse or clustered link geometry leaves parts of the grid unobserved &
Report metrics as a function of local link density. Mask unobservable cells rather than
extrapolating into them. \\
\addlinespace
Wet-antenna attenuation and baseline drift bias the extracted attenuation &
Apply an explicit wet-antenna correction in WP2 and quantify its effect by ablation. \\
\addlinespace
Limited number of labeled storm events relative to model capacity &
Start with the low-capacity branches (KED, small CNN). Use event-level cross-validation and
data augmentation before scaling model size. \\
\addlinespace
GPU contention or training cost &
Keep a CPU-trainable baseline in the comparison at all times. \\
\bottomrule
\end{tabular}
\end{center}

% ============================================================
\begin{thebibliography}{9}
\small

\bibitem{eshel2019}
A. Eshel, J. Ostrometzky, S. Gat, P. Alpert, and H. Messer,
``Spatial reconstruction of rain fields from wireless telecommunication networks - scenario-dependent
analysis of IDW-based algorithms,''
\emph{IEEE Geoscience and Remote Sensing Letters}, vol. 17, no. 5, pp. 770-774, 2019.

\bibitem{giuli1991}
D. Giuli, A. Toccafondi, G. B. Gentili, and A. Freni,
``Tomographic reconstruction of rainfall fields through microwave attenuation measurements,''
\emph{Journal of Applied Meteorology}, vol. 30, no. 9, pp. 1323-1340, 1991.

\bibitem{goldshtein2009}
O. Goldshtein, H. Messer, and A. Zinevich,
``Rain rate estimation using measurements from commercial telecommunications links,''
\emph{IEEE Transactions on Signal Processing}, vol. 57, no. 4, pp. 1616-1625, 2009.

\bibitem{moufad2026}
B. Moufad, A. Ilina, H. V. Habi, S. Lahlou, Y. Janati, H. Messer, and E. Moulines,
``Bayesian rain field reconstruction using commercial microwave links and diffusion model priors,''
in \emph{Proceedings of the 43rd International Conference on Machine Learning (ICML)}, PMLR 306, 2026.

\bibitem{leijnse2007}
H. Leijnse, R. Uijlenhoet, and J. N. M. Stricker,
``Rainfall measurement using radio links from cellular communication networks,''
\emph{Water Resources Research}, vol. 43, no. 3, 2007.

\bibitem{eshel2021}
A. Eshel, H. Messer, H. Kunstmann, P. Alpert, and C. Chwala,
``Quantitative analysis of the performance of spatial interpolation methods for rainfall estimation
using commercial microwave links,''
\emph{Journal of Hydrometeorology}, vol. 22, no. 4, pp. 831-843, 2021.

\bibitem{andersson2022}
J. C. M. Andersson, J. Olsson, R. van de Beek, and J. Hansryd,
``OpenMRG: Open data from Microwave links, Radar, and Gauges for rainfall quantification in
Gothenburg, Sweden,''
\emph{Earth System Science Data}, vol. 14, pp. 5411-5426, 2022.

\end{thebibliography}

\end{document}