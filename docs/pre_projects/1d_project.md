\documentclass[11pt,a4paper]{article}
\usepackage[margin=2.2cm]{geometry}
\usepackage{amsmath}
\usepackage{enumitem}
\setlist{itemsep=1pt,topsep=3pt}
\usepackage[hidelinks]{hyperref}
\urlstyle{same}

\begin{document}

\begin{center}
{\Large\bfseries Final Project Proposal}\\[6pt]
{\large\bfseries Discovering the Laws of Microwave-Link Attenuation}\\[2pt]
Learning path-integrated sensor models from data, with application to commercial microwave links\\[6pt]
[Student names] \quad | \quad Supervisor: [name]
\end{center}

\section{Topic and Field}
Develop a method that learns readable equations from data for a line sensor: a sensor that reports one number for a whole path, an integral of the phenomenon along the path plus terms that are not the phenomenon. Application: commercial microwave links (CMLs) for rain estimation.

\textbf{Field:} signal processing, physics-informed machine learning (symbolic regression, SINDy), opportunistic sensing, hydrometeorology.

\section{Project Content}
\textbf{Background.} Cellular networks connect towers with microwave links, each logging its transmitted and received power. Rain along the path attenuates the signal, so rain can be estimated from it \cite{messer}. CMLs are an opportunistic sensor: they were not designed to measure rain, their length and location were set for communication, power is logged with coarse quantization, and the signal also contains non-rain components \cite{chwala}.

\textbf{The problem.} After removing the baseline, the measured attenuation is
\[
A(t)=\underbrace{\int_0^L a\,R(x,t)^{b}\,dx}_{\text{rain along the path}}+\underbrace{\delta(t)}_{\text{non-rain terms}}+n(t).
\]
The standard retrieval assumes uniform rain and $\delta=0$, so $A\approx a\bar R^{\,b}L$. This hides two separate problems:
\begin{enumerate}
\item \textbf{The path law.} When rain is not uniform along the path and the law is nonlinear in $R$, the estimate is biased. The problem grows with link length \cite{berne}.
\item \textbf{Non-rain terms $\delta$}, such as wet antennas and baseline residuals. They do not grow with $L$, so they dominate on short links \cite{habi,janco,ostrometzkyWA}.
\end{enumerate}

\textbf{Goal.} Learn readable equations for both parts from data, so the sensor gives reliable measurements, and show that they improve rain estimates over the standard models. The work has three stages:
\begin{enumerate}
\item[\textbf{0.}] \textbf{Method validation:} test SINDy on physics problems with known answers (RLC circuit, heat equation).
\item[\textbf{A.}] \textbf{Path law:} replace the linear law $aR^bL$ with a general $f(R,L)$ learned by symbolic regression.
\item[\textbf{B.}] \textbf{Non-rain terms:} a per-link model of $\delta$ with memory (starting with the wet antenna), learned with SINDy, because water builds up and dries over time.
\end{enumerate}

\section{Final Deliverables}
\begin{itemize}
\item Simulator: moving 2-D rain fields, links of many lengths, a $\delta$ model, noise and quantization.
\item A learned path law $\hat f(R,L)$ and the range of lengths where the linear law holds.
\item A learned per-link model of the non-rain term, $\dot\delta=F(\delta,R)$, compared with literature models \cite{schleiss,pastorek}.
\item A rain-estimation module combining both laws, with an evaluation report against gauges and radar.
\item Open, documented code in the project repository.
\end{itemize}

\section{Implementation}
Software only, in Python, no dedicated hardware. Main libraries: PySINDy \cite{sindy}, PySR \cite{pysr}, pycomlink and poligrain for link data, xarray. Data come from open datasets (below). Every method runs first in simulation, where the truth is known, and only then on real data.

\section{Current State and Existing Implementations}
\begin{itemize}
\item \textbf{Processing:} a public OpenSense notebook runs the full chain on OpenMRG: quality control, wet/dry detection, baseline, wet-antenna correction with a fixed literature model \cite{pastorek}, and inversion of the linear law. The project builds on it and replaces its last step.
\item \textbf{Existing models of $\delta$:} constant or dynamic baseline \cite{ostrometzkyBL}, and constant, time-dependent \cite{schleiss} or rain-dependent \cite{pastorek} wet-antenna models. All fix the model form in advance.
\item \textbf{Preliminary work (Stage 0, partly done):} on a driven RLC circuit, SINDy recovered the resonance and forcing terms within 0.5\%, and the damping term, the smallest one, within 11.5\%. Savitzky-Golay filtering made it worse (23\%) and added a spurious term. For the heat equation, a diagnostics pipeline for term robustness under noise was built, including a weak-form comparison \cite{weak}. Conclusion: estimating derivatives from noisy data is the weak point, so weak-form SINDy will be used on link data.
\end{itemize}

\section{Scope of Work}
\begin{itemize}
\item \textbf{Stage 0, completion:} final heat-equation results, and RLC with weak form and a smaller library.
\item \textbf{Data:} download OpenMRG and OpenRainER, run the notebook, match radar along each link and a nearby gauge, split events into training and test.
\item \textbf{Simulator:} rain fields with controlled cell size, exact path integral per link, a known wet-antenna model, noise and quantization.
\item \textbf{Task A, path law:} compare the exact integral with the linear law in simulation, learn $\hat f(R,L)$ with PySR, and apply it to real data with radar along the path as reference.
\item \textbf{Task B, non-rain terms:} isolate $\hat\delta=A-f(R,L)$, fit literature models per link, learn $\dot\delta=F(\delta,R)$ with weak-form SINDy with rain as input, and identify wet-antenna links from their post-event drying tail.
\item \textbf{Rain estimation:} invert the combined model on held-out events.
\end{itemize}

\textbf{How success is tested:}
\begin{itemize}
\item \textbf{Simulation:} recovery of the correct terms and coefficients as a function of noise, quantization and link length.
\item \textbf{Real data:} normalized bias and normalized RMSE against gauges and radar, and false rain after events, broken down by link length. Four arms are compared: linear law with no correction, literature model, learned $\delta$, and learned $\delta$ with $\hat f$.
\item \textbf{Physical consistency:} drying time learned by SINDy vs drying time estimated directly from post-event tails. Stability of terms across links and events (bootstrap).
\end{itemize}

\section*{Datasets}
\begin{itemize}
\item \textbf{OpenMRG:} CMLs, rain gauges and radar, Gothenburg, Sweden. \url{https://zenodo.org/records/7107689}
\item \textbf{OpenRainER:} CMLs, rain gauges and radar, Emilia-Romagna, Italy. \url{https://zenodo.org/records/14731404}
\item \textbf{Example processing:} \url{https://github.com/OpenSenseAction/opensense_example_data}
\item \textbf{Project repository:} \url{https://github.com/drorjac/FieldSense}
\end{itemize}

\begin{thebibliography}{99}\small
\bibitem{messer} H. Messer, A. Zinevich, P. Alpert, ``Environmental monitoring by wireless communication networks,'' \emph{Science}, 312(5774):713, 2006.
\bibitem{chwala} C. Chwala, H. Kunstmann, ``Commercial microwave link networks for rainfall observation: Assessment of the current status and future challenges,'' \emph{WIREs Water}, 6(2):e1337, 2019.
\bibitem{berne} A. Berne, R. Uijlenhoet, ``Path-averaged rainfall estimation using microwave links: Uncertainty due to spatial rainfall variability,'' \emph{Geophys. Res. Lett.}, 34:L07403, 2007.
\bibitem{habi} H. V. Habi, H. Messer, ``Uncertainties in short commercial microwave links fading due to rain,'' \emph{ICASSP}, 2020.
\bibitem{janco} R. Janco, J. Ostrometzky, H. Messer, ``In-city rain mapping from commercial microwave links - challenges and opportunities,'' \emph{Sensors}, 23(10):4653, 2023.
\bibitem{ostrometzkyWA} J. Ostrometzky et al., ``The wet-antenna effect - a factor to be considered in future communication networks,'' \emph{IEEE Trans. Antennas Propag.}, 66(1):315-322, 2018.
\bibitem{ostrometzkyBL} J. Ostrometzky, H. Messer, ``Dynamic determination of the baseline level in microwave links for rain monitoring from minimum attenuation values,'' \emph{IEEE JSTARS}, 2018.
\bibitem{schleiss} M. Schleiss, J. Rieckermann, A. Berne, ``Quantification and modeling of wet-antenna attenuation for commercial microwave links,'' \emph{IEEE GRSL}, 10(5):1195-1199, 2013.
\bibitem{pastorek} J. Pastorek et al., ``Precipitation estimates from commercial microwave links: Practical approaches to wet-antenna correction,'' \emph{IEEE TGRS}, 2021.
\bibitem{sindy} S. L. Brunton, J. L. Proctor, J. N. Kutz, ``Discovering governing equations from data by sparse identification of nonlinear dynamical systems,'' \emph{PNAS}, 113(15):3932-3937, 2016.
\bibitem{weak} D. A. Messenger, D. M. Bortz, ``Weak SINDy: Galerkin-based data-driven model selection,'' \emph{Multiscale Model. Simul.}, 19(3):1474-1497, 2021.
\bibitem{pysr} M. Cranmer, ``Interpretable machine learning for science with PySR and SymbolicRegression.jl,'' arXiv:2305.01582, 2023.
\end{thebibliography}

\end{document}