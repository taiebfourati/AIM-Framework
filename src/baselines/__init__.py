"""
baselines/ — external baseline detectors used for the CRIT-7 comparison
in Section~\\ref{sec:eval:baselines} of the thesis.

The submodules here are intentionally self-contained: each implements
the standard published algorithm without reaching into the production
RTP/AIF/ATM code paths.  This keeps the comparison fair (every baseline
sees exactly the same observation stream the production detectors see)
and easy to audit (the algorithm is in one file with one citation in
the docstring).
"""
