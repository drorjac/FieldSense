"""Helpers for the path-law starter notebooks: inputs, targets and scores, no method.

``simulate``  links of many lengths and frequencies over simulated rain; the exact
              path integral against the linear law ``a R^b L``, as a tidy table
``events``    storm events from the radar, a train/test split by event, time masks
``scoring``   normalized bias and NRMSE against radar and gauges, by link length
``tails``     post-event drying tails, and the literature wet-antenna models fitted
              per link (the baseline a learned ``d delta / dt = F(delta, R)`` must beat)
"""
