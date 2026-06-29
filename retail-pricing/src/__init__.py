"""Retail pricing optimization: elasticity estimation -> margin-maximizing prices.

Pipeline:
    data.load_prepared()  -> clean panel with log columns
    elasticity.*          -> naive OLS / product FE / IV-2SLS estimates of epsilon
    optimize.*            -> margin-maximizing price given epsilon and unit cost
"""
