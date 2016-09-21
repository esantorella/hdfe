A Python package for running regressions with high-dimensional fixed effects or an arbitrary sprase design matrix.

The 'alternating projections' option has a C++ backend and similar functionality to Simen Gaure's lfe package in R. It currently doesn't recover fixed effects and may not be faster than the 'brute force' option.

This is very much a work in progress. The aim is for speed, not comprehensiveness, so this package will probably never compute standard errors or other standard regression outputs.
