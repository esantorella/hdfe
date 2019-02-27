This package contains functionality useful for econometrics.
Its name, originally standing for "high-dimensional fixed effects," is now misleading.

Useful features are
* groupby.Groupby: A class allowing for fast operations similar to Pandas groupby-apply and groupby-transform 
functionality, but performing significantly faster with user-written functions. See 
documentation [here](http://esantorella.com/2016/06/16/groupby/).
* multicollinearity.find_collinear_cols and multicollinearity.remove_collinear_cols: Functions
for dealing with multicollinearity which operate quickly on CSC matrices.

