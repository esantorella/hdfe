import cppimport
import numpy as np

cppplay = cppimport.imp("cppplay")
input = ["fish", "spider", "fish", 3, 4, []]
_, input =  np.unique(input, return_inverse = True)
print(input)
g = cppplay.Groupby(input.tolist())
vals = np.linspace(0,1,6).tolist()
print(type(vals[0]))
print(type(np.mean))
print(g.apply(vals, lambda x: np.mean(x).tolist()))
