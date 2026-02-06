from scipy.spatial.distance import jensenshannon

print(jensenshannon([1.0, 0.0], [0.0, 1.0]))
print(jensenshannon([1.0, 0.0], [0.0, 1.0], base=2))
