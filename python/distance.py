# import scipy
# from scipy.stats import wasserstein_distance
# #print(wasserstein_distance([0, 2, 0,1], [0, 1, 2,0]))
# print(scipy.stats.entropy([0.2, 0.1, 0.3,0.4],  [0.1, 0.1, 0.4,0.4]))

import numpy as np
from scipy import stats as sts
x = np.asarray([0, 2, 6])
y = np.asarray([0, 2, 6])
def JS_divergence(p, q):
    M = (p + q) / 2
    return 0.5 * sts.entropy(p, M) + 0.5 * sts.entropy(q, M)

dis = JS_divergence(x, y)
print(dis)