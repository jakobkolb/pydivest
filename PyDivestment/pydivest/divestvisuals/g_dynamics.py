"""
This little script is meant to evaluate the resource depletion
dynamic and the actual relevance of the resource depletion timescale.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

t_G = 100
s = 0.23
e = 100.
d_c = 0.06
b_d = 1.4
P = 1000
alphas = [.5, .25, 0.1]

colors = 'rgbk'
lines = []

fig = plt.figure(figsize=(4, 4), tight_layout=True)
ax = fig.add_subplot(111)

for i, alpha in enumerate(alphas):
    c = colors[i]
    b_R = alpha ** 2 * e

    G_0 = t_G * (P * s * b_d ** 2) / (e * d_c)
    G_star = pow(b_R / e, 1. / 2.) * G_0

    t = np.linspace(0, 2 * t_G, 100)


    def dot_G(G, t):
        print G
        return -s / (e * d_c) * P * (1 - b_R / e * (G_0 / G) ** 2) * b_d ** 2


    G = odeint(dot_G, [G_0], t)

    lines.append(ax.plot(t, G, c, label=r'$\alpha= {:3.2f}$'.format(alpha)))
ax.plot((t_G, t_G), (0., G_0), '--k', label=r'$t^*_G$')

ax.set_ylim([0, 1.1 * G_0])
ax.set_ylabel(r'$G(t)$')
ax.set_xlabel(r'$t$')
plt.legend(loc=0)

plt.savefig('g_depletion.pdf')
