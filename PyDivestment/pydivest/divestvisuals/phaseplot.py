"""
Using streamplot to produce some phase space plots of the
ode systems for capital accumulation to put on the
documentation.
"""
import numpy as np
import matplotlib.pyplot as plt

delta = 0.06
eps = 0.25
alpha = 100.
s = 0.25

k_min = 0
k_max = 2 * pow((alpha * pow(s, 1 - eps * 0.5) / delta), 2 / (1 - eps))
c_min = 0
c_max = 2 * pow((alpha * pow(s, 0.5) / delta), 2 / (1 - eps))

c, k = np.mgrid[c_min:c_max:100j, k_min:k_max:100j]

k_dot = s * alpha * pow(k, 0.5) * pow(c, (0.5 * eps)) - delta * k
c_dot = alpha * pow(k, 0.5) * pow(c, (0.5 * eps)) - delta * c

speed = np.sqrt(k_dot * k_dot + c_dot * c_dot)

fig0, ax0 = plt.subplots(figsize=(4, 4), tight_layout=True)

lw = 5 * speed / speed.max()

strm = ax0.streamplot(k, c, k_dot, c_dot, density=0.6, linewidth=lw)
ax0.plot(k_max / 2, c_max / 2., 'xr', ms=10)

ax0.set_xlim([k_min, k_max])
ax0.set_ylim([c_min, c_max])
ax0.set_xlabel('C')
ax0.set_ylabel('K')

plt.savefig('phasespace.pdf')
