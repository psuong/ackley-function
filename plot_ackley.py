from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

fig = plt.figure()
ax = fig.gca(projection='3d')

# Make data.
X = np.arange(-32, 32, 0.25)
Y = np.arange(-32, 32, 0.25)
X, Y = np.meshgrid(X, Y)

a = 20
b = 0.2
c = 2 * np.pi

sum_sq_term = -a * np.exp(-b * np.sqrt(X*X + Y*Y) / 2)
cos_term = -np.exp((np.cos(c*X) + np.cos(c*Y)) / 2)
Z = a + np.exp(1) + sum_sq_term + cos_term

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()

