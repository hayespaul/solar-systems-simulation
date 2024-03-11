# cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
import numpy as np
cimport numpy as np
from libc.math cimport sqrt
import matplotlib.pyplot as plt
import time

def main(int part_num):
    cdef:
        int dim = 3
        double timestep = 600.0  # 10 minutes in seconds
        int steps = 2000
        double G = 6.67408e-11  # Gravitational constant
        double ast_mass = 2e15  # Mass of asteroids
        double[:,:,:] r = np.zeros((steps + 1, part_num, dim), dtype=np.double)
        double[:,:,:] v = np.zeros((steps + 1, part_num, dim), dtype=np.double)
        double[:,:] r_half = np.zeros((part_num, dim), dtype=np.double)
        double[:,:,:] a_half = np.zeros((steps, part_num, dim), dtype=np.double)
        double[:] mass = np.zeros(part_num, dtype=np.double)
        double x_tot_mom = 0
        double y_tot_mom = 0
        double z_tot_mom = 0

        int i, j, k, t
        double x, y, z, r_dist

    # Masses of celestial bodies and asteroids
    mass[:] = [1.989e30, 3.302e23, 4.868e24, 5.972e24, 6.417e23, 1.898e27,
               5.685e26, 8.682e25, 1.024e26] + [ast_mass] * (part_num - 9)

    # Initial positions and velocities
    init_positions_and_velocities(r, v, mass, G, part_num)

    # Simulation loop
    start = time.time()
    for t in range(steps):
        update_positions_and_velocities(r, v, a_half, r_half, mass, G, timestep, t, part_num, dim)
    final = time.time()

    print(final - start)
    plt.plot(r[:, :, 0], r[:, :, 1])
    plt.show()

cdef void init_positions_and_velocities(double[:,:,:] r, double[:,:,:] v, double[:] mass, double G, int part_num):
    # Distances for planets in AU and velocities in m/s calculated from circular orbits
    cdef double[:] distances = [0, 0.387, 0.723, 1.0, 1.524, 5.203, 9.537, 19.191, 30.069]
    cdef double AU = 1.496e11  # Astronomical unit in meters
    cdef int i

    for i in range(part_num):
        if i < len(distances):
            r[0, i, 0] = distances[i] * AU
            if i > 0:  # Skipping the Sun
                v[0, i, 1] = sqrt(G * mass[0] / r[0, i, 0])  # Circular orbit velocity
        else:  # Asteroid belt and beyond
            r[0, i, 0] = r[0, i-1, 0] + 3.5e10  # Arbitrary spacing
            v[0, i, 1] = sqrt(G * mass[0] / r[0, i, 0])  # Circular orbit velocity

    # Adjusting the Sun's velocity for momentum conservation
    for i in range(1, part_num):
        x_tot_mom += mass[i] * v[0, i, 0]
        y_tot_mom += mass[i] * v[0, i, 1]
        z_tot_mom += mass[i] * v[0, i, 2]

    v[0, 0, 0] = -x_tot_mom / mass[0]
    v[0, 0, 1] = -y_tot_mom / mass[0]
    v[0, 0, 2] = -z_tot_mom / mass[0]

cdef void update_positions_and_velocities(double[:,:,:] r, double[:,:,:] v, double[:,:,:] a_half, double[:,:] r_half, double[:] mass, double G, double timestep, int t, int part_num, int dim):
    cdef int i, j, k
    cdef double x, y, z, r_dist

    for i in range(part_num):
        for k in range(dim):
            r_half[i, k] = r[t, i, k] + v[t, i, k] * (timestep / 2)

    for i in range(part_num):
        for j in range(part_num):
            if i != j:
                x = r_half[j, 0] - r_half[i, 0]
                y = r_half[j, 1] - r_half[i, 1]
                z = r_half[j, 2] - r_half[i, 2]
                r_dist = sqrt(x * x + y * y + z * z)

                # Acceleration calculation
                a_half[t, i, 0] += G * mass[j] / (r_dist * r_dist * r_dist) * x
                a_half[t, i, 1] += G * mass[j] / (r_dist * r_dist * r_dist) * y
                a_half[t, i, 2] += G * mass[j] / (r_dist * r_dist * r_dist) * z

        # Update velocities and positions
        for k in range(dim):
            v[t + 1, i, k] = v[t, i, k] + a_half[t, i, k] * timestep
            r[t + 1, i, k] = r[t, i, k] + (v[t, i, k] + v[t + 1, i, k]) * (timestep / 2)
