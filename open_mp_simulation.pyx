# cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
import numpy as np
from libc.math cimport sqrt
from cython.parallel cimport prange
cimport openmp

def main(int threads):
    cdef int dim = 3
    cdef int part_num = 1000
    cdef double timestep = 60 * 60 * 24 * 10  # 10 days in seconds
    cdef int steps = 10
    cdef double G = 6.67408e-11  # Gravitational constant

    # Initialize arrays
    cdef double[:,:,:] r = np.zeros((steps + 1, part_num, dim), dtype=np.double)
    cdef double[:,:,:] v = np.zeros((steps + 1, part_num, dim), dtype=np.double)
    cdef double[:] mass = np.zeros(part_num, dtype=np.double)

    # Set initial conditions
    initialize_conditions(r, v, mass, part_num, G)

    # Simulation loop
    simulate(r, v, mass, part_num, steps, timestep, G, threads)

def initialize_conditions(double[:,:,:] r, double[:,:,:] v, double[:] mass, int part_num, double G):
    cdef double ast_mass = 2e15  # Mass of an asteroid

    # Masses of the sun and planets
    mass[:10] = [1.989e30, 3.302e23, 4.868e24, 5.972e24, 6.417e23, 1.898e27, 5.685e26, 8.682e25, 1.024e26]
    mass[10:] = ast_mass  # Assign asteroid mass to the rest

    # Initial positions and velocities for the sun and planets
    # Positions are in meters and calculated from their average distance from the sun
    # Velocities are calculated assuming circular orbits
    cdef double[::1] distances = [0, 5.79e10, 1.08e11, 1.49e11, 2.27e11, 7.78e11, 1.43e12, 2.87e12, 4.50e12]  # in meters
    cdef int i
    for i in range(len(distances)):
        r[0, i, 0] = distances[i]
        if i > 0:  # Skip the sun for velocity calculation
            v[0, i, 1] = sqrt(G * mass[0] / distances[i])  # Circular orbit velocity

    # Set initial conditions for asteroids in the asteroid belt
    for i in range(10, part_num):
        r[0, i, 0] = r[0, i - 1, 0] + 3.5e10  # Arbitrary spacing for simplicity
        v[0, i, 1] = sqrt(G * mass[0] / r[0, i, 0])  # Circular orbit velocity

def simulate(double[:,:,:] r, double[:,:,:] v, double[:] mass, int part_num, int steps, double timestep, double G, int threads):
    cdef int t, i, j, k
    cdef double[:,:] r_half = np.zeros((part_num, 3), dtype=np.double)
    cdef double[:,:,:] a_half = np.zeros((steps, part_num, 3), dtype=np.double)
    cdef double x, y, z, r_dist

    start = openmp.omp_get_wtime()
    for t in range(steps):
        with nogil, parallel(num_threads=threads):
            for i in prange(part_num):
                for k in range(3):
                    r_half[i, k] = r[t, i, k] + v[t, i, k] * timestep / 2

            for i in prange(part_num):
                for j in range(part_num):
                    if i != j:
                        x = r_half[j, 0] - r_half[i, 0]
                        y = r_half[j, 1] - r_half[i, 1]
                        z = r_half[j, 2] - r_half[i, 2]
                        r_dist = sqrt(x * x + y * y + z * z)
                        a_half[t, i, 0] += G * mass[j] / (r_dist ** 3) * x
                        a_half[t, i, 1] += G * mass[j] / (r_dist ** 3) * y
                        a_half[t, i, 2] += G * mass[j] / (r_dist ** 3) * z

                for k in range(3):
                    v[t + 1, i, k] = v[t, i, k] + a_half[t, i, k] * timestep
                    r[t + 1, i, k] = r[t, i, k] + (v[t, i, k] + v[t + 1, i, k]) * timestep / 2

    final = openmp.omp_get_wtime()
    print("Simulation time:", final - start)
