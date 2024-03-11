# cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
import numpy as np
cimport numpy as np
from libc.math cimport sqrt
from cython.parallel cimport prange
cimport openmp

def main(int part_num):
    # Constants and Parameters
    cdef int dim = 3, threads = 5, steps = 100000
    cdef double timestep = 60 * 60 * 24 * 10  # 10 days in seconds
    cdef double G = 6.67408E-11  # Gravitational constant

    # Arrays Initialization
    cdef double[:,:,:] r = np.zeros((steps+1, part_num, dim), dtype=np.double)
    cdef double[:,:,:] v = np.zeros((steps+1, part_num, dim), dtype=np.double)
    cdef double[:] mass = np.zeros(part_num, dtype=np.double)
    cdef double[:] tot_energy = np.zeros(steps, dtype=np.double)
    cdef double[:] frq = np.linspace(0, 1/timestep, steps, dtype=np.double)
    cdef double[:] ave_ener = np.zeros(steps // 1000, dtype=np.double)

    # Initial conditions setup
    setup_initial_conditions(r, v, mass, part_num, G)

    # Run simulation
    simulate(r, v, mass, part_num, dim, steps, timestep, G, threads, tot_energy, ave_ener)

    # Post-processing
    plot_results(r, ave_ener, timestep)

def setup_initial_conditions(double[:,:,:] r, double[:,:,:] v, double[:] mass, int part_num, double G):
    # Assign masses (Sun, planets, and asteroids)
    ast_mass = 2e15
    mass[:10] = [1.9891E30, 3.302E23, 4.868E24, 5.972E24, 6.417E23, 1.898E27, 5.685E26, 8.682E25, 1.024E26]
    mass[10:] = ast_mass

    # Planetary data: position (AU) and initial velocity calculation
    cdef double[9] positions_AU = [0, 0.387, 0.723, 1, 1.524, 5.203, 9.537, 19.191, 30.069]  # AU distances
    cdef double AU_to_meters = 1.496e11  # AU in meters
    cdef int i
    for i in range(9):
        r[0, i, 0] = positions_AU[i] * AU_to_meters
        if i > 0:  # Skip the Sun
            v[0, i, 1] = sqrt(G * mass[0] / r[0, i, 0])  # Assuming circular orbits

    # Asteroid belt initialization
    initialize_asteroid_belt(r, v, mass, G, part_num)

def initialize_asteroid_belt(double[:,:,:] r, double[:,:,:] v, double[:] mass, double G, int part_num):
    cdef int i
    for i in range(10, part_num):
        r[0, i, 0] = r[0, i-1, 0] + 3.5e10  # Arbitrary spacing
        v[0, i, 1] = sqrt(G * mass[0] / r[0, i, 0])  # Circular orbit velocity assumption

def simulate(double[:,:,:] r, double[:,:,:] v, double[:] mass, int part_num, int dim, int steps, double timestep, double G, int threads, double[:] tot_energy, double[:] ave_ener):
    cdef int t, i, j, k, p = 0
    cdef double[:,:] r_half = np.zeros((part_num, dim), dtype=np.double)
    cdef double[:,:,:] a_half = np.zeros((steps, part_num, dim), dtype=np.double)
    cdef double pot_energy = 0, v_mag = 0, ave_en = 0
    cdef double x, y, z, r_dist

    cdef double start = openmp.omp_get_wtime()
    for t in range(steps):
        # Update positions and velocities
        update_positions_velocities(r, v, a_half, r_half, mass, part_num, dim, timestep, t, threads)

        # Calculate total energy and average energy periodically
        calculate_energies(r, v, mass, part_num, dim, t, G, tot_energy, pot_energy, v_mag)

        if t % 1000 == 0 and t > 0:
            ave_ener[p] = np.mean(tot_energy[t-1000:t])
            p += 1

    cdef double final = openmp.omp_get_wtime()
    print('Simulation time:', final - start)

def update_positions_velocities(double[:,:,:] r, double[:,:,:] v, double[:,:,:] a_half, double[:,:] r_half, double[:] mass, int part_num, int dim, double timestep, int t, int threads):
    with nogil, parallel(num_threads=threads):
        for i in prange(part_num):
            for k in range(dim):
                r_half[i, k] = r[t, i, k] + v[t, i, k] * timestep / 2

    for i in range(part_num):
        for j in range(part_num):
            if i != j:
                calculate_acceleration(r_half, a_half, mass, i, j, t, G)

        for k in range(dim):
            v[t + 1, i, k] = v[t, i, k] + a_half[t, i, k] * timestep
            r[t + 1, i, k] = r[t, i, k] + (v[t, i, k] + v[t + 1, i, k]) * timestep / 2

def calculate_acceleration(double[:,:] r_half, double[:,:,:] a_half, double[:] mass, int i, int j, int t, double G):
    cdef double x = r_half[j, 0] - r_half[i, 0]
    cdef double y = r_half[j, 1] - r_half[i, 1]
    cdef double z = r_half[j, 2] - r_half[i, 2]
    cdef double r_dist = sqrt(x * x + y * y + z * z)

    a_half[t, i, 0] += G * mass[j] / (r_dist ** 3) * x
    a_half[t, i, 1] += G * mass[j] / (r_dist ** 3) * y
    a_half[t, i, 2] += G * mass[j] / (r_dist ** 3) * z

def calculate_energies(double[:,:,:] r, double[:,:,:] v, double[:] mass, int part_num, int dim, int t, double G, double[:] tot_energy, double pot_energy, double v_mag):
    cdef int i, k
    for i in range(part_num):
        v_mag = sqrt(v[t, i, 0] ** 2 + v[t, i, 1] ** 2 + v[t, i, 2] ** 2)
        for j in range(part_num):
            if i != j:
                pot_energy += -G * mass[i] * mass[j] / sqrt((r[t, i, 0] - r[t, j, 0]) ** 2 + (r[t, i, 1] - r[t, j, 1]) ** 2 + (r[t, i, 2] - r[t, j, 2]) ** 2)
        tot_energy[t] += 0.5 * mass[i] * v_mag ** 2 + pot_energy
        pot_energy = 0  # Reset potential energy for the next particle

def plot_results(double[:,:,:] r, double[:] ave_ener, double timestep):
    cdef int steps = ave_ener.shape[0] * 1000
    cdef double[:] times2 = np.linspace(0, steps * timestep, ave_ener.shape[0], dtype=np.double)

    # Plotting the average energy over time
    plt.figure()
    plt.ylabel("Average energy")
    plt.xlabel("Time [s]")
    plt.plot(times2, ave_ener)
    plt.show()

    # Plotting the orbits in the xy plane
    plt.figure()
    plt.plot(r[:, :, 0], r[:, :, 1])
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.show()
