import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt

def initialize_simulation(part_num, steps):
    dimensions = 3
    gravitational_constant = 6.67408e-11
    asteroid_mass = 2e15
    masses = np.array([1.989e30, 3.302e23, 4.868e24, 5.972e24, 6.417e23, 1.898e27, 5.685e26, 8.682e25, 1.024e26] + [asteroid_mass] * (part_num - 9))

    positions = np.zeros((steps + 1, part_num, dimensions))
    velocities = np.zeros((steps + 1, part_num, dimensions))

    # Initial conditions for celestial bodies
    celestial_data = [  # (position_x, position_z, velocity_y)
        (0, 0, 0),  # Sun
        (-5.79487e10, 0, None),
        (-1.08e11, 0, None),
        (1.52e11, 0, None),
        (-2.27e11, 0, None),
        (-7.78e11, 0, None),
        (14.27e11, 0, None),
        (28.7e11, 0, None),
        (44.97e11, 0, None),
    ]

    for i, (x, z, vy) in enumerate(celestial_data):
        z += (np.random.rand() * 1e4) - 5e2
        positions[0, i] = [x, 0, z]
        if vy is None:
            distances = np.linalg.norm(positions[0, i])
            velocities[0, i, 1] = np.sqrt((gravitational_constant * masses[0]) / distances)

    # Initial conditions for asteroids
    for i in range(9, part_num):
        x = np.linspace(3.29e11, 4.79e11, num=part_num - 9)[i - 9]
        z = (np.random.rand() * 1e5) - 5e4
        positions[0, i] = [x, 0, z]
        distances = np.linalg.norm(positions[0, i, [0, 2]])
        velocities[0, i, 1] = np.sqrt((gravitational_constant * masses[0]) / distances)

    # Adjust the sun's velocity to maintain the system's momentum balance
    total_momentum = np.sum(velocities[0, 1:] * masses[1:, None], axis=0)
    velocities[0, 0] = -total_momentum / masses[0]

    forces = np.zeros((part_num, dimensions))
    return positions, velocities, forces, masses

def calculate_forces(positions, forces, offset, chunk_size, gravitational_constant, masses, number_of_particles):
    for i in range(1, number_of_particles):
        relative_positions = positions[offset:(offset + chunk_size)] - np.roll(positions, shift=i, axis=0)[offset:(offset + chunk_size)]
        distances_squared = np.sum(relative_positions**2, axis=1)
        force_magnitude = gravitational_constant * masses[i] / np.sqrt(distances_squared)**3
        forces[offset:(offset + chunk_size)] -= (relative_positions.T * force_magnitude).T
    return forces

def main_simulation():
    comm = MPI.COMM_WORLD
    num_ranks = comm.Get_size()
    rank = comm.Get_rank()
    master_rank = 0
    master_tag = 1
    worker_tag = 2

    part_num = 2000
    steps = 200
    gravitational_constant = 6.67408e-11
    timestep = 10 * 24 * 60 * 60

    if rank == master_rank:
        positions, velocities, forces, masses = initialize_simulation(part_num, steps)
        start_time = MPI.Wtime()

    chunk_size = 2 * part_num // (2 * num_ranks + 1)
    if rank != master_rank:
        positions = np.zeros((steps, part_num, 3))
        forces = np.zeros((part_num, 3))
        masses = np.zeros(part_num)

    for step in range(steps):
        if rank == master_rank:
            half_step_positions = positions[step] + velocities[step] * (timestep / 2)
            for worker_rank in range(1, num_ranks):
                comm.send(half_step_positions, dest=worker_rank, tag=master_tag)

            forces = calculate_forces(half_step_positions, forces, 0, chunk_size, gravitational_constant, masses, part_num)

            for worker_rank in range(1, num_ranks):
                worker_forces = comm.recv(source=worker_rank, tag=worker_tag)
                forces += worker_forces

            velocities[step + 1] = velocities[step] + forces * timestep
            positions[step + 1] = positions[step] + 0.5 * (velocities[step] + velocities[step + 1]) * timestep

        else:
            half_step_positions = comm.recv(source=master_rank, tag=rank)
            forces_chunk = calculate_forces(half_step_positions, np.zeros((chunk_size, 3)), rank * chunk_size, chunk_size, gravitational_constant, masses, part_num)
            comm.send(forces_chunk, dest=master_rank, tag=rank)

    if rank == master_rank:
        end_time = MPI.Wtime()
        print(f"Simulation completed in {end_time - start_time} seconds.")
        for dim, label in zip((1, 2), ('y', 'z')):
            plt.figure()
            for i in range(part_num):
                plt.plot(positions[:, i, 0], positions[:, i, dim])
            plt.xlabel('x (m)')
            plt.ylabel(f'{label} (m)')
            plt.show()

if __name__ == "__main__":
    main_simulation()
