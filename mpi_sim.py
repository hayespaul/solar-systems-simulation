import numpy as np
from mpi4py import MPI
from numpy import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# MPI code with python, with attempts to speedup with numpy

def init_arrays(part_num,steps):
    dim      = 3
    G        = 6.67408E-11
    ast_mass = 2e15

    r      = np.zeros((steps+1,part_num,dim))
    v      = np.zeros((steps+1,part_num,dim))
    mass = np.zeros(part_num)
    x_tot_mom=0
    y_tot_mom=0
    z_tot_mom=0

    mass[0]=1.989E30 # sun
    mass[1]=3.302E23 # mercury
    mass[2]=4.868E24 # venus
    mass[3]=5.972E24 # earth
    mass[4]=6.417E23  # mars
    mass[5]=1.898E27 # jupiter
    mass[6]=5.685E26 # saturn
    mass[7]=8.682E25 # uranus
    mass[8]=1.024E26 # neptune
    mass[9:]=ast_mass

    ############ sun ##############
    v[0,0,0]=0
    v[0,0,1]=0
    v[0,0,2]=0

    r[0,0,0]=0
    r[0,0,1]=0
    r[0,0,2]=0
    ###############################

    ############ mercury #############
    r[0,1,0]=-5.79487E10
    r[0,1,1]=0
    r[0,1,2]=(np.random.rand()*1E4) -5e2

    v[0,1,1] = np.sqrt((G*mass[0])/np.sqrt(r[0,1,0]**2 + r[0,1,2]**2))
    v[0,1,0]=0
    v[0,1,2]=0
    ###############################

    ############ venus #############
    r[0,2,0]=-1.08E11
    r[0,2,1]=0
    r[0,2,2]=(np.random.rand()*1E4) -5e2

    v[0,2,0]=0
    v[0,2,1]=np.sqrt((G*mass[0])/np.sqrt(r[0,2,0]**2 + r[0,2,2]**2))
    v[0,2,2]=0
    ###############################

    ############ earth ############
    r[0,3,0]=1.52E11
    r[0,3,1]=0
    r[0,3,2]=(np.random.rand()*1E4) -5e2

    v[0,3,1] = np.sqrt((G*mass[0])/np.sqrt(r[0,3,0]**2 + r[0,3,2]**2))
    v[0,3,0]=0
    v[0,3,2]=0
    ###############################

    ############ mars #############
    r[0,4,0]=-2.27E11
    r[0,4,1]=0
    r[0,4,2] = (np.random.rand()*1E4) - 5e2

    v[0,4,1]=np.sqrt((G*mass[0])/np.sqrt(r[0,4,0]**2 + r[0,4,2]**2))
    v[0,4,0]=0
    v[0,4,2]=0
    ###############################

    ############ asteroid belt #############

    r[0,9,0]=3.29E11
    v[0,9,1] = np.sqrt((G*mass[0])/r[0,9,0])

    for i in range(10,part_num):
        r[0,i,0] = r[0,i-1,0] + (4.79E11 - 3.29E11)/(part_num-9)
        r[0,i,2] = (np.random.rand()*1E5) -5E4
        v[0,i,1] = np.sqrt((G*mass[0])/np.sqrt(r[0,i,0]**2 + r[0,i,2]**2))

    ###############################

    ############ jupiter #############
    r[0,5,0]=-7.78E11
    r[0,5,1]=0
    r[0,5,2]=(np.random.rand()*1E4) -5e2

    v[0,5,1]=np.sqrt((G*mass[0])/np.sqrt(r[0,5,0]**2 + r[0,5,2]**2))
    v[0,5,0]=0
    v[0,5,2]=0
    ###############################

    ############ saturn #############
    r[0,6,0]=14.27E11
    r[0,6,1]=0
    r[0,6,2]=(np.random.rand()*1E4) -5e2

    v[0,6,1]=np.sqrt((G*mass[0])/np.sqrt(r[0,6,0]**2 + r[0,6,2]**2))
    v[0,6,0]=0
    v[0,6,2]=0
    ###############################

    ############ uranus #############
    r[0,7,0]=28.7E11
    r[0,7,1]=0
    r[0,7,2]=(np.random.rand()*1E4) -5e2

    v[0,7,1]=np.sqrt((G*mass[0])/np.sqrt(r[0,7,0]**2 + r[0,7,2]**2))
    v[0,7,0]=0
    v[0,7,2]=0
    ###############################

    ############ neptune #############
    r[0,8,0]=44.97E11
    r[0,8,1]=0
    r[0,8,2]=(np.random.rand()*1E4) -5e2

    v[0,8,1]=np.sqrt((G*mass[0])/np.sqrt(r[0,8,0]**2 + r[0,8,2]**2))
    v[0,8,0]=0
    v[0,8,2]=0
    ###################
    forces=np.zeros(shape=(N,3))

    for i in range(1,part_num):
       x_tot_mom += mass[i]*v[0][i][0]
       y_tot_mom += mass[i]*v[0][i][1]
       z_tot_mom += mass[i]*v[0][i][2]

    v[0,0,0] = -x_tot_mom/mass[0]
    v[0,0,1] = -y_tot_mom/mass[0]
    v[0,0,2] = -z_tot_mom/mass[0]

    return r,v,forces,mass

# Try couple ways of this
def force_calc(positions,forces,offset,chunk,G,M,N,steps):
    _,_,_,M=init_arrays(N,steps)
    positions2 = positions.copy()
    positions2 = roll(positions2,1,axis=0) # shifts values along so j loop can be avoided
    M=roll(M,1,axis=0)
    for i in range(N-1):
        rel_pos = positions[offset:(offset+chunk)]-positions2[offset:(offset+chunk)]
        rel_pos_mag = sum(power(rel_pos,2),axis=1)
        forces[offset:(offset+chunk)][:,0] -= multiply(
                                                G*M[offset:(offset+chunk)]/(power(rel_pos_mag,(3/2))),
                                                rel_pos[:,0])
        forces[offset:(offset+chunk)][:,1] -= multiply(
                                                G*M[offset:(offset+chunk)]/(power(rel_pos_mag,(3/2))),
                                                rel_pos[:,1])
        forces[offset:(offset+chunk)][:,2] -= multiply(
                                                G*M[offset:(offset+chunk)]/(power(rel_pos_mag,(3/2))),
                                                rel_pos[:,2])
        positions2 = roll(positions2,1,axis=0) #shifts along for next group of particles
        M=roll(M,1,axis=0) # shifts masses for same reason
    return forces


# MPI variables and stuff
comm = MPI.COMM_WORLD

numranks = comm.Get_size()
rank = comm.Get_rank()

Master = 0
Master_tag = 1
Worker_tag = 2

# Normal python variables and stuff
N = 2000
steps = 200
G = 6.67e-11
timestep = 10*24*60*60
r_half=np.zeros(shape=(N,3))
mass=np.zeros(shape=(N))

rank = comm.Get_rank()
chunk = 2*N // (2*numranks+1)
# Simply Initialising
if rank == Master:
    # Initialising arrays
    start = MPI.Wtime()
    r,v,forces,mass= init_arrays(N,steps) # initialise arrays at master and send to workers

if rank != Master:
    r = zeros(shape=(steps,N,3))
    forces    = zeros(shape=(N,3))
    mass = zeros(shape=(N))  # workers have buffer arrays

for t in range(steps):
    if rank == Master:
        for i in range(N):
           for k in range(3):
                r_half[i,k] = r[t,i,k] + v[t,i,k]*(timestep/2)
        offset = 0
        for worker_num in range(1,numranks):
            comm.send(offset,dest=worker_num,tag=Master_tag)
            comm.Send(r_half[:,:],dest=worker_num,tag=Master_tag)
            #comm.Send(mass[:],dest=worker_num,tag=Master_tag)

            offset += chunk

        forces[offset:N:] = 0
        forces = force_calc(r_half,forces,offset,N-offset,G,mass,N,steps)  # let master do a force calc to decrease computation time

        for worker_num in range(1,numranks):
            offset = comm.recv(source = worker_num, tag=Worker_tag)
            comm.Recv([forces[offset:offset+chunk],N*3,MPI.DOUBLE],source=worker_num,tag=Worker_tag)

        v[t+1,:,:] = v[t,:,:] + forces[:,:]*(timestep)
        r[t+1,:,:] = r[t,:,:] + (v[t,:,:] + v[t+1,:,:])*(timestep/2)

    if rank != Master:  # worker does force calculation
        offset = comm.recv(source=Master, tag=Master_tag)
        #if t == 0: print(offset)
        comm.Recv([r_half[:,:],N*3,MPI.DOUBLE],source=Master,tag=Master_tag)
        #comm.Recv([mass[:],N,MPI.DOUBLE],source=Master,tag=Worker_tag)
        forces[offset:offset+chunk:] = 0
        forces = force_calc(r_half,forces,offset,chunk,G,mass,N,steps)

        comm.send(offset,dest=Master,tag=Worker_tag)
        comm.Send(forces[offset:offset+chunk],dest=Master,tag=Worker_tag) # sends back to master

if rank == Master: # once finished, time and print graphs
    end = MPI.Wtime()
    #save("FullPositions.npy",PositionsFull)
    print(end-start)
    print(N)
    print(steps)
    #plt.plot(r[:,:,0],r[:,:,1])
    #plt.xlabel('x(m)')
    #plt.ylabel('y(m)')
    #plt.show()
    #plt.plot(r[:,:,0],r[:,:,2])
    #plt.xlabel('x(m)')
    #plt.ylabel('z(m)')
    #plt.show()
