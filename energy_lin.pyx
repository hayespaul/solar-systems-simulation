# cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
import numpy as np
import scipy.linalg as lin
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from cython.parallel cimport parallel, prange
from libc.math cimport sqrt
cimport openmp


############# CODE USED TO FIND ENERGIES OF ORBIT AND FFT PLOT ####################

def main(int part_num):
    cdef:
        int threads = 5
        int dim     = 3
        double timestep = 60*60*24*10
        int steps    = 100000
        double G     = 6.67408E-11
        double tot_init_e = 0
        double tot_final_e= 0
        double x
        double y
        double z
        double ave_en
        int p = 0

        int i,j,k,t
        double r_dist
        double ast_mass          = 2e15
        double[:,:,:] r          = np.zeros((steps+1,part_num,dim), dtype=np.double)
        double[:,:,:] v          = np.zeros((steps+1,part_num,dim), dtype=np.double)
        double[:,:] r_half       = np.zeros((part_num,dim), dtype=np.double)
        double[:,:,:] a_half     = np.zeros((steps,part_num,dim), dtype=np.double)
        double[:]     r_ij       = np.zeros(dim,dtype=np.double)
        double[:]     mass       = np.zeros(part_num,dtype=np.double)
        double     pot_energy
        double     v_mag
        double[:]     tot_energy = np.zeros(steps,dtype=np.double)
        double[:]     stepsa     = np.linspace(0,steps,steps,dtype=np.double)
        double[:]     times      = np.zeros(steps,dtype=np.double)
        double[:]     stepsa2   = np.linspace(0,steps,steps/1000,dtype=np.double)
        double[:]     times2      = np.zeros(steps/1000,dtype=np.double)

        double x_tot_mom
        double y_tot_mom
        double z_tot_mom
        double[:]   frq=np.linspace(0,1/timestep,steps,dtype=np.double)
        double[:]   yea=np.arange(steps,dtype=np.double)
        double[:]   ave_ener=np.zeros(steps/1000,dtype=np.double)


    ##### INITIAL CONDITIONS #####

    mass[0]=1.9891E30 # sun
    mass[1]=3.302E23 # mercury
    mass[2]=4.868E24 # venus
    mass[3]=5.972E24 # earth
    mass[4]=6.417E23 # mars
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
    r[0,1,0]=5.79487E10
    r[0,1,1]=0
    r[0,1,2]=0

    v[0,1,1] = sqrt((G*mass[0])/(r[0,1,0]))
    v[0,1,0]=0
    v[0,1,2]=0
    ###############################

    ############ venus #############
    r[0,2,0]=1.08E11
    r[0,2,1]=0
    r[0,2,2]=0

    v[0,2,0]=0
    v[0,2,1]=-np.sqrt((G*mass[0])/r[0,2,0])
    v[0,2,2]=0
    ###############################

    ############ earth ############
    r[0,3,0]=1.496E11
    r[0,3,1]=0
    r[0,3,2]=0

    v[0,3,1] = np.sqrt((G*mass[0])/r[0,3,0])
    v[0,3,0]=0
    v[0,3,2]=0
    ###############################

    ############ mars #############
    r[0,4,0]=2.27E11
    r[0,4,1]=0
    r[0,4,2]=0

    v[0,4,1]=np.sqrt((G*mass[0])/r[0,4,0])
    v[0,4,0]=0
    v[0,4,2]=0
    ###############################

    ############ asteroid belt #############

    r[0,9,0]=3.29E11
    v[0,9,1] = np.sqrt((G*mass[0])/r[0,9,0])

    for i in range(10,part_num):
        r[0,i,0] = r[0,i-1,0] + (4.79E11 - 3.29E11)/(part_num-9)
        v[0,i,1] = np.sqrt((G*mass[0])/r[0,i,0])

    ###############################

    ############ jupiter #############
    r[0,5,0]=7.78E11
    r[0,5,1]=0
    r[0,5,2]=0

    v[0,5,1]=np.sqrt((G*mass[0])/r[0,5,0])
    v[0,5,0]=0
    v[0,5,2]=0
    ###############################

    ############ saturn #############
    r[0,6,0]=14.27E11
    r[0,6,1]=0
    r[0,6,2]=00

    v[0,6,1]=np.sqrt((G*mass[0])/r[0,6,0])
    v[0,6,0]=0
    v[0,6,2]=0
    ###############################

    ############ uranus #############
    r[0,7,0]=28.7E11
    r[0,7,1]=0
    r[0,7,2]=0

    v[0,7,1]=np.sqrt((G*mass[0])/r[0,7,0])
    v[0,7,0]=0
    v[0,7,2]=0
    ###############################

    ############ neptune #############
    r[0,8,0]=44.97E11
    r[0,8,1]=0
    r[0,8,2]=0

    v[0,8,1]=np.sqrt((G*mass[0])/r[0,8,0])
    v[0,8,0]=0
    v[0,8,2]=0
    ###############################

    for i in range(1,part_num):
       x_tot_mom += mass[i]*v[0][i][0]
       y_tot_mom += mass[i]*v[0][i][1]
       z_tot_mom += mass[i]*v[0][i][2]

    v[0,0,0] = -x_tot_mom/mass[0]
    v[0,0,1] = -y_tot_mom/mass[0]
    v[0,0,2] = -z_tot_mom/mass[0]

    start = openmp.omp_get_wtime()
    for t in range(steps):
        for i in prange(part_num,nogil=True,num_threads=threads):
           for k in range(dim):
                r_half[i,k] = r[t,i,k] + v[t,i,k]*(timestep/2)

        for i in range(part_num):
            for j in range(part_num):
                if i!=j:
                    x=r_half[j,0] - r_half[i,0]
                    y=r_half[j,1] - r_half[i,1]
                    z=r_half[j,2] - r_half[i,2]

                    r_dist = sqrt((x*x) + (y*y) + (z*z))

                    a_half[t,i,0] += G*((mass[j])/(r_dist**3)) * x
                    a_half[t,i,1] += G*((mass[j])/(r_dist**3)) * y
                    a_half[t,i,2] += G*((mass[j])/(r_dist**3)) * z

                    pot_energy += G*mass[j]*mass[i]/r_dist

            for k in range(dim):
                v[t+1,i,k] = v[t,i,k] + a_half[t,i,k]*(timestep)
                r[t+1,i,k] = r[t,i,k] + (v[t,i,k] + v[t+1,i,k])*(timestep/2)

            v_mag=sqrt(((v[t][i][0])**2 + (v[t][i][1])**2 + (v[t][i][2])**2))
            tot_energy[t] += (0.5*(mass[i]*(v_mag**2)) - pot_energy)

            v_mag=0
            r_dist=0
            pot_energy=0
        if t%1000==0:
          # WORK OUT the average energy every 1000 steps
          for i in range(1000):
            ave_en+=tot_energy[t-i]
          ave_ener[p]=ave_en/1000
          p+=1
        ave_en=0
        print(t,'/',steps)

    for i in range(len(times2)):
      times2[i]= stepsa2[i]*timestep  # change to time

    final = openmp.omp_get_wtime()
    diff = final - start

    print('time=',diff)
    #print('thread=',threads)
    print('part_num=',part_num)

    ave_en=np.sum(tot_energy)/steps

    for i in range(len(tot_energy)):
      tot_energy[i] = tot_energy[i] - ave_en  # offset to average = 0


    fft = np.fft.fft(tot_energy)

    for i in range(len(times)):
      times[i]=stepsa[i]*timestep

    N = steps
    """
    plt.ylabel("Amplitude")
    plt.xlabel("Time [s]")
    plt.plot(times, tot_energy)
    plt.show()

    plt.ylabel("Amplitude")
    plt.xlabel("Frequency [Hz]")
    plt.plot(frq[:N // 2], abs(fft)[:N//2]*1/N)
    plt.show()
    """
    top=np.max(ave_ener[1:])
    bottom=np.min(ave_ener[1:])

    diff=top-bottom

    plt.ylabel("Average energy")
    plt.xlabel("Time(s)")
    plt.plot(times2[1:],ave_ener[1:])
    plt.ylim(bottom - diff*100,top + diff*100)
    plt.show()

    plt.plot(r[:,:,0],r[:,:,1])
    plt.show()
