import numpy as np
import matplotlib.pyplot as plt


### PLOTS FOR THE PARTICLE NUMBER AGAINST COMPUTE TIME ####


part_m=[2500,4000,4500,5000,5500,6000,6500,7000,7500,10000,12500,15000,17500,20000]
part_l=[2500,5000,10000,12500,15000,20000,40000]
time_m=[6.2,11.93,14.2,16.6,19.2,22.05,25.1,27.9,31.9,50.5,73.9,100.3,131.995,168.9]
time_m_8=[7.8,16.3,20.1,23.8,27.9,32.5,37.8,42.7,48.3,82.7,122.5,173.4,230.4,287.4]
time_l=[4.04,16.1,64.2,100,143.9,253.9,1077.1]

part_omp=[10,20,30,40,50,100,150,200,300,400,500,600,1000,1500,2000,3000,4000,5000,6000,7000]
time_omp_12=[0.0025,0.00253,0.0027,0.0143,0.00566,0.00455,0.00744,0.01,0.022,0.037,0.055,0.079,0.213,0.47,0.84,1.87,3.32,5.19,7.46,10.15]
time_omp_16=[0.00067,0.00089,0.00105,0.0013,0.00139,0.0023,0.00487,0.0079,0.016,0.0268,0.043,0.059,0.16,0.356,0.63,1.422,2.5,3.89,5.6,7.6]
part_omp2=[5,10,20,30,40,50,60,70,80,90,100,200,300,400,500,600,700,800,900,1000]
time_omp_16_2=[0.00288,0.00384,0.0057,0.00771,0.00967,0.01087,0.01157,0.0184,0.0153,0.02258,0.02236,0.073,0.155,0.265,0.413,0.59,0.79,1.05,1.3,1.6]
time_omp_12_2=[0.00198,0.00294,0.00603,0.00616,0.0072,0.00985,0.0144,0.016,0.01773,0.02237,0.02712,0.09419,0.20533,0.35542,0.5442,0.776,1.05,1.36,1.67,2.05]
time_omp_8_2=[0.00197,0.00255,0.00429,0.00529,0.00756,0.01007,0.01315,0.01882,0.0211,0.02772,0.03396,0.12421,0.27663,0.483,0.75525,1.087,1.465,1.922,2.4127,2.983]
for i in range(len(time_omp_16_2)):
    time_omp_16_2[i]=time_omp_16_2[i]/10
    time_omp_12_2[i]=time_omp_12_2[i]/10
    time_omp_8_2[i]=time_omp_8_2[i]/10
time_omp_8=[0.001,0.0017,0.0016,0.00235,0.00193,0.00455,0.00854,0.0137,0.0298,0.05,0.078,0.112,0.307,0.68,1.21,2.72,4.8,7.53,10.8,14.9]
#time_omp_8=[0.0047,0.0006,0.00069,0.001,0.001,0.0035,0.00729,0.0126,0.028,0.048,0.0758,0.108,0.297,0.661,1.18,2.64,4.73,7.37,10.6,14.4]
time_omp_4=[0.00089,0.00099,0.001,0.00175,0.0022,0.0069,0.015,0.0256,0.057,0.101,0.158,0.22,0.627,1.4,2.5,5.6,9.95,15.5,22.3,29.7]
#1100,1200,1300,1400,1500,1600,1700,1800,1900,2000,2100,2200
#[10,20,30,40,50,60,70,80,90,100,200,300,400,600,700,800,900,1000]
#lin[6.22E-5, 0.00023,0.00053,0.00094,0.00146,0.00211,0.00288,0.00375,0.00476,0.00588,0.0236,0.0588,0.0236,0.0542,0.095,0.2148,0.292,0.383,0.484,0.597]
#omp[0.00281,0.00285,0.0162,0.02156,0.03,0.00315,0.017,0.016,0.0178,0.022,0.023,0.033,0.038,0.094,0.087,0.11,0.14,0.16]
parts=[10,20,30,40,50,60,70,80,90,100,200,300,400,600,700,800,900,1000,1000,1100,1200,1300,1400,1600,1700,1800,1900,2000,2100,2200,2500,5000,10000,12500,15000,20000]
time=[0.00281,0.00285,0.0162,0.02156,0.03,0.00315,0.017,0.016,0.0178,0.022,0.023,0.033,0.038,0.094,0.087,0.11,0.14,0.16,0.16,0.19,0.22,0.268,0.311,0.4,0.457,0.511,0.568,0.63,0.695,0.763,0.98,3.899,15.5,24.4,34.87,61.97]
time_lin=[6.22E-5,0.00023,0.00053,0.00094,0.00146,0.00211,0.00288,0.00375,0.00476,0.00588,0.0236,0.0542,0.0953,0.2148,0.292,0.383,0.484,0.597,0.59,0.722,0.859,1.009,1.17,1.529,1.727,1.935,2.158,2.392,2.63,2.89,4.04,16.1,64.22,100,143,253.97]
log_part=np.log(parts)
log_time=np.log(time)
log_time_l=np.log(time_lin)
log_part_m=np.log(part_m)
log_part_l=np.log(part_l)
log_t_m=np.log(time_m)
log_t_l=np.log(time_l)
log_part_omp=np.log(part_omp)
log_t_omp_12=np.log(time_omp_12)


fit=np.polyfit(log_part,log_time,1)
fit_fn=np.poly1d(fit)
fit2=np.polyfit(log_part,log_time_l,1)
fit_fn2=np.poly1d(fit2)
#fit3=np.polyfit(log_part_m,log_time_t_m,1)
#fit_fn2=np.poly1d()


plt.xlabel('Number of particles')
plt.ylabel('Time (s)')
plt.plot(parts,time,'ro')
plt.show()

plt.xlabel('log(particle number)')
plt.ylabel('log(time(s))')
#plt.plot(log_part,log_time,marker='o',label='openmp')
#plt.plot(log_part,fit_fn(log_part),'--k')
plt.plot(log_part,log_time_l,marker='o',label='linear',markersize=4,linewidth=1.5)
#plt.plot(log_part_omp,log_t_omp_12,marker='o',label='omp_12',markersize=4,linewidth=1.5)
#plt.plot(log_part_omp,np.log(time_omp_16),marker='o',label='omp_16',markersize=4,linewidth=1.5)
#plt.plot(log_part_omp,np.log(time_omp_8),marker='o',label='omp_8',markersize=4,linewidth=1.5)
#plt.plot(log_part_omp,np.log(time_omp_4),marker='o',label='omp_4',markersize=4,linewidth=1.5)
plt.plot(np.log(part_omp2),np.log(time_omp_16_2),marker='o',label='omp_16_2',markersize=4,linewidth=1.5)
plt.plot(np.log(part_omp2),np.log(time_omp_12_2),marker='o',label='omp_12_2',markersize=4,linewidth=1.5)
plt.plot(np.log(part_omp2),np.log(time_omp_8_2),marker='o',label='omp_8_2',markersize=4,linewidth=1.5)
#plt.plot(log_part,fit_fn2(log_part),'--k')
plt.legend(loc='best')
plt.show()


plt.xlabel('log(particle number)')
plt.ylabel('log(time(s))')
plt.plot(log_part_m,log_t_m,marker='o',label='mpi 16 thread')
plt.plot(log_part_m,np.log(time_m_8),marker='o',label='mpi 8 thread')
#plt.plot(log_part,fit_fn(log_part),'--k')
plt.plot(log_part_l,log_t_l,marker='o',label='linear')
#plt.plot(log_part,fit_fn2(log_part),'--k')
plt.legend(loc='best')
plt.show()
