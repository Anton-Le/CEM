#!/bin/env python3

import numpy as np
import scipy as sp
from matplotlib import pyplot as plt



eps0 = 8.8541878e-12;
mu0 = 4e-7 * np.pi;
c0= 299792458;

# geometry data
Lx = .05; 
Ly = .04; 
Lz = .03;
Nx = 25; 
Ny = 20; 
Nz = 15;
Cx = Nx / Lx;
Cy = Ny / Ly;
Cz = Nz / Lz;
Nt = 8192;
Dt = 1.0/(c0*np.linalg.norm(np.array([Cx, Cy, Cz]), ord=2 ))
print(Dt)

Ex = np.zeros((Nx ,Ny+1, Nz+1));
Ey = np.zeros((Nx+1,Ny , Nz+1));
Ez = np.zeros((Nx+1,Ny+1, Nz ));
Hx = np.zeros((Nx+1,Ny , Nz ));
Hy = np.zeros((Nx ,Ny+1, Nz ));
Hz = np.zeros((Nx ,Ny , Nz+1));

Et = np.zeros((Nt,3));

Ex[ : , 1:-1, 1:-1] = np.random.rand( Nx , Ny-1, Nz-1) - 0.5
Ey[1:-1, : , 1:-1] = np.random.rand(Nx-1, Ny, Nz-1) - 0.5
Ez[1:-1, 1:-1, : ] = np.random.rand(Nx-1, Ny-1, Nz) - 0.5

import h5py
# Load reference data from Octave
#datafile=h5py.File("Exarray.h5")
#data = datafile['Ex']['value'][:,:,:]
#Ex = data.transpose()
#datafile.close()

#datafile=h5py.File("Eyarray.h5")
#data = datafile['Ey']['value'][:,:,:]
#Ey = data.transpose()
#datafile.close()

#datafile=h5py.File("Ezarray.h5")
#data = datafile['Ez']['value'][:,:,:]
#Ez = data.transpose()
#datafile.close()

for t in range(Nt):
    Hx = Hx + (Dt/mu0)*(np.diff(Ey,axis=2)*Cz - np.diff(Ez,axis=1)*Cy);
    Hy = Hy + (Dt/mu0)*(np.diff(Ez,axis=0)*Cx - np.diff(Ex,axis=2)*Cz);
    Hz = Hz + (Dt/mu0)*(np.diff(Ex,axis=1)*Cy - np.diff(Ey,axis=0)*Cx);
    # update E fields
    Ex[:,1:-1,1:-1] += (Dt /eps0) * (np.diff(Hz[:,:,1:-1],axis=1)*Cy - np.diff(Hy[:,1:-1,:],axis=2)*Cz);
    Ey[1:-1,:,1:-1] += (Dt /eps0) * (np.diff(Hx[1:-1,:,:],axis=2)*Cz - np.diff(Hz[:,:,1:-1],axis=0)*Cx);
    Ez[1:-1,1:-1,:] += (Dt /eps0) * (np.diff(Hy[:,1:-1,:],axis=0)*Cx - np.diff(Hx[1:-1,:,:],axis=1)*Cy);
    Et[t,:] = np.array([Ex[3,3,3], Ey[3,3,3], Ez[3,3,3]]);
