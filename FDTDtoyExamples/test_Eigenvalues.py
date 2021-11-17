import numpy as np
from fdtd3d import YeeGrid, UnifiedYeeGrid

eps0 = 8.8541878e-12;
mu0 = 4e-7 * np.pi;
c0= 299792458;


import h5py


if __name__=='__main__':
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
    Dt = 1/(c0*np.linalg.norm([Cx, Cy, Cz]))
    # Load reference data from Octave
    datafile=h5py.File("Exarray.h5")
    data = datafile['Ex']['value'][:,:,:]
    Ex = data.transpose()
    datafile.close()

    datafile=h5py.File("Eyarray.h5")
    data = datafile['Ey']['value'][:,:,:]
    Ey = data.transpose()
    datafile.close()

    datafile=h5py.File("Ezarray.h5")
    data = datafile['Ez']['value'][:,:,:]
    Ez = data.transpose()
    datafile.close()

    Et = np.zeros((Nt,3));
    grid = UnifiedYeeGrid(Nx, Ny, Nz, pmlCells=(0,0,0), cellDim=(1./Cx, 1./Cy, 1./Cz) )
    # initialize
    #grid.initRandom()
    grid.fldE[0,:-1,:,:] = Ex[:,:,:]
    grid.fldE[1,:,:-1,:] = Ey[:,:,:]
    grid.fldE[2,:,:,:-1] = Ez[:,:,:]
    #iterate
    for t in range(Nt):
            grid.updateOwn(eps0, mu0, Dt, (1./Cx, 1./Cy, 1./Cz) );
            Et[t, :] = np.array([ grid.Ex(3,3,3), grid.Ey(3,3,3), grid.Ez(3,3,3) ])

    datafile=h5py.File("Etfile.h5")
    data = datafile['Et']['value'][:,:]
    EtReference = data.transpose()
    datafile.close()
    Diff2 = Et - EtReference
    print(Diff2)
