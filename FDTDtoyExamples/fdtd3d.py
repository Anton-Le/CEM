#!/bin/env python3

import numpy as np
import scipy as sp
from matplotlib import pyplot as plt


class YeeGrid:
        def __init__(self, Nx, Ny, Nz):
                '''
                Constructor. Builds a grid of Nx x Ny x Nz
                Yee cells.
                '''
                self.Nx = Nx;
                self.Ny = Ny;
                self.Nz = Nz;
                # define a virtual grid, which extends beyond the actual limits
                self.Nx_virt = Nx+1
                self.Ny_virt = Ny+1
                self.Nz_virt = Nz+1
                #create the grids
                self.fldEx = np.zeros( (self.Nx_virt, self.Ny_virt, self.Nz_virt) )
                self.fldEy = np.zeros( (self.Nx_virt, self.Ny_virt, self.Nz_virt) )
                self.fldEz = np.zeros( (self.Nx_virt, self.Ny_virt, self.Nz_virt) )
                self.fldHx = np.zeros( (self.Nx_virt, self.Ny_virt, self.Nz_virt) )
                self.fldHy = np.zeros( (self.Nx_virt, self.Ny_virt, self.Nz_virt) )
                self.fldHz = np.zeros( (self.Nx_virt, self.Ny_virt, self.Nz_virt) )
        def Ex(self, xIdx, yIdx, zIdx):
                return self.fldEx[xIdx,yIdx,zIdx]
        def Ey(self, xIdx, yIdx, zIdx):
                return self.fldEy[xIdx,yIdx,zIdx]
        def Ez(self, xIdx, yIdx, zIdx):
                return self.fldEz[xIdx,yIdx,zIdx]
        def Hx(self, xIdx, yIdx, zIdx):
                return self.fldHx[xIdx,yIdx,zIdx]
        def Hy(self, xIdx, yIdx, zIdx):
                return self.fldHy[xIdx,yIdx,zIdx]
        def Hz(self, xIdx, yIdx, zIdx):
                return self.fldHz[xIdx,yIdx,zIdx]
        def initRandom(self):
                self.fldEx[:-1,1:-1,1:-1] = np.random.randn(self.Nx, self.Ny - 1, self.Nz - 1)
                self.fldEy[1:-1,:-1,1:-1] = np.random.randn(self.Nx - 1, self.Ny, self.Nz - 1)
                self.fldEz[1:-1,1:-1,:-1] = np.random.randn(self.Nx - 1, self.Ny - 1, self.Nz)
        def updateMatlabStyle(self, eps:float, mu:float, dt:float, cellDim:tuple):
            dx, dy, dz = cellDim
            self.fldHx[:,:-1,:-1] = self.fldHx[:,:-1,:-1] + (dt/mu)*(np.diff(self.fldEy[:,:-1,:],axis=2)/dz - np.diff(self.fldEz[:,:,:-1],axis=1)/dy);
            self.fldHy[:-1,:,:-1] = self.fldHy[:-1,:,:-1] + (dt/mu)*(np.diff(self.fldEz[:,:,:-1],axis=0)/dx - np.diff(self.fldEx[:-1,:,:],axis=2)/dz);
            self.fldHz[:-1,:-1,:] = self.fldHz[:-1,:-1,:] + (dt/mu)*(np.diff(self.fldEx[:-1,:,:],axis=1)/dy - np.diff(self.fldEy[:,:-1,:],axis=0)/dx);
            # update E fields
            self.fldEx[:-1,1:-1,1:-1] = self.fldEx[:-1,1:-1,1:-1] + (dt /eps) * (np.diff(self.fldHz[:-1,:-1,1:-1],axis=1)/dy - np.diff(self.fldHy[:-1,1:-1,:-1],axis=2)/dz);
            self.fldEy[1:-1,:-1,1:-1] = self.fldEy[1:-1,:-1,1:-1] + (dt /eps) * (np.diff(self.fldHx[1:-1,:-1,:-1],axis=2)/dz - np.diff(self.fldHz[:-1,:-1,1:-1],axis=0)/dx);
            self.fldEz[1:-1,1:-1,:-1] = self.fldEz[1:-1,1:-1,:-1] + (dt /eps) * (np.diff(self.fldHy[:-1,1:-1,:-1],axis=0)/dx - np.diff(self.fldHx[1:-1,:-1,:-1],axis=1)/dy);

        def updateOwn(self, eps:float, mu:float, dt:float, cellDim:tuple):
            '''
            Function used to update the fields
            '''
            dx, dy,dz = cellDim
            #First we update the magnetic field
            # X component
            for idxX in range(self.Nx_virt):
                for idxY in range(self.Ny):
                    for idxZ in range(self.Nz):
                        #update the X component of H
                        self.fldHx[idxX, idxY, idxZ] = self.fldHx[idxX, idxY, idxZ] + dt /mu *(( self.fldEy[idxX,idxY, idxZ+1] -self.fldEy[idxX, idxY, idxZ] ) / dz\
                                    -  ( self.fldEz[idxX, idxY+1, idxZ] - self.fldEz[idxX, idxY, idxZ  ]  )/dy );
            # Y component
            for idxX in range(self.Nx):
                for idxY in range(self.Ny_virt):
                    for idxZ in range(self.Nz):
                        #update the X component of H
                        self.fldHy[idxX, idxY, idxZ] = self.fldHy[idxX, idxY, idxZ]+ dt /(mu * dx) *( self.fldEz[idxX+1,idxY, idxZ] -self.fldEz[idxX, idxY, idxZ] )\
                                    - dt /(mu * dz) * ( self.fldEx[idxX, idxY, idxZ+1] - self.fldEx[idxX, idxY, idxZ]  );
            # Z component
            for idxX in range(self.Nx):
                for idxY in range(self.Ny):
                    for idxZ in range(self.Nz_virt):
                        #update the X component of H
                        self.fldHz[idxX, idxY, idxZ] =  self.fldHz[idxX, idxY, idxZ] + dt /(mu * dy) *( self.fldEx[idxX,idxY+1, idxZ] -self.fldEx[idxX, idxY, idxZ] )\
                                    - dt /(mu * dx) * ( self.fldEy[idxX+1, idxY, idxZ] - self.fldEy[idxX, idxY, idxZ]  );
            #########################################
            # Electric field
            # X component
            for idxX in range(self.Nx):
                for idxY in range(1,self.Ny):
                    for idxZ in range(1, self.Nz):
                        #update the X component of H
                        self.fldEx[idxX, idxY, idxZ] = self.fldEx[idxX, idxY, idxZ] - dt /(eps * dz) *( self.fldHy[idxX, idxY, idxZ] -self.fldHy[idxX, idxY, idxZ-1] )\
                                    + dt /(eps * dy) * ( self.fldHz[idxX, idxY, idxZ] - self.fldHz[idxX, idxY-1, idxZ]  );
            # Y component
            for idxX in range(1, self.Nx):
                for idxY in range(self.Ny):
                    for idxZ in range(1, self.Nz):
                        #update the X component of H
                        self.fldEy[idxX, idxY, idxZ] = self.fldEy[idxX, idxY, idxZ] - dt /(eps * dx) *( self.fldHz[idxX,idxY, idxZ] -self.fldHz[idxX-1, idxY, idxZ] )\
                                    + dt /(eps * dz) * ( self.fldHx[idxX, idxY, idxZ] - self.fldHx[idxX, idxY, idxZ-1]  );
            # Z component
            for idxX in range(1, self.Nx):
                for idxY in range(1,self.Ny):
                    for idxZ in range(self.Nz):
                        #update the X component of H
                        self.fldEz[idxX, idxY, idxZ] = self.fldEz[idxX, idxY, idxZ] - dt /(eps * dy) *( self.fldHx[idxX,idxY, idxZ] -self.fldHx[idxX, idxY-1, idxZ] )\
                                    + dt /(eps * dx) * ( self.fldHy[idxX, idxY, idxZ] - self.fldHy[idxX-1, idxY, idxZ]  );

class UnifiedYeeGrid:
        def __init__(self, Nx, Ny, Nz, pmlNx=0, pmlNy=0, pmlNz=0, dx=1.0, dy=1.0, dz=1.0):
                '''
                Constructor. Builds a grid of Nx x Ny x Nz
                Yee cells.
                with a PML boundary on each of the 6 faces of the domain (halo grid).

                '''
                self.Nx = Nx;
                self.Ny = Ny;
                self.Nz = Nz;
                self.pmlNx = pmlNx
                self.pmlNy = pmlNy
                slef.pmlNz = pmlNz;
                # define a virtual grid, which extends beyond the actual limits
                self.Nx_virt = Nx+1 + 2*pmlNx
                self.Ny_virt = Ny+1 + 2*pmlNy
                self.Nz_virt = Nz+1 + 2*pmlNz
                #create the grids
                self.fldE = np.zeros( (3, self.Nx_virt, self.Ny_virt, self.Nz_virt) )
                self.fldH = np.zeros( (3, self.Nx_virt, self.Ny_virt, self.Nz_virt) )
                # Define the coordinates of the (0,0,0) cell
                # implicitly we assme that the box extends in z direction with its xy-face
                # being located at z=0
                self.originCellCoordinates = (-dx * Nx/2, -dy * Ny / 2,0.)
                # Create the auxiliary arrays to store the PML constants


        def Ex(self, xIdx, yIdx, zIdx):
                return self.fldE[0, xIdx,yIdx,zIdx]
        def Ey(self, xIdx, yIdx, zIdx):
                return self.fldE[1, xIdx,yIdx,zIdx]
        def Ez(self, xIdx, yIdx, zIdx):
                return self.fldE[2, xIdx,yIdx,zIdx]
        def Hx(self, xIdx, yIdx, zIdx):
                return self.fldH[0, xIdx,yIdx,zIdx]
        def Hy(self, xIdx, yIdx, zIdx):
                return self.fldH[1, xIdx,yIdx,zIdx]
        def Hz(self, xIdx, yIdx, zIdx):
                return self.fldH[2, xIdx,yIdx,zIdx]
        def initRandom(self):
                self.fldE[0, :-1,1:-1,1:-1] = np.random.randn(self.Nx, self.Ny - 1, self.Nz - 1)
                self.fldE[1, 1:-1,:-1,1:-1] = np.random.randn(self.Nx - 1, self.Ny, self.Nz - 1)
                self.fldE[2, 1:-1,1:-1,:-1] = np.random.randn(self.Nx - 1, self.Ny - 1, self.Nz)
        def setGridPositions(x0:float, y0:float, z0:float):
                '''
                This function is used to define the position of the (0,0,0) cell in a global coordinate
                system. From this the coordinates of all cells can be determined
                '''
                self.originCellCoordinates(x0, y0, z0);
                return None;
        def updateMatlabStyle(self, eps:float, mu:float, dt:float, cellDim:tuple):
            dx, dy, dz = cellDim
            self.fldH[0, :,:-1,:-1] = self.fldH[0,:,:-1,:-1] + (dt/mu)*(np.diff(self.fldE[1,:,:-1,:],axis=2)/dz - np.diff(self.fldE[2,:,:,:-1],axis=1)/dy);
            self.fldH[1, :-1,:,:-1] = self.fldH[1,:-1,:,:-1] + (dt/mu)*(np.diff(self.fldE[2,:,:,:-1],axis=0)/dx - np.diff(self.fldE[0,:-1,:,:],axis=2)/dz);
            self.fldH[2, :-1,:-1,:] = self.fldH[2,:-1,:-1,:] + (dt/mu)*(np.diff(self.fldE[0,:-1,:,:],axis=1)/dy - np.diff(self.fldE[1,:,:-1,:],axis=0)/dx);
            # update E fields
            self.fldE[0, :-1,1:-1,1:-1] = self.fldE[0,:-1,1:-1,1:-1] + (dt /eps) * (np.diff(self.fldH[2,:-1,:-1,1:-1],axis=1)/dy - np.diff(self.fldH[1,:-1,1:-1,:-1],axis=2)/dz);
            self.fldE[1, 1:-1,:-1,1:-1] = self.fldE[1,1:-1,:-1,1:-1] + (dt /eps) * (np.diff(self.fldH[0,1:-1,:-1,:-1],axis=2)/dz - np.diff(self.fldH[2,:-1,:-1,1:-1],axis=0)/dx);
            self.fldE[2, 1:-1,1:-1,:-1] = self.fldE[2,1:-1,1:-1,:-1] + (dt /eps) * (np.diff(self.fldH[1,:-1,1:-1,:-1],axis=0)/dx - np.diff(self.fldH[0,1:-1,:-1,:-1],axis=1)/dy);

        def updateOwn(self, eps:float, mu:float, dt:float, cellDim:tuple):
            '''
            Function used to update the fields
            '''
            dx, dy,dz = cellDim
            #First we update the magnetic field
            # X component
            for idxX in range(self.Nx_virt):
                for idxY in range(self.Ny):
                    for idxZ in range(self.Nz):
                        #update the X component of H
                        self.fldH[0,idxX, idxY, idxZ] = self.fldH[0,idxX, idxY, idxZ] + dt /mu *(( self.fldE[1,idxX,idxY, idxZ+1] -self.fldE[1,idxX, idxY, idxZ] ) / dz\
                                    -  ( self.fldE[2,idxX, idxY+1, idxZ] - self.fldE[2,idxX, idxY, idxZ  ]  )/dy );
            # Y component
            for idxX in range(self.Nx):
                for idxY in range(self.Ny_virt):
                    for idxZ in range(self.Nz):
                        #update the X component of H
                        self.fldH[1,idxX, idxY, idxZ] = self.fldH[1,idxX, idxY, idxZ]+ dt /(mu * dx) *( self.fldE[2,idxX+1,idxY, idxZ] -self.fldE[2,idxX, idxY, idxZ] )\
                                    - dt /(mu * dz) * ( self.fldE[0,idxX, idxY, idxZ+1] - self.fldE[0,idxX, idxY, idxZ]  );
            # Z component
            for idxX in range(self.Nx):
                for idxY in range(self.Ny):
                    for idxZ in range(self.Nz_virt):
                        #update the X component of H
                        self.fldH[2,idxX, idxY, idxZ] =  self.fldH[2,idxX, idxY, idxZ] + dt /(mu * dy) *( self.fldE[0,idxX,idxY+1, idxZ] -self.fldE[0,idxX, idxY, idxZ] )\
                                    - dt /(mu * dx) * ( self.fldE[1,idxX+1, idxY, idxZ] - self.fldE[1,idxX, idxY, idxZ]  );
            #########################################
            # Electric field
            # X component
            for idxX in range(self.Nx):
                for idxY in range(1,self.Ny):
                    for idxZ in range(1, self.Nz):
                        #update the X component of H
                        self.fldE[0,idxX, idxY, idxZ] = self.fldE[0,idxX, idxY, idxZ] - dt /(eps * dz) *( self.fldH[1,idxX, idxY, idxZ] -self.fldH[1,idxX, idxY, idxZ-1] )\
                                    + dt /(eps * dy) * ( self.fldH[2,idxX, idxY, idxZ] - self.fldH[2,idxX, idxY-1, idxZ]  );
            # Y component
            for idxX in range(1, self.Nx):
                for idxY in range(self.Ny):
                    for idxZ in range(1, self.Nz):
                        #update the X component of H
                        self.fldE[1,idxX, idxY, idxZ] = self.fldE[1,idxX, idxY, idxZ] - dt /(eps * dx) *( self.fldH[2,idxX,idxY, idxZ] -self.fldH[2,idxX-1, idxY, idxZ] )\
                                    + dt /(eps * dz) * ( self.fldH[0,idxX, idxY, idxZ] - self.fldH[0,idxX, idxY, idxZ-1]  );
            # Z component
            for idxX in range(1, self.Nx):
                for idxY in range(1,self.Ny):
                    for idxZ in range(self.Nz):
                        #update the X component of H
                        self.fldE[2,idxX, idxY, idxZ] = self.fldE[2,idxX, idxY, idxZ] - dt /(eps * dy) *( self.fldH[0,idxX,idxY, idxZ] -self.fldH[0,idxX, idxY-1, idxZ] )\
                                    + dt /(eps * dx) * ( self.fldH[1,idxX, idxY, idxZ] - self.fldH[1,idxX-1, idxY, idxZ]  );
