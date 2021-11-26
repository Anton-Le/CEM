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
        def __init__(self, Nx, Ny, Nz, pmlCells=(0,0,0), cellDim=(1.0,1.0,1.0) ):
                '''
                Constructor. Builds a grid of Nx x Ny x Nz
                Yee cells.
                with a PML boundary on each of the 6 faces of the domain (halo grid).

                '''
                self.Nx = Nx;
                self.Ny = Ny;
                self.Nz = Nz;
                # define a virtual grid, which extends beyond the actual limits
                self.pmlCells=pmlCells
                self.Nx_virt = Nx+1 + 2*pmlCells[0]
                self.Ny_virt = Ny+1 + 2*pmlCells[1]
                self.Nz_virt = Nz+1 + 2*pmlCells[2]
                #create the grids
                self.fldE = np.zeros( (3, self.Nx_virt, self.Ny_virt, self.Nz_virt) )
                self.fldH = np.zeros( (3, self.Nx_virt, self.Ny_virt, self.Nz_virt) )
                # Create the separate coefficient grid for PMLs
                # Said grid consists of 2*pmlNx * 2*pmlNy * 2*pmlNz cells for the 8 corners,
                # 
                self.CoeffC = np.zeros( (3, 2*pmlCells[0] + pmlCells[0] ) )
                # Define the coordinates of the (0,0,0) cell
                # implicitly we assme that the box extends in z direction with its xy-face
                # being located at z=0
                self.dx, self.dy, self.dz = cellDim
                # Create the auxiliary arrays to store the PML constants
                self.originCellCoordinates = (-self.dx * self.Nx/2, -self.dy * self.Ny / 2, -self.dz * self.Nz / 2)
        def setCellOrigin( x0, y0, z0  ):
                self.originCellCoordinates = (x0, y0, z0);

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
                #initialize only the interior parts
                pmlX, pmlY, pmlZ = self.pmlCells
                self.fldE[0, pmlX:-1-pmlX,1+pmlY:-1-pmlY,1+pmlZ:-1-pmlZ] = np.random.randn(self.Nx, self.Ny - 1, self.Nz - 1)
                self.fldE[1, 1+pmlX:-1-pmlX,pmlY:-1-pmlY,1+pmlZ:-1-pmlZ] = np.random.randn(self.Nx - 1, self.Ny, self.Nz - 1)
                self.fldE[2, 1+pmlX:-1-pmlX,1+pmlY:-1-pmlY,pmlZ:-1-pmlZ] = np.random.randn(self.Nx - 1, self.Ny - 1, self.Nz)

        def initE(self, initFunc):
            '''
            The function initializes the internal electric field array using a user-provided function `initFunc` which takes 
            physical coordinates (x,y,z) as arguments and reutrns the electric field vector at the given position.
            **NOTE**: Due to the shifted nature of the components.
            x component: [1:-1, 1:-1, 1:-1] + [0, 1:-1, 1:-1]
            y component: [1:-1, 1:-1, 1:-1] + [1:-1, 0, 1:-1]
            z component: [1:-1, 1:-1, 1:-1] + [1:-1, 1:-1, 0]
            '''
            # Array centre
            for idxX in range(1, self.Nx):
                for idxY in range(1, self.Ny):
                    for idxZ in range(1, self.Nz):
                        # x component
                        fldValueX = initFunc(self.originCellCoordinates[0] + self.dx * (idxX + 0.5),
                                            self.originCellCoordinates[1] + self.dy * idxY,
                                            self.originCellCoordinates[2] + self.dz * idxZ )
                        # y component
                        fldValueY = initFunc(self.originCellCoordinates[0] + self.dx * idxX,
                                            self.originCellCoordinates[1] + self.dy * (idxY + 0.5),
                                            self.originCellCoordinates[2] + self.dz * idxZ)
                        # z component
                        fldValueZ = initFunc(self.originCellCoordinates[0] + self.dx * idxX,
                                            self.originCellCoordinates[1] + self.dy * idxY,
                                            self.originCellCoordinates[2] + self.dz * (idxZ+0.5) )
                        self.fldE[:,idxX, idxY, idxZ] = np.array( [fldValueX[0], fldValueY[1], fldValueZ[2]] )
                        
            # iterate over the boundaries
            for idxY in range(1, self.Ny):
                for idxZ in range(1, self.Nz):
                    self.fldE[0, 0, idxY, idxZ] = initFunc(self.dx * 0.5 + self.originCellCoordinates[0],
                                                           self.dy * (0.5 + idxY)+ self.originCellCoordinates[1],
                                                           self.dz *(0.5 + idxZ)+ self.originCellCoordinates[2])[0]
            # 
            for idxX in range(1, self.Nx):
                for idxZ in range(1, self.Nz):
                    self.fldE[1, idxX, 0, idxZ] = initFunc(self.dx * (0.5 + idxX) + self.originCellCoordinates[0],
                                                           self.dy * 0.5+ self.originCellCoordinates[1],
                                                           self.dz *(0.5+idxZ) + self.originCellCoordinates[2])[1]
            for idxX in range(1, self.Nx):
                for idxY in range(1, self.Ny):
                    self.fldE[2, idxX, idxY, 0] = initFunc(self.dx * (0.5 + idxX) + self.originCellCoordinates[0],
                                                           self.dy * (0.5 + idxY)+ self.originCellCoordinates[1],
                                                           self.dz * 0.5 + self.originCellCoordinates[2])[2]

        def updateMatlabStyle(self, eps:float, mu:float, dt:float, cellDim:tuple):
            dx, dy, dz = cellDim
            self.fldH[0, :,:-1,:-1] = self.fldH[0,:,:-1,:-1] + (dt/mu)*(np.diff(self.fldE[1,:,:-1,:],axis=2)/dz - np.diff(self.fldE[2,:,:,:-1],axis=1)/dy);
            self.fldH[1, :-1,:,:-1] = self.fldH[1,:-1,:,:-1] + (dt/mu)*(np.diff(self.fldE[2,:,:,:-1],axis=0)/dx - np.diff(self.fldE[0,:-1,:,:],axis=2)/dz);
            self.fldH[2, :-1,:-1,:] = self.fldH[2,:-1,:-1,:] + (dt/mu)*(np.diff(self.fldE[0,:-1,:,:],axis=1)/dy - np.diff(self.fldE[1,:,:-1,:],axis=0)/dx);
            # update E fields
            self.fldE[0, :-1,1:-1,1:-1] = self.fldE[0,:-1,1:-1,1:-1] + (dt /eps) * (np.diff(self.fldH[2,:-1,:-1,1:-1],axis=1)/dy - np.diff(self.fldH[1,:-1,1:-1,:-1],axis=2)/dz);
            self.fldE[1, 1:-1,:-1,1:-1] = self.fldE[1,1:-1,:-1,1:-1] + (dt /eps) * (np.diff(self.fldH[0,1:-1,:-1,:-1],axis=2)/dz - np.diff(self.fldH[2,:-1,:-1,1:-1],axis=0)/dx);
            self.fldE[2, 1:-1,1:-1,:-1] = self.fldE[2,1:-1,1:-1,:-1] + (dt /eps) * (np.diff(self.fldH[1,:-1,1:-1,:-1],axis=0)/dx - np.diff(self.fldH[0,1:-1,:-1,:-1],axis=1)/dy);

        def updateOwn(self, eps:float, mu:float, dt:float, cellDim:tuple = None):
            '''
            Function used to update the fields
            '''
            if cellDim is not None:
                dx ,dy ,dz = cellDim
            else:
                dx, dy,dz = (self.dx, self.dy, self.dz)

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
