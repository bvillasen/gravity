import sys, time, os
import numpy as np
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
# import pycuda.cumath as cumath
#import pycuda.curandom as curandom
from pycuda.elementwise import ElementwiseKernel
from pycuda.reduction import ReductionKernel
import h5py as h5
import matplotlib.pyplot as plt
from mpi4py import MPI
#Add Modules from other directories
currentDirectory = os.getcwd()
parentDirectory = currentDirectory[:currentDirectory.rfind("/")]
toolsDirectory = parentDirectory + "/tools"
volumeRenderDirectory = parentDirectory + "/volumeRender"
sys.path.extend( [toolsDirectory, volumeRenderDirectory] )
from cudaTools import *
from tools import ensureDirectory, printProgressTime

cudaP = "double"
nPoints = 128
useDevice = None
usingAnimation = False

for option in sys.argv:
  if option == "float": cudaP = "float"
  if option == "anim": usingAnimation = True
  if option == "mem": showKernelMemInfo = True
  if option == "128" or option == "256": nPoints = int(option)
  if option.find("dev=") != -1: useDevice = int(option[-1])
precision  = {"float":(np.float32, np.complex64), "double":(np.float64,np.complex128) }
cudaPre, cudaPreComplex = precision[cudaP]

#Initialize MPI
MPIcomm = MPI.COMM_WORLD
pId = MPIcomm.Get_rank()
nProcess = MPIcomm.Get_size()
name = MPI.Get_processor_name()

if pId == 0:
  print "\nMPI-CUDA 3D POISOON SOLVER"
  print " nProcess: {0}\n".format(nProcess)
MPIcomm.Barrier()




for i in range(nProcess):
  if pId == i:
    print "[pId {0}] Host: {1}".format( pId, name )
  MPIcomm.Barrier()



#set simulation volume dimentions
nWidth = nPoints
nHeight = nPoints
nDepth = nPoints
nData = nWidth*nHeight*nDepth

Lx = 2.
Ly = 2.
Lz = 1.
xMax, xMin = Lx/2, -Lx/2
yMax, yMin = Ly/2, -Ly/2
zMax, zMin = Lz/2, -Lz/2
dx, dy, dz = Lx/(nWidth-1), Ly/(nHeight-1), Lz/(nDepth-1 )
Z, Y, X = np.mgrid[ zMin:zMax:nDepth*1j, yMin:yMax:nHeight*1j, xMin:xMax:nWidth*1j ]
xPoints = X[0,0,:]
yPoints = Y[0,:,0]
zPoints = Z[0,0,:]
R = np.sqrt( X*X + Y*Y + Z*Z )
sphereR = 0.1
sphereOffCenter = 0.25
sphere = np.sqrt( X*X + Y*Y + Z*Z ) < sphereR
sphere_left  = ( np.sqrt( (X+sphereOffCenter)*(X+sphereOffCenter) + Y*Y + Z*Z ) < sphereR )
sphere_right = ( np.sqrt( (X-sphereOffCenter)*(X-sphereOffCenter) + Y*Y + Z*Z ) < sphereR )
spheres = sphere_right + sphere_left


#For analitical solution
sigma = 0.2
r2 = X*X + Y*Y
rho_teo = ( r2 - 2*sigma**2 )/sigma**4 * np.exp( -r2/(2*sigma**2) )
phi_teo = np.exp( -r2/(2*sigma**2) )


#Change precision of the parameters
dx, dy, dz = cudaPre(dx), cudaPre(dy), cudaPre(dz)
dx2, dy2, dz2 = dx*dx, dy*dy, dz*dz
D = 2*(dy2*dz2 + dx2*dz2 + dx2*dy2);
Dx = dy2*dz2/D
Dy = dx2*dz2/D
Dz = dx2*dy2/D
Drho = dx2*dy2*dz2/D
# Lx, Ly, Lz = cudaPre(Lx), cudaPre(Ly), cudaPre(Lz)
# xMin, yMin, zMin = cudaPre(xMin), cudaPre(yMin), cudaPre(zMin)
pi4 = cudaPre( 4*np.pi )



#initialize pyCUDA context
cudaCtx, cudaDev = mpi_setCudaDevice(pId, 0, MPIcomm, show=False)
#set thread grid for CUDA kernels
block_size_x, block_size_y, block_size_z = 32,4,4   #hardcoded, tune to your needs
gridx = nWidth // block_size_x + 1 * ( nWidth % block_size_x != 0 )
gridy = nHeight // block_size_y + 1 * ( nHeight % block_size_y != 0 )
gridz = nDepth // block_size_z + 1 * ( nDepth % block_size_z != 0 )
block3D = (block_size_x, block_size_y, block_size_z)
grid3D = (gridx, gridy, gridz)
nBlocks3D = grid3D[0]*grid3D[1]*grid3D[2]
grid3D_poisson = (gridx//2, gridy, gridz)


if pId == 0: print "\nCompiling CUDA code"
cudaCodeFile = open("cuda_gravity.cu","r")
cudaCodeString = cudaCodeFile.read().replace( "cudaP", cudaP )
cudaCode = SourceModule(cudaCodeString)
iterPoissonStep_kernel = cudaCode.get_function('iterPoissonStep')
########################################################################
convertToUCHAR = ElementwiseKernel(arguments="cudaP normaliztion, cudaP *values, unsigned char *psiUCHAR".replace("cudaP", cudaP),
			      operation = "psiUCHAR[i] = (unsigned char) ( -255*( values[i]*normaliztion -1 ) );",
			      name = "sendModuloToUCHAR_kernel")

def poisonIteration( parity, omega ):
  iterPoissonStep_kernel( np.int32(parity),
  np.int32( nWidth ), np.int32( nHeight ), np.int32( nDepth ),
  Dx, Dy, Dz, Drho, dx2, cudaPre(omega), pi4,
  rho_d, phi_d, converged, grid=grid3D_poisson, block=block3D  )

rJacobi = ( np.cos(np.pi/nWidth) + (dx/dy)**2*np.cos(np.pi/nHeight) ) / ( 1 + (dx/dy)**2 )
def poissonStep( omega ):
  converged.set( one_Array )
  poisonIteration( 0, omega )
  poisonIteration( 1, omega )
  hasConverged = converged.get()[0]
  return hasConverged

########################################################################
def solvePoisson( show=False ):
  maxIter = 500000
  omega = 2. / ( 1 + np.pi / nWidth  )
  # omega = 1
  for n in range(maxIter):
    hasConverged = poissonStep( omega )
    if hasConverged == 1:
      phi_1 = phi_d.get()
      poisonIteration( 0, omega )
      phi_2 = phi_d.get()
      phi_avrg = ( phi_1 + phi_2 )/2.
      if show: print 'Poisson converged: ', n+1
      # return phi_1, phi_2, phi_avrg
      return phi_avrg
  if show: print 'Poisson converged: ', maxIter
  return phi_d.get()

########################################################################
########################################################################
if pId == 0:
  print "\nInitializing Data"
  initialMemory = getFreeMemory( show=True )
rho = np.zeros( X.shape, dtype=cudaPre )  #density
#####################################################
#Initialize a centerd sphere
overDensity = spheres
rho[ overDensity ] = 1.
rho[ np.logical_not(overDensity) ] = 0.6
rho = rho_teo
# phi = np.ones( X.shape, dtype=cudaPre )   #gravity potencial
phi = rho   #gravity potencial
zeros_h = np.zeros_like( rho )
#####################################################
#Initialize device global data
phi_d = gpuarray.to_gpu( phi )
rho_d = gpuarray.to_gpu( rho )
rho_re_d = gpuarray.to_gpu( rho )
rho_im_d = gpuarray.to_gpu( zeros_h )
rho_FFT_re_d = gpuarray.to_gpu( zeros_h )
rho_FFT_im_d = gpuarray.to_gpu(zeros_h)
one_Array = np.array([ 1 ]).astype( np.int32 )
converged = gpuarray.to_gpu( one_Array )
if usingAnimation:
  plotData_d = gpuarray.to_gpu(np.zeros([nDepth, nHeight, nWidth], dtype = np.uint8))
  volumeRender.plotData_dArray, copyToScreenArray = gpuArray3DtocudaArray( plotData_d )
print "Total Global Memory Used: {0:.2f} MB\n".format(float(initialMemory-getFreeMemory( show=False ))/1e6)

print 'Getting initial Gravity Force...'
start, end = cuda.Event(), cuda.Event()
start.record() # start timing
phi = solvePoisson( show=True )
phi = phi - phi.min()
# print phi.max()
# phi = phi/phi.max()
end.record(), end.synchronize()
secs = start.time_till( end )*1e-3
print 'Time: {0:0.4f}\n'.format( secs )

phi_slide_teo = phi_teo[nDepth/2,:,:]
extent = [xMin, xMax, yMin, yMax]
plt.figure(0)
plt.clf()
plt.imshow(phi_slide_teo, extent=extent, interpolation='nearest')
plt.colorbar()
# plt.show()

phi_slide = phi[nDepth/2,:,:]
plt.figure(2)
plt.clf()
plt.imshow(phi_slide, extent=extent, interpolation='nearest')
plt.colorbar()
# plt.show()

plt.figure(3)
plt.clf()
plt.imshow(np.abs(phi_slide - phi_slide_teo), interpolation='nearest')
plt.colorbar()
plt.show()


######################################################################
#Clean and Finalize
MPIcomm.Barrier()
#Terminate CUDA
cudaCtx.pop()
cudaCtx.detach() #delete it
#Terminate MPI
MPIcomm.Barrier()
for i in range(nProcess):
  if pId == i: print "##########################################################END-{0}".format(pId)
  MPIcomm.Barrier()
MPI.Finalize()
