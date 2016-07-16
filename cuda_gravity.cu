#include <pycuda-helpers.hpp>
#define N_W N_WIDTH
#define N_H N_HEIGHT
#define N_D N_DEPTH


__device__ double loadBound(  const int boundAxis,  double *bound,
								const int t_j, const int t_i, const int t_k ){
	int boundId;
	if ( boundAxis == 1 ) boundId = t_i + t_k*N_H;   //X BOUNDERIES
	if ( boundAxis == 2 ) boundId = t_j + t_k*N_W;   //Y BOUNDERIES
	if ( boundAxis == 3 ) boundId = t_j + t_i*N_W;   //Z BOUNDERIES
	return bound[boundId];
}

__device__ void writeBound(  const int boundAxis,
								double *phi, double *bound,
								const int t_j, const int t_i, const int t_k, const int writeId ){
	int boundId;
	if ( boundAxis == 1 ) boundId = t_i + t_k*N_H;   //X BOUNDERIES
	if ( boundAxis == 2 ) boundId = t_j + t_k*N_W;   //Y BOUNDERIES
	if ( boundAxis == 3 ) boundId = t_j + t_i*N_W;   //Z BOUNDERIES
	bound[boundId] = phi[writeId];
}

__global__ void setBounderies(
			 double* phi,
			 double* bound_l, double* bound_r, double* bound_d, double* bound_u, double* bound_b, double *bound_t ){
	int t_j = blockIdx.x*blockDim.x + threadIdx.x;
	int t_i = blockIdx.y*blockDim.y + threadIdx.y;
	int t_k = blockIdx.z*blockDim.z + threadIdx.z;
	// int tid = t_j + t_i*blockDim.x*gridDim.x + t_k*blockDim.x*gridDim.x*blockDim.y*gridDim.y;

	bool boundBlock = false;
	if ( blockIdx.x==0 || blockIdx.y==0 || blockIdx.z==0 ) boundBlock = true;
	if ( blockIdx.x==(gridDim.x-1) || blockIdx.y==(gridDim.y-1) || blockIdx.z==(gridDim.z-1) ) boundBlock = true;

	if ( !boundBlock ) return;

	int writeId, id_w ;

	if ( t_j==0 )
		id_w = 1;
		writeId = id_w + t_i*N_W + t_k*N_W*N_H;
		writeBound( 1, phi,	bound_l, t_j, t_i, t_k, writeId );
	if ( t_j==( N_W - 1) )
		id_w = N_W - 2;
		writeId = id_w + t_i*N_W + t_k*N_W*N_H;
		writeBound( 1, phi, bound_r, t_j, t_i, t_k, writeId );
	if ( t_i==0 )
		id_w = 1;
		writeId = t_j + id_w*N_W + t_k*N_W*N_H;
		writeBound( 2, phi, bound_d, t_j, t_i, t_k, writeId );
	if ( t_i==( N_H - 1 ) )
		id_w = N_H - 2;
		writeId = t_j + id_w*N_W + t_k*N_W*N_H;
		writeBound( 2, phi,	bound_u, t_j, t_i, t_k, writeId );
	if ( t_k==0 )
		id_w = 1;
		writeId = t_j + t_i*N_W + id_w*N_W*N_H;
		writeBound( 3, phi, bound_b, t_j, t_i, t_k, writeId );
	if ( t_k==( N_D -1 ) )
		id_w = N_D -2;
		writeId = t_j + t_i*N_W + id_w*N_W*N_H;
		writeBound( 3, phi,	bound_t, t_j, t_i, t_k, writeId );
}

__global__ void iterPoissonStep(  const int paridad,
         const double Dx, const double Dy, const double Dz,
         const double Drho, const double dx2,
				 const double omega, const double pi4,
				 double *rho_all, double *phi_all, int *converged,
				 double *bound_l, double *bound_r, double *bound_d, double *bound_u, double *bound_b, double *bound_t ){
  int t_j = 2*(blockIdx.x*blockDim.x + threadIdx.x);
  int t_i = blockIdx.y*blockDim.y + threadIdx.y;
  int t_k = blockIdx.z*blockDim.z + threadIdx.z;
  //Make a checkboard 3D grid
  if ( t_i%2 == 0 ){
    if ( t_k%2 == paridad ) t_j +=1;
  }
  else if ( (t_k+1)%2 == paridad ) t_j +=1;
  int tid = t_j + t_i*N_W + t_k*N_W*N_H;

  double rho, phi_c, phi_l, phi_r, phi_d, phi_u, phi_b, phi_t, phi_new;
	rho = rho_all[ tid ];
	phi_c = phi_all[tid];

	if ( t_j==0 )
		phi_l = loadBound( 1, bound_l, t_j, t_i, t_k );
	else
		phi_l = phi_all[ (t_j-1) + t_i*N_W + t_k*N_W*N_H ];

	if ( t_j==( N_W - 1) )
		phi_r = loadBound( 1, bound_r, t_j, t_i, t_k );
	else
		phi_r = phi_all[ (t_j+1) + t_i*N_W + t_k*N_W*N_H ];

	if ( t_i==0 )
		phi_d = loadBound( 2, bound_d, t_j, t_i, t_k );
	else
		phi_d = phi_all[ t_j + (t_i-1)*N_W + t_k*N_W*N_H ];

	if ( t_i==( N_H - 1 ) )
		phi_u = loadBound( 2, bound_u, t_j, t_i, t_k );
	else
		phi_u = phi_all[ t_j + (t_i+1)*N_W + t_k*N_W*N_H ];

	if ( t_k==0 )
		phi_b = loadBound( 3, bound_b, t_j, t_i, t_k );
	else
		phi_b = phi_all[ t_j + t_i*N_W + (t_k-1)*N_W*N_H ];

	if ( t_k==( N_D -1 ) )
		phi_t = loadBound( 3, bound_t, t_j, t_i, t_k );
	else
		phi_t = phi_all[ t_j + t_i*N_W + (t_k+1)*N_W*N_H ];

  // phi_new = (1-omega)*phi_c + omega/6*( phi_l + phi_r + phi_d + phi_u + phi_b + phi_t - dx2*rho );
	phi_new = (1-omega)*phi_c + omega*( Dx*( phi_r + phi_l ) + Dy*( phi_d + phi_u) + Dz*( phi_b + phi_t ) - Drho*rho );
	phi_all[ tid ] = phi_new;

  if ( ( abs( ( phi_new - phi_c ) / phi_c ) > 0.001 ) ) converged[0] = 0;
}

__global__ void FFT_divideK2_kernel( double *kxfft, double *kyfft, double *kzfft,
	double *data_re, double *data_im){
		int t_j = blockIdx.x*blockDim.x + threadIdx.x;
		int t_i = blockIdx.y*blockDim.y + threadIdx.y;
		int t_k = blockIdx.z*blockDim.z + threadIdx.z;
		int tid = t_j + t_i*blockDim.x*gridDim.x + t_k*blockDim.x*gridDim.x*blockDim.y*gridDim.y;

		double kx = kxfft[t_j];
		double ky = kyfft[t_i];
		double kz = kzfft[t_k];
		double k2 = kx*kx + ky*ky + kz*kz;
		data_re[tid] = -data_re[tid]/k2;
		data_im[tid] = -data_im[tid]/k2;
	}
