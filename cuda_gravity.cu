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
		writeId = id_w + t_i*blockDim.x*gridDim.x + t_k*blockDim.x*gridDim.x*blockDim.y*gridDim.y;
		writeBound( 1, phi,	bound_l, t_j, t_i, t_k, writeId );
	if ( t_j==( N_W - 1) )
		id_w = N_W - 2;
		writeId = id_w + t_i*blockDim.x*gridDim.x + t_k*blockDim.x*gridDim.x*blockDim.y*gridDim.y;
		writeBound( 1, phi, bound_r, t_j, t_i, t_k, writeId );
	if ( t_i==0 )
		id_w = 1;
		writeId = t_j + id_w*blockDim.x*gridDim.x + t_k*blockDim.x*gridDim.x*blockDim.y*gridDim.y;
		writeBound( 2, phi, bound_d, t_j, t_i, t_k, writeId );
	if ( t_i==( N_H - 1 ) )
		id_w = N_H - 2;
		writeId = t_j + id_w*blockDim.x*gridDim.x + t_k*blockDim.x*gridDim.x*blockDim.y*gridDim.y;
		writeBound( 2, phi,	bound_u, t_j, t_i, t_k, writeId );
	if ( t_k==0 )
		id_w = 1;
		writeId = t_j + t_i*blockDim.x*gridDim.x + id_w*blockDim.x*gridDim.x*blockDim.y*gridDim.y;
		writeBound( 3, phi, bound_b, t_j, t_i, t_k, writeId );
	if ( t_k==( N_D -1 ) )
		id_w = N_D -2;
		writeId = t_j + t_i*blockDim.x*gridDim.x + id_w*blockDim.x*gridDim.x*blockDim.y*gridDim.y;
		writeBound( 3, phi,	bound_t, t_j, t_i, t_k, writeId );
}

__global__ void iterPoissonStep(  const int paridad,
				 const int nX, const int nY, const int nZ,
         const cudaP Dx, const cudaP Dy, const cudaP Dz,
         const cudaP Drho, const cudaP dx2,
				 const cudaP omega, const cudaP pi4,
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
  int tid = t_j + t_i*nX + t_k*nX*blockDim.y*gridDim.y;

  cudaP rho, phi_c, phi_l, phi_r, phi_d, phi_u, phi_b, phi_t, phi_new;
	rho = rho_all[ tid ];
	phi_c = phi_all[tid];

	if ( t_j==0 )
		phi_l = loadBound( 1, bound_l, t_j, t_i, t_k );
	else
		phi_l = phi_all[ (t_j-1) + t_i*nX + t_k*nX*blockDim.y*gridDim.y ];

	if ( t_j==( N_W - 1) )
		phi_r = loadBound( 1, bound_r, t_j, t_i, t_k );

	if ( t_i==0 )
		phi_d = loadBound( 2, bound_d, t_j, t_i, t_k );

	if ( t_i==( N_H - 1 ) )
		phi_u = loadBound( 2, bound_u, t_j, t_i, t_k );

	if ( t_k==0 )
		phi_b = loadBound( 3, bound_b, t_j, t_i, t_k );

	if ( t_k==( N_D -1 ) )
		phi_t = loadBound( 3, bound_t, t_j, t_i, t_k );


	// //Set neighbors ids
	// int  l_indx, r_indx, d_indx, u_indx, b_indx, t_indx;
	// l_indx = t_j==0    ?    1 : t_j-1;  //Left
	// r_indx = t_j==nX-1 ? nX-2 : t_j+1;  //Right
	// d_indx = t_i==0    ?    1 : t_i-1;  //Down
	// u_indx = t_i==nY-1 ? nY-2 : t_i+1;  //Up
	// b_indx = t_k==0    ?    1 : t_k-1;  //bottom
	// t_indx = t_k==nZ-1 ? nZ-2 : t_k+1;  //top

	// phi_l = phi_all[ l_indx + t_i*nX + t_k*nX*blockDim.y*gridDim.y ];
	// phi_r = phi_all[ r_indx + t_i*nX + t_k*nX*blockDim.y*gridDim.y ];
	// phi_d = phi_all[ t_j + d_indx*nX + t_k*nX*blockDim.y*gridDim.y ];
	// phi_u = phi_all[ t_j + u_indx*nX + t_k*nX*blockDim.y*gridDim.y ];
	// phi_b = phi_all[ t_j + t_i*nX + b_indx*nX*blockDim.y*gridDim.y ];
	// phi_t = phi_all[ t_j + t_i*nX + t_indx*nX*blockDim.y*gridDim.y ];

  // phi_new = (1-omega)*phi_c + omega/6*( phi_l + phi_r + phi_d + phi_u + phi_b + phi_t - dx2*rho );
	phi_new = (1-omega)*phi_c + omega*( Dx*( phi_r + phi_l ) + Dy*( phi_d + phi_u) + Dz*( phi_b + phi_t ) - Drho*rho );
	phi_all[ tid ] = phi_new;

  if ( ( abs( ( phi_new - phi_c ) / phi_c ) > 0.001 ) ) converged[0] = 0;
}

__global__ void FFT_divideK2_kernel( cudaP *kxfft, cudaP *kyfft, cudaP *kzfft,
	cudaP *data_re, cudaP *data_im){
		int t_j = blockIdx.x*blockDim.x + threadIdx.x;
		int t_i = blockIdx.y*blockDim.y + threadIdx.y;
		int t_k = blockIdx.z*blockDim.z + threadIdx.z;
		int tid = t_j + t_i*blockDim.x*gridDim.x + t_k*blockDim.x*gridDim.x*blockDim.y*gridDim.y;

		cudaP kx = kxfft[t_j];
		cudaP ky = kyfft[t_i];
		cudaP kz = kzfft[t_k];
		cudaP k2 = kx*kx + ky*ky + kz*kz;
		data_re[tid] = -data_re[tid]/k2;
		data_im[tid] = -data_im[tid]/k2;
	}
