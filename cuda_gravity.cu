#include <pycuda-helpers.hpp>

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

__global__ void iterPoissonStep(  const int paridad,
				 const int nX, const int nY, const int nZ,
         const cudaP dx2, const cudaP dy2, const cudaP dz2,
         const cudaP dAll, const cudaP omega, const cudaP pi4,
				 cudaP* rho_all, cudaP* phi_all, int* converged ){
  int t_j = 2*(blockIdx.x*blockDim.x + threadIdx.x);
  int t_i = blockIdx.y*blockDim.y + threadIdx.y;
  int t_k = blockIdx.z*blockDim.z + threadIdx.z;
  //Make a checkboard 3D grid
  if ( t_i%2 == 0 ){
    if ( t_k%2 == paridad ) t_j +=1;
  }
  else if ( (t_k+1)%2 == paridad ) t_j +=1;
  int tid = t_j + t_i*nX + t_k*nX*blockDim.y*gridDim.y;

	//Set neighbors ids
	int l_indx, r_indx, d_indx, u_indx, b_indx, t_indx;
	l_indx = t_j==0    ?    1 : t_j-1;  //Left
	r_indx = t_j==nX-1 ? nX-2 : t_j+1;  //Right
	d_indx = t_i==0    ?    1 : t_i-1;  //Down
	u_indx = t_i==nY-1 ? nY-2 : t_i+1;  //Up
	b_indx = t_k==0    ?    1 : t_k-1;  //bottom
	t_indx = t_k==nZ-1 ? nZ-2 : t_k+1;  //top

  cudaP rho, phi_c, phi_l, phi_r, phi_d, phi_u, phi_b, phi_t, phi_new;
	rho = rho_all[ tid ];
	phi_c = phi_all[tid];
	phi_l = phi_all[ l_indx + t_i*nX + t_k*nX*blockDim.y*gridDim.y ];
	phi_r = phi_all[ r_indx + t_i*nX + t_k*nX*blockDim.y*gridDim.y ];
	phi_d = phi_all[ t_j + d_indx*nX + t_k*nX*blockDim.y*gridDim.y ];
	phi_u = phi_all[ t_j + u_indx*nX + t_k*nX*blockDim.y*gridDim.y ];
	phi_b = phi_all[ t_j + t_i*nX + b_indx*nX*blockDim.y*gridDim.y ];
	phi_t = phi_all[ t_j + t_i*nX + t_indx*nX*blockDim.y*gridDim.y ];

  phi_new = (1-omega)*phi_c + omega/6*( phi_l + phi_r + phi_d + phi_u + phi_b + phi_t - dx2*pi4*rho );
	// phi_new = (1-omega)*phi_c + omega*( dx*( phi_r + phi_l ) + dy*( phi_d + phi_u) + dz*( phi_b + phi_t ) - dAll*pi4*rho );
	phi_all[ tid ] = phi_new;


  if ( ( abs( ( phi_new - phi_c ) / phi_c ) > 0.001 ) ) converged[0] = 0;


}
