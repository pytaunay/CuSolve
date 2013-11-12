#ifndef _VECTOR_HPP_
#define _VECTOR_HPP_


#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>

#include "Vector.hpp"

using namespace thrust;

namespace LinearAlgebra {
	namespace Containers {
		template<class T>
		class Vector {
			protected: 
				#ifdef USE_GPU
				T *dV; 	// Raw pointer
				thrust::device_ptr<T> dpV; // Thrust device_ptr to dV; allows for using thrust and cusp functions

				#endif
				thrust::host_vector<T> hV; // CPU holder

				int N; // Size

			public:
				// Constructor
				Vector(int N) {	

					hV.resize(N);
					this->N = N;

					#ifdef USE_GPU
					cudaMalloc((void**)&dV,N*sizeof(T));	
					dpV = thrust::device_pointer_cast(dV); 
					#endif	
				}
				
				// Destructor
				~Vector() {

					#ifdef USE_GPU
					cudaFree(dV);
					#endif
					
					hV.clear();

					N = 0;
				}

				//// GPU Methods
				// Getter for GPU kernels
				#ifdef USE_GPU
				__inline__ __device__ T at_gpu(int N) {
					return dV[N]; 
				}

				// Setter for GPU kernels
				__host__ __device__ T operator[](int N) const {
					#ifdef __CUDA_ARCH__
						return dV[N];
					#else
						return hV[N];
					#endif	
				}	
				__inline__ __device__ T & operator[](int N) {
					#ifdef __CUDA_ARCH__
						return dV[N];
					#else
						return hV[N];
					#endif	
				}	

				// Copy data back and forth
				void copyFromDevice() {
					thrust::copy_n(dpV,N, hV.begin() );
				}	

				void copyToDevice() {
					thrust::copy(hV.begin(), hV.end(), dpV );
				}	

				#endif	

				const int size() {
					return N;
				}	
		};
	}
}	


#endif

