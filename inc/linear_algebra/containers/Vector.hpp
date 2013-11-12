#ifndef _VECTOR_HPP_
#define _VECTOR_HPP_

#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>

#include <thrust/host_vector.h>

#include "Vector.hpp"

using namespace thrust;

namespace LinearAlgebra {
	namespace Containers {
		template<class T>
		class Vector {
			protected: 
				thrust::host_vector<T> hV; // CPU holder
				int N; // Size
				T *dV; 	// Raw pointer
				thrust::device_ptr<T> dpV; // Thrust device_ptr to dV; allows for using thrust and cusp functions


			public:
				// Constructor
				Vector(int N) {	

					hV.resize(N);
					this->N = N;

					cudaMalloc((void**)&dV,N*sizeof(T));	
					dpV = thrust::device_pointer_cast(dV); 
				}
				
				// Destructor
				~Vector() {

					cudaFree(dV);
					hV.clear();
					N = 0;
				}

				//// GPU Methods
				// Getter for GPU and CPU 
				__host__ __device__ T operator[](int N) const {
					#ifdef __CUDA_ARCH__
						return dV[N];
					#else
						return hV[N];
					#endif	
				}	
				// Setter for GPU and CPU 
				__host__ __device__ T & operator[](int N) {
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

				const int size() {
					return N;
				}	
		};
	}
}	
#endif

