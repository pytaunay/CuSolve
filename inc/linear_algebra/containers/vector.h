#ifndef _VECTOR_HPP_
#define _VECTOR_HPP_

#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>

#include <thrust/host_vector.h>

#include "Vector.cuh"

using namespace thrust;

namespace LinearAlgebra {
	namespace Containers {

		/*! \brief Wrapper for a vector stored on the host and device side
		 *
		 *
		 */ 
		template<class T>
		class Vector {
			protected: 
				int N; /*!< Size of the vector */

			public:	
				thrust::host_vector<T> hV; /*! < CPU holder*/
				T *dpV; 	/*!< Raw device pointer */
				cusp::array1d<T,cusp::device_memory> dV; /*! < CUSP device vector; allows for using thrust functionality */

			public:
				// Constructor
				Vector(int N) {	

					hV.resize(N);
					this->N = N;

					dV.resize(N);
					dpV = thrust::raw_pointer_cast(&dV[0]); 
				}

				Vector() {
					this->N = 0;
					dpV = NULL;
				}	


				// Destructor
				~Vector() {

					cudaFree(dV);
					hV.clear();
					N = 0;
				}

				//// GPU Methods
				/ Getter for GPU and CPU 
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
					thrust::copy(dV.begin(),dV.end(),hV.begin() );
				}	

				void copyToDevice() {
					thrust::copy(hV.begin(), hV.end(), dV.begin() );
				}	
		};
	}
}	
#endif

