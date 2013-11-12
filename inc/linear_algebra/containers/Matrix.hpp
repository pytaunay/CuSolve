#ifndef _MATRIX_HPP_
#define _MATRIX_HPP_

#ifdef USE_GPU
#include <cuda.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#endif

#include <thrust/host_vector.h>

// The matrices are represented in a flat array fashion. They are row major.

namespace LinearAlgebra {
	namespace Containers {

		template<class T>
		class Matrix {

			protected:
				// CPU Holder
				thrust::host_vector<T> hM; 	// CPU Holder		
				int nbCol;			// Number of columns
				int nbRow;			// Number of rows
				
				// GPU data
				#ifdef USE_GPU
				thrust::device_ptr<T> dpM;	//  Thrust device ptr to dM for Thrust and CUSP integration
				T *dM;				//  Raw GPU pointer
				#endif
			
			public:
				// Constructor
				Matrix(int nbRow, int nbCol) {
					hM.resize(nbRow*nbCol);
					this->nbCol = nbCol;
					this->nbRow = nbRow;

					#ifdef USE_GPU
					cudaMalloc((void**)&dM,nbCol*nbRow*sizeof(T));
					dpM = thrust::device_pointer_cast(dM);
					#endif
				}	
				// Destructor
				~Matrix() {
					#ifdef USE_GPU
					cudaFree(dM);
					#endif
					hM.clear();
					nbCol = 0;
					nbRow = 0;
				}	

				//// GPU Methods
				// Getter and setters for CPU and GPU 
				#ifdef USE_GPU
				__host__ __device__ T operator[][](int I, int J) const {
					#ifdef __CUDA_ARCH__
						return dM[J+I*nbCol];
					#else
						return hV[J+I*nbCol];
					#endif	
				}	
				__host__ __device__ T & operator[][](int I, int J) const {
					#ifdef __CUDA_ARCH__
						return dM[J+I*nbCol];
					#else
						return hV[J+I*nbCol];
					#endif
				}
				// Copy data
				void copyFromDevice() {
					thrust::copy_n(dpM,nbCol*nbRow,hM.begin());
				}
				void copyToDevice() {
					thrust::copy(hM.begin(), hM.end(), dpM);
				}
				#else
				// Getters and setters if CPU only
				T operator[][](int I,int J) const {
					return hV[J+I*nbCol];
				}	
				T & operator[](int N) {
					return hV[N];
				}	
				#endif

				const int getNbRow() {
					return nbRow;
				}
				const int getNbCol() {
					return nbCol;
				}	
			};	
	}	
}

#endif
