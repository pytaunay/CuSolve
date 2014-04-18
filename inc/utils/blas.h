/**
 * @file blas.h
 * @author Pierre-Yves Taunay (py.taunay@psu.edu)
 * @date April 2014
 * @brief Utilities
 *
 */

#pragma once

#include <cublas_v2.h>

namespace utils {
	// AXPY wrapper
	namespace { 		
	template<typename T>
	__host__ void axpyWrapper(cublasHandle_t &handle,
			int n,
			const T *alpha,
			const T *X,int incx,
			T *Y, int incy) {
				std::cout << "ERROR: axpy only implemented for double and float types" << std::endl;
			}	

	template<>
	__host__ void axpyWrapper<float>(cublasHandle_t &handle,
			int n,
			const float *alpha,
			const float *X,int incx,
			float *Y, int incy) {
				cublasStatus_t stat;
				stat = cublasSaxpy(handle,n,alpha,X,incx,Y,incy);

				if( stat != CUBLAS_STATUS_SUCCESS) {
					std::cout << "AXPY failed !" << std::endl;
				}	
			}	

	template<>
	__host__ void axpyWrapper<double>(cublasHandle_t &handle,
			int n,
			const double *alpha,
			const double *X,int incx,
			double *Y, int incy) {
				cublasStatus_t stat;

				stat = cublasDaxpy(handle,n,alpha,X,incx,Y,incy);

				if( stat != CUBLAS_STATUS_SUCCESS) {
					std::cout << "AXPY failed ! Error status: " << stat << std::endl;
				}	
			}	
	}
}

//#include <utils/blas.inl>
