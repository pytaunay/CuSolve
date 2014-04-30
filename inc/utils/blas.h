/**
 * @file blas.h
 * @author Pierre-Yves Taunay (py.taunay@psu.edu)
 * @date April 2014
 * @brief Utilities
 *
 */

#pragma once

#include <iostream>
#include <cublas_v2.h>

namespace cusolve {
namespace utils {
// AXPY wrapper
namespace { 		
	/**
	 * \brief Wrapper for CUBLAS AXPY
	 *
	 * \param [in] handle CUBLAS handle
	 * \param [in] n Total number of elements in each vector
	 * \param [in] alpha Pointer to the scalar value in AXPY
	 * \param [in] X Vector X
	 * \param [in] incx Total stride between elements in X
	 * \param [in] Y Vector Y
	 * \param [in] incy Total stride between elements in Y
	 */
	template<typename T>
	__host__ void axpy_wrapper(
			cublasHandle_t &handle,
			int n,
			const T *alpha,
			const T *X,int incx,
			T *Y, int incy) {
				std::cout << "ERROR GPU AXPY only implemented for double and float types" << std::endl;
			}	

	template<>
	__host__ void axpy_wrapper<float>(cublasHandle_t &handle,
			int n,
			const float *alpha,
			const float *X,int incx,
			float *Y, int incy) {
				cublasStatus_t stat;
				stat = cublasSaxpy(handle,n,alpha,X,incx,Y,incy);

				if( stat != CUBLAS_STATUS_SUCCESS) {
					std::cout << "ERROR GPU AXPY failed ! Error status: " << stat << std::endl;
				}	
			}	

	template<>
	__host__ void axpy_wrapper<double>(cublasHandle_t &handle,
			int n,
			const double *alpha,
			const double *X,int incx,
			double *Y, int incy) {
				cublasStatus_t stat;

				stat = cublasDaxpy(handle,n,alpha,X,incx,Y,incy);

				if( stat != CUBLAS_STATUS_SUCCESS) {
					std::cout << "ERROR GPU AXPY failed ! Error status: " << stat << std::endl;
				}	
			}	
} // Anonymous namespace
} // utils namespace
} // cusolve namespace
