/**
 * @file gmres.inl
 * @author Pierre-Yves Taunay (py.taunay@psu.edu)
 * @date November, 2013
 * @brief Implementations of the methods for the class GMRES
 *
 */
#include <iostream>


#include <cusp/krylov/gmres.h>
#include <cusp/monitor.h>
#include <cusp/exception.h>
#include <cusp/format.h>
#include <cusp/detail/matrix_base.h>
#include <cusp/array1d.h>
#include <cusp/print.h>

//#include <cusp/precond/aggregation/smoothed_aggregation.h>
#include <cusp/precond/diagonal.h>
#include <cusp/precond/ainv.h>

#include <cusp/coo_matrix.h>

namespace NumericalSolver {
	
		template<typename T>
		void GMRES<T>::
			compute(
				cusp::coo_matrix<int,T,cusp::device_memory> &A,
				cusp::array1d<T,cusp::device_memory> &x,
				cusp::array1d<T,cusp::device_memory>  &b) {

			cusp::verbose_monitor<T> monitor(b,this->maxIter,this->relTol,this->absTol);
		//	cusp::default_monitor<T> monitor(b,this->maxIter,this->relTol,this->absTol);
		//	cusp::precond::aggregation::smoothed_aggregation<int, T, cusp::device_memory> M(A);
//		        cusp::precond::diagonal<T, cusp::device_memory> M(A);
//		        cusp::precond::scaled_bridson_ainv<T, cusp::device_memory> M(A, .1);

			std::cout << "INFO Starting GMRES..." << std::endl;
			cusp::krylov::gmres(A,x,b,this->restartIter,monitor);
//			cusp::krylov::gmres(A,x,b,this->restartIter,monitor,M);
		}	

		template<typename T>
		void GMRES<T>::
			compute(
				const cusp::detail::matrix_base<int,T,cusp::device_memory,cusp::dense_format> &A,
				const cusp::array1d<T,cusp::device_memory> &b,
				cusp::array1d<T,cusp::device_memory> &x) {

				throw cusp::not_implemented_exception("The GMRES method is not implemented for dense matrices");

			}	
}		



