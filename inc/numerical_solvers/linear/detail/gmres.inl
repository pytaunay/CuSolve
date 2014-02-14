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

#include <cusp/coo_matrix.h>

namespace NumericalSolver {
	
		template<typename T>
		void GMRES<T>::
			compute(
				//const cusp::detail::matrix_base<int,T,cusp::device_memory,cusp::known_format> &A,
				cusp::coo_matrix<int,T,cusp::device_memory> &A,
				cusp::array1d<T,cusp::device_memory>  &b,
				cusp::array1d<T,cusp::device_memory> &x) {

			cusp::verbose_monitor<T> monitor(b,this->maxIter,this->relTol,this->absTol);
			std::cout << "INFO Starting GMRES..." << std::endl;
			cusp::krylov::gmres(A,x,b,this->restartIter,monitor);
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



