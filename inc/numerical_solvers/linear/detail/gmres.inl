/**
 * @file gmres.inl
 * @author Pierre-Yves Taunay (py.taunay@psu.edu)
 * @date November, 2013
 * @brief Implementations of the methods for the class GMRES
 *
 */
//// STD 
#include <iostream>

//// CUSP
// Matrix types
#include <cusp/format.h>
#include <cusp/array1d.h>
#include <cusp/coo_matrix.h>
#include <cusp/detail/matrix_base.h>

// GMRES
#include <cusp/krylov/gmres.h>

// Monitors
#include <cusp/monitor.h>

// Exceptions
#include <cusp/exception.h>

namespace cusolve {
	namespace NumericalSolver {

			template<typename T>
			GMRES<T>::
				GMRES() {
					this->restartIter = 1000;
					this->maxIter = 5000;
					this->relTol = (T)1e-8;
					this->absTol = (T)1e-8;
				}	

			template<typename T>
			GMRES<T>::
				GMRES(T _relTol, T _absTol) {
					this->restartIter = 1000;
					this->maxIter = 5000;
					this->relTol = _relTol;
					this->absTol = _absTol;
				}	
		
			template<typename T>
			void GMRES<T>::
				compute(
					cusp::coo_matrix<int,T,cusp::device_memory> &A,
					cusp::array1d<T,cusp::device_memory> &x,
					cusp::array1d<T,cusp::device_memory>  &b) {
				
				// Set up a verbose monitor
				cusp::verbose_monitor<T> monitor(b,this->maxIter,this->relTol,this->absTol);

				// Launch the method
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

			template<typename T>
			void GMRES<T>::
				compute(
					const cusp::csr_matrix<int,T,cusp::device_memory> &A,
					const cusp::array1d<T,cusp::device_memory> &b,
					cusp::array1d<T,cusp::device_memory> &x) {
					
					throw cusp::not_implemented_exception("The GMRES method is not implemented for CSR matrices");
				}
	}		
}
