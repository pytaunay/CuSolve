/**
 * @file gmres.h
 * @author Pierre-Yves Taunay (py.taunay@psu.edu)
 * @date November, 2013
 * @brief Class representation of a GMRES solver
 *
 */
#pragma once

#include <cuda.h>

//// CUSP
#include <cusp/format.h>
#include <cusp/detail/matrix_base.h>
#include <cusp/array1d.h>
#include <cusp/coo_matrix.h>

//// Thrust

//// CuSolve
#include <numerical_solvers/linear/linearsolver.h>

namespace NumericalSolver {
	
	/*!\class GMRES gmres.h "inc/numerical_solvers/linear/gmres.h"
	 * \brief GMRES class
	 *
	 *
	 * Class wrapper for the GMRES solver, an iterative solver for the linear equation
	 * \f[
	 * A X = b
	 * \f]
	 * where \f$A\in{\Re}^{N\times N}\f$, \f$X\in{\Re}^{N}\f$, and \f$b\in{\Re}^{N}\f$
	 */
	template<typename T>
	class GMRES : public LinearSolver<T> {
		protected:
			int restartIter; 	/*!< Number of iterations before restart */ 	
			int maxIter;		/*!< Maximum number of iterations */
			T relTol;		/*!< Relative tolerance */
			T absTol;		/*!< Absolute tolerance */

		public:
			GMRES() {
				this->restartIter = 500;
				this->maxIter = 5000;
				this->relTol = (T)1e-8;
				this->absTol = (T)1e-8;
			}	

			GMRES(T _relTol, T _absTol) {
				this->restartIter = 500;
				this->maxIter = 5000;
				this->relTol = _relTol;
				this->absTol = _absTol;
			}	

			void compute(
				//	const cusp::detail::matrix_base<int,T,cusp::device_memory,cusp::known_format> &A,
					cusp::coo_matrix<int,T,cusp::device_memory> &A,
					cusp::array1d<T,cusp::device_memory> &b,
					cusp::array1d<T,cusp::device_memory> &x);
			void compute(
					const cusp::csr_matrix<int,T,cusp::device_memory> &A,
					const cusp::array1d<T,cusp::device_memory> &b,
					cusp::array1d<T,cusp::device_memory> &x);

			void compute(
					const cusp::detail::matrix_base<int,T,cusp::device_memory,cusp::dense_format> &A,
					const cusp::array1d<T,cusp::device_memory> &b,
					cusp::array1d<T,cusp::device_memory> &x);

			// Sparse matrix implementation
			/*
			void compute	(	SparseMatrix<float> &A, 
						cusp::array1d<float,cusp::device_memory> &b, 
						cusp::array1d<float,cusp::device_memory> &x
					); 

			void compute	(	SparseMatrix<double> &A, 
						cusp::array1d<double,cusp::device_memory> &b, 
						cusp::array1d<double,cusp::device_memory> &x
					); 

			// Full matrix implementation
			void compute	(	float *A, 
						float *b,
						float *x
					); 

			void compute	(	double *A, 
						double *b,
						double *x
					); 
			*/
	};		
}

#include <numerical_solvers/linear/detail/gmres.inl>
