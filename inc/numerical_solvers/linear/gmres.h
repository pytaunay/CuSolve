/**
 * @file gmres.h
 * @author Pierre-Yves Taunay (py.taunay@psu.edu)
 * @date November, 2013
 * @brief Class representation of a GMRES solver
 *
 */
#pragma once

//// CUDA
#include <cuda.h>

//// CUSP
#include <cusp/format.h>
#include <cusp/detail/matrix_base.h>
#include <cusp/array1d.h>
#include <cusp/coo_matrix.h>

//// CuSolve
#include <numerical_solvers/linear/linearsolver.h>

namespace cusolve {
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

				/**
				 * \brief Constructor
				 *
				 * Default constructor. Sets the total number of iterations to 5,000, the number of iterations before restart to 1,000, and both the absolute and relative tolerance to \f$ 10^{-8} \f$.
				 *
				 */
				GMRES(); 	
				
				/**
				 * \brief Constructor
				 *
				 * This constructor sets the total number of iterations to 5,000, the number of iterations before restart to 1,000, and both the absolute and relative tolerance to specified values. 
				 *
				 */

				GMRES(T _relTol, T _absTol);	

				/**
				 * \brief Compute method
				 *
				 * Wrapper for the CUSP GMRES method, for COO matrices.
				 *
				 */
				void compute(
						cusp::coo_matrix<int,T,cusp::device_memory> &A,
						cusp::array1d<T,cusp::device_memory> &b,
						cusp::array1d<T,cusp::device_memory> &x);

				/**
				 * \brief Compute method
				 *
				 * Wrapper for the CUSP GMRES method, for CSR matrices.
				 *
				 */
				void compute(
						const cusp::csr_matrix<int,T,cusp::device_memory> &A,
						const cusp::array1d<T,cusp::device_memory> &b,
						cusp::array1d<T,cusp::device_memory> &x);

				/**
				 * \brief Compute method
				 *
				 * Wrapper for the CUSP GMRES method, for dense matrices.
				 *
				 */
				void compute(
						const cusp::detail::matrix_base<int,T,cusp::device_memory,cusp::dense_format> &A,
						const cusp::array1d<T,cusp::device_memory> &b,
						cusp::array1d<T,cusp::device_memory> &x);
		};		
	}
}

#include <numerical_solvers/linear/detail/gmres.inl>
