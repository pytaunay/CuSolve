/**
 * @file newtonraphson.h
 * @author Pierre-Yves Taunay (py.taunay@psu.edu)
 * @date November, 2013
 * @brief Class representation of a Newton-Raphson solver
 *
 */

#pragma once

//// CUSP
#include <cusp/detail/matrix_base.h>
#include <cusp/array1d.h>
#include <cusp/format.h>

//// CuSolve
// Equation system
#include <equation_system/systemjacobian.h>
#include <equation_system/systemfunctional.h>

#include <equation_system/coojacobian.h>

// Numerical solvers
#include <numerical_solvers/nonlinear/nonlinearsolver.h>
#include <numerical_solvers/linear/linearsolver.h>


#include <iostream>

using namespace System;

namespace NumericalSolver {
	/*!\class NewtonRaphson newtonraphson.h "inc/numerical_solvers/nonlinear/newtonraphson.h"
	 * \brief NewtonRaphson Class
	 *
	 *
	 * Class wrapper for the non linear solver Newton-Raphson, which solves iteratively
	 * \f[
	 *	F\left( X \right ) = 0,
	 * \f]
	 * with \f$F:{\Re}^{N}\mapsto{\Re}^{N}\f$ and \f$X\in{\Re}^{N}\f$
	 * \tparam T float or double
	 * \tparam Format Matrix format for the Jacobian representation
	 */
	template<typename T>
	class NewtonRaphson : public NonLinearSolver<T> {

		protected:
			LinearSolver<T> *lsolve; /*!< Generic pointer to a linear solver for the Newton-Raphson method */
			int maxIter; /*!< Maximum number of iterations */ 
			T tol; /*!< Tolerance */

		public:
			/*!\brief Default constructor
			 * 
			 * 
			 * If nothing is specified, the maximum number of iterations is set to 50, the tolerance
			 * to 1e-6, and the linear solver to a simple LU direct
			 */
			NewtonRaphson(); 

			/*!\brief Constructor with max. number of iterations
			 *
			 *
			 * Just sets the maximum number of iterations. Assumes LU decomposition for the linear solver
			 * @param[in] maxIter the max. number of iterations
			 */
			NewtonRaphson(int maxIter);

			/*!\brief Full constructor setting all parameters
			 *
			 *
			 * @param[in] lsolve a pointer to a linear solver
			 */
			NewtonRaphson(LinearSolver<T> *lsolve, int maxIter, T tol);

			~NewtonRaphson();


			/*!\brief Compute method
			 *
			 *
			 * Perform Newton-Raphson iterations J*d = -f to solve the non linear equation \f$F\left( X \right ) = 0,\f$
			 * where \f$F:{\Re}^{N}\mapsto{\Re}^{N}\f$ and \f$X\in{\Re}^{N}\f$
			 * @param[in] F the system functional
			 * @param[in] J the system Jacobian
			 * @param[in] Fv an array to store the numerical evaluation of the system functional
			 * @param[in] Jv a matrix representation to store the numerical evaluation of the system Jacobian
			 * @param[in] d the result of J*d = -f
			 * @param[inout] Y the solution
			 */
			void compute(const SystemFunctional<T> &F,
				const cooJacobian<T> &J,
				cusp::array1d<T,cusp::device_memory> &Fv,
				cusp::coo_matrix<int,T,cusp::device_memory> &Jv,
				cusp::array1d<T,cusp::device_memory> &d,
				cusp::array1d<T,cusp::device_memory> &Y
				);
				
			

			/*
			void compute(
				const SystemFunctional<T> &F,
				const SystemJacobian<T> &J,
				cusp::array1d<T,cusp::device_memory> &Fv,
				cusp::detail::matrix_base<int,T,cusp::device_memory,cusp::known_format> &Jv,
				cusp::array1d<T,cusp::device_memory> &d,
				cusp::array1d<T,cusp::device_memory> &Y
				); 
				*/
	};
}	

#include <numerical_solvers/nonlinear/detail/newtonraphson.inl>

