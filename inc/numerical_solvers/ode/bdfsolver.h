/**
 * @file bdfodesolver.h
 * @author Pierre-Yves Taunay (py.taunay@psu.edu)
 * @date March, 2014
 * @brief Class representation of a backwards differentiation formula solver
 *
 */

#pragma once

#include <numerical_solvers/ode/odesolver.h>

namespace NumericalSolver {
	
	template<typename T>
	class BDFsolver : public ImplicitODESolver, LMMODESolver {

		protected:
			NonLinearSolver<T> *nlsolve; /*!< Non linear solver */
			T tol; /*!< Tolerance */

			short q; /*!< BDF order */
			T dt; /*!< Current time step */
			T t; /*!< Current time */

			int nist; /*!< Number of internal steps taken */
			int neq; /*!< Number of independent SYSTEM of ODE */
			int N; /*!< Size of the system of ODE */

			// Stored on device
			// Nordsieck array -- should be invisible for user
			T *ZN;

			// L polynomial -- should be invisible for user
			T *Lpoly;


		public:
			BDFsolver();
			~BDFsolver();


			void compute();






	};
}

#include <numerical_solvers/ode/detail/bdfsolver.inl>
