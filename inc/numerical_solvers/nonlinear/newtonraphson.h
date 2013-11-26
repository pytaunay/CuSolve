/**
 * @file newtonraphson.h
 * @author Pierre-Yves Taunay (py.taunay@psu.edu)
 * @date November, 2013
 * @brief Class representation of a Newton-Raphson solver
 *
 */

#pragma once

#include <cusolve.h>


namespace NumericalSolver {
	class NewtonRaphson : public NonLinearSolver {

		protected:
			LinearSolver *lsolve;
			int maxIter; 
			double tol;

		public:
			void compute();

	};
}	
