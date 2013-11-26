/**
 * @file linearsolver.h
 * @author Pierre-Yves Taunay (py.taunay@psu.edu)
 * @date November, 2013
 * @brief Abstract linear solver class. All linear solvers inherit from this class.
 *
 */
#pragma once

namespace NumericalSolver {
	class LinearSolver {
		public:
			virtual void compute() = 0;
	};	
}

