/**
 * @file nonlinearsolver.h
 * @author Pierre-Yves Taunay (py.taunay@psu.edu)
 * @date November, 2013
 * @brief Abstract class representation of a non linear solver. All non linear solvers inherit from this class.
 *
 */

#pragma once

namespace NumericalSolver {
	class NonLinearSolver {
		public:
			virtual void compute() = 0;
	};		
}	

