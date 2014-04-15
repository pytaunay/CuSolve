/**
 * @file linearsolver.h
 * @author Pierre-Yves Taunay (py.taunay@psu.edu)
 * @date November, 2013
 * @brief Abstract linear solver class. All linear solvers inherit from this class.
 *
 */
#pragma once


#include <numerical_solvers/numericalsolver.h>


namespace NumericalSolver {
	
	template<typename T>
	class LinearSolver : public Solver {
		public:
			virtual void compute(
					cusp::coo_matrix<int,T,cusp::device_memory> &A,
					cusp::array1d<T,cusp::device_memory> &b,
					cusp::array1d<T,cusp::device_memory> &x) = 0;
	};	
}

