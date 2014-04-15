/**
 * @file nonlinearsolver.h
 * @author Pierre-Yves Taunay (py.taunay@psu.edu)
 * @date November, 2013
 * @brief Abstract class representation of a non linear solver. All non linear solvers inherit from this class.
 *
 */

#pragma once

#include <numerical_solvers/numericalsolver.h>

//// CuSolve
// Equation system
#include <equation_system/systemjacobian.h>
#include <equation_system/systemfunctional.h>
#include <equation_system/coojacobian.h>

using namespace System;

namespace NumericalSolver {
	template<typename T>
	class NonLinearSolver : public Solver {
		

		public:
			virtual void compute(
				const SystemFunctional<T> &F,
				const cooJacobian<T> &J,
				cusp::array1d<T,cusp::device_memory> &Fv,
				cusp::coo_matrix<int,T,cusp::device_memory> &Jv,
				cusp::array1d<T,cusp::device_memory> &d,
				cusp::array1d<T,cusp::device_memory> &Y
				) = 0;

	};		
}	

