/**
 * @file systemjacobian.h
 * @author Pierre-Yves Taunay (py.taunay@psu.edu)
 * @date November, 2013
 * @brief Class representation of the Jacobian of a system of equations
 *
 */

#pragma once

//// STD
#include <vector>
#include <map>

//// CuSolve 
#include <equation_system/evalnode.h>

namespace System {

	template<typename T> 
	class SystemJacobian {
		protected:
			// Host
			std::vector<int> terms;	
			std::vector<T> kInds;
			std::vector<T>  constants;
			std::vector< std::map<T,T> > jFull;

			int maxElements;

			// Device wrappers
			EvalNode<T> *d_jNodes;
			int *d_jTerms;
			int *d_jOffsetTerms;	
			
		/*
		public:
			__host__ virtual void evaluate() = 0;
			__global__ void k_evaluate();
		*/
			
	};
} // End namespace System	




			
