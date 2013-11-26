/**
 * @file systemjacobian.h
 * @author Pierre-Yves Taunay (py.taunay@psu.edu)
 * @date November, 2013
 * @brief Class representation of the Jacobian of a system of equations
 *
 */
#pragma once

#include <vector>
#include <map>


#include <cusolve.h>

using namespace LinearAlgebra::Containers;

namespace System {

	template<typename T, template<class> class MType> 
	class SystemJacobian {
		protected:
			// Host
			std::vector<int> terms;	
			std::vector<T> kInds;
			std::vector<T>  constants;
			std::vector<map<T,T> > jFull;

			int maxElements;

			// Device wrappers
			EvalNode<T> *d_jNodes;
			int *d_jTerms;
			int *d_jOffsetTerms;	
			
		public:




	};

	// Partial specialization for COOMatrices
	template<typename T>
	class SparseSystemJacobian <T,COOMatrix> : public SystemJacobian <T,COOMatrix> {
		protected:
			// Host
			std::vector<int> idxI;
			std::vector<int> idxJ;
		
		public:


		



	};

} // End namespace System	




			
