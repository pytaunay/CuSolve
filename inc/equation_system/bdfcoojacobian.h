/**
 * @file bdfcoojacobian.h
 * @author Pierre-Yves Taunay (py.taunay@psu.edu)
 * @date April, 2014
 * @brief Derived class from COO Jacobian to add more functionality for the BDF solver
 *
 */

#pragma once

#include <equation_system/coojacobian.h>

namespace System {
	/*!\class BDFcooJacobian
	 *
	 *
	 *
	 * \tparam T type
	 */
	template<typename T>
	class BDFcooJacobian : public cooJacobian<T> {
		protected:
			// Inherited attributes
			// Host
		//	std::vector<int> idxI;
		//	std::vector<int> idxJ;
			
		//	std::vector<int> terms;	
		//	std::vector<T> kInds;
		//	std::vector<T>  constants;
		//	std::vector< std::map<T,T> > jFull;
		//	int maxElements;

			// Device wrappers
		//	EvalNode<T> *d_jNodes;
		//	int *d_jTerms;
		//	int *d_jOffsetTerms;	
			
			cusp::coo_matrix<int,T,cusp::device_memory> ID;

		public:
			BDFcooJacobian(const cooJacobian<T> &J,int nEq); 

 			__host__ void evaluate(
					cusp::coo_matrix<int,T,cusp::device_memory> &J,
					const cusp::array1d<T,cusp::device_memory> &Y,
					const cusp::array1d<T,cusp::device_memory> &d_kData) const;

			__host__ void setConstants(const T gamma);		
	};


	template<typename T>
	__global__ void
		k_BDFcooJacobianSetConstants(const T gamma, EvalNode<T> *d_jNodes, const int *d_fTerms, const int *d_fOffsetTerms, const int num_leaves);

} // end of namespace System

#include <equation_system/detail/bdfcoojacobian.inl>
