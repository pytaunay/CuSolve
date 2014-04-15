/** 
 * @file bdfjacobian.h
 * @author Pierre-Yves Taunay (py.taunay@psu.edu)
 * @date April, 2014
 * @brief Interface for the modified Jacobian classes necessary in the BDF solver
 */

#pragma once


namespace System {
	template<typename T>
	class BDFjacobian : virtual public SystemJacobian<T> {

		// Inherited attributes
	//	protected:
			// Host
	//		std::vector<int> terms;	
	//		std::vector<T> kInds;
	//		std::vector<T>  constants;
	//		std::vector< std::map<T,T> > jFull;

	//		int maxElements;
	//		int nbElem; /*!< Number of stored elements in the matrix */

			// Device wrappers
	//		EvalNode<T> *d_jNodes;
	//		int *d_jTerms;
	//		int *d_jOffsetTerms;	
	
		public:
			__host__ virtual void setConstants(const T gamma) = 0;
			__host__ virtual void resetConstants(const cooJacobian<T> &J) = 0;
	};
}	
	
