/**
 * @file systemfunctional.h
 * @author Pierre-Yves Taunay (py.taunay@psu.edu)
 * @date November, 2013
 * @brief Class representation of the functional of a system of equations
 *
 */
#pragma once

#include <clientp.hpp>

#include <vector>
#include <map>
#include <string>

#include <cusolve.h>
#include <evalnode.h>

using namespace LinearAlgebra::Containers;

namespace System {

	template<typename T>
	class SystemFunctional {
		protected:
			// Host 
			//std::vector<T> kInds; /*!< k indices for equations */ 
			Vector<T> kInds;
			std::vector<T> constants; /*!< all constant and sign data */
			std::vector<map<T,T>> yFull; /*!< all y data (yindex(key) & power(value)) */
			std::vector<int> terms; /*!< Number of terms per equation evaluation */

			int maxElements; /*!< Maximum number of terms in the equation evaluation */
			int maxTermSize; /*!< Maximum number of elements encountered in equation term, for node size */

			// Device wrappers 
			EvalNode *d_fNodes;
			int *d_fTerms;
			int *d_fOffsetTerms;

		public:
			/*!\brief Constructor with filename
			 *
			 *
			 * @param[in] filename location of the k data
			 */ 
			SystemFunctional(string filename); 

			/*!\brief Evaluation of the system functional, based on the data stored in the device memory
			 *
			 *
			 * @param[in] Y vector at which we want to evaluate the functional
			 */ 
			__host__ void eval(const Vector<T> Y);

		private:
			/*!\brief Kernel for the evaluation of the system functional
			 *
			 */ 
			__global__ void k_eval(void); 
	};
} // end of System	

#include <detail/systemfunctional.inl>
