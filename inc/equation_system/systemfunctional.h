/**
 * @file systemfunctional.h
 * @author Pierre-Yves Taunay (py.taunay@psu.edu)
 * @date November, 2013
 * @brief Class representation of the functional of a system of equations
 *
 */
#pragma once


// STD
#include <vector>
#include <map>
#include <string>

// CUSP
#include <cusp/array1d.h>

// CuSolve
#include <equation_system/clientp.hpp>
#include <equation_system/evalnode.h>

namespace System {

	template<typename T>
	class SystemFunctional {
		protected:
			// Host 
			std::vector<T> h_kInds; /*!< k indices for equations */ 
			std::vector<T> constants; /*!< all constant and sign data */
			std::vector< map<T,T> > yFull; /*!< all y data (yindex(key) & power(value)) */
			std::vector<int> terms; /*!< Number of terms per equation evaluation */

			int maxElements; /*!< Maximum number of terms in the equation evaluation */
			int maxTermSize; /*!< Maximum number of elements encountered in equation term, for node size */

			// Device wrappers 
			EvalNode *d_fNodes;
			int *d_fTerms;
			int *d_fOffsetTerms;
			cusp::array1d<T,cusp::device_memory> d_kInds;

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
			 * @param[inout] F vector where the evaluation is stored
			 * @param[in] Y vector at which we want to evaluate the functional
			 */ 
			__host__ void evaluate(
					cusp::array1d<T,cusp::device_memory> &F,
					const cusp::array1d<T,cusp::device_memory> &Y);

		private:
			/*!\brief Kernel for the evaluation of the system functional
			 *
			 *
			 * @param[in] d_fp device function pointer, which is obtained from a raw pointer cast of a cusp array1d
			 */ 
			__global__ void k_evaluate(T *d_fp); 
	};
} // end of System	

#include <equation_system/detail/systemfunctional.inl>
