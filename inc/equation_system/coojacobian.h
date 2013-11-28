/**
 * @file coojacobian.h
 * @author Pierre-Yves Taunay (py.taunay@psu.edu)
 * @date November, 2013
 * @brief Class representation of the Jacobian of a system of equations. Uses a COO matrix representation for the Jacobian.
 *
 */

#pragma once
//// STD
#include <vector>


//// CUDA
#include <cuda.h>

// CUSP
#include <cusp/array1d.h>
#include <cusp/coo_matrix.h>

//// CuSolve
#include <equation_system/systemjacobian.h>
#include <equation_system/systemfunctional.h>

namespace System {
	
	/*!\class COOJacobian  
	 * 
	 *
	 *
	 * \tparam T Type 
	 */
	template<typename T>
	class cooJacobian : public SystemJacobian<T> 
	{
		protected:
			// Host
			std::vector<int> idxI;
			std::vector<int> idxJ;

		public:	
			/*! \brief Default constructor
			 *
			 *
			 */
			cooJacobian();

			/*! \brief Constructor based on a given functional
			 *
			 *
			 * The constructor uses members of a system functional to populate all the members of the Jacobian representation
			 * @param[in] F a system functional
		         * \todo Add getters for systemFunctional to access k_inds, constants, y_complete, terms, from the functional
			 * \todo Add this-> for clarity on all members of Jacobian  
			 */
			cooJacobian(const SystemFunctional<T> &F);


			/*! \brief Host wrapper for the kernel call to evaluate the Jacobian
			 *
			 *
			 * Simple host wrapper that sets up all pointers and texture memory in order to numerically evaluate the Jacobian of a system of equations
			 * @param[in] Y the evaluation point
			 * @param[out] J the numerical evaluation of the matrix
			 *
			 * \todo Determine nthreads, nblocks
			 * \todo Fix textures 
			 */
			__host__ void evaluate(
					cusp::coo_matrix<T,cusp::device_memory> &J,
					const cusp::array1d<T,cusp::device_memory> &Y);

		private:
			/*! \brief Kernel  for the Jacobian evaluation
			 *
			 *
			 */
			__global__ void k_evaluate(void);

	};		
}	

#include <detail/coojacobian.inl>
	
