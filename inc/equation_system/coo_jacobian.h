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

//// cusolve
#include <equation_system/system_jacobian.h>
#include <equation_system/system_functional.h>

namespace cusolve {
	
	/*!\class coo_jacobian coo_jacobian.h "inc/equation_system/coo_jacobian.h" 
	 * 
	 *
	 *
	 * \tparam T Type 
	 */
	template<typename T>
	class coo_jacobian : virtual public system_jacobian<T> 
	{
		protected:
			// Host
			std::vector<int> idxI; /*!< Row indices of non-zero elements (COO representation) */
			std::vector<int> idxJ; /*!< Column indices of non-zero elements (COO representation) */
			
			// Inherited attributes
		//	std::vector<int> terms;	
		//	std::vector<T> kInds;
		//	std::vector<T>  constants;
		//	std::vector< std::map<T,T> > jFull;
		//	int maxElements;
		//	int nbElem;

		//	dim3 blocks_j, threads_j;

			// Device wrappers
		//	eval_nodes<T> *d_jNodes;
		//	int *d_jTerms;
		//	int *d_jOffsetTerms;	

		public:	
			/*! \brief Default constructor
			 *
			 * Empty.
			 */
			coo_jacobian();

			/*! \brief Constructor based on a given functional
			 *
			 * Calculates the analytical Jacobian \f$ J \f$:
			 * \f[
			 * 	J = \partial F / \partial Y 
			 * \f]
			 * The constructor uses members of a system functional to populate all the members of the Jacobian representation
			 * \param[in] F A system functional
			 */
			coo_jacobian(const system_functional<T> &F);

			/*! \brief Destructor
			 *
			 * Resets vectors idxI and idxJ. The rest of deallocation is done through the base class destructor 
			 */
			~coo_jacobian();

			/*! \brief Host wrapper for the kernel call to evaluate the Jacobian
			 *
			 *
			 * Simple host wrapper that sets up all pointers and texture memory in order to numerically evaluate the Jacobian of a system of equations.
			 * The function is virtual so that it can be reimplemented in the BDF COO Jacobian and others.
			 *
			 * @param[in] Y the evaluation point
			 * @param[out] J the numerical evaluation of the matrix
			 *
			 */
			__host__ virtual void evaluate(
					cusp::coo_matrix<int,T,cusp::device_memory> &J,
					const cusp::array1d<T,cusp::device_memory> &Y,
					const cusp::array1d<T,cusp::device_memory> &d_kData) const;
			
			/*! \brief Getter for the idxI member
			 *
			 */
			__host__ std::vector<int> const & get_idx_i() const {
				return this->idxI;
			}	

			/*! \brief Getter for the idxJ member
			 *
			 */
			__host__ std::vector<int> const & get_idx_j() const {
				return this->idxJ;
			}	
	};		

	/*! \brief Kernel for the Jacobian evaluation
	 *
	 *
	 */
	template<typename T>
	__global__ void k_jacobian_evaluate(
					T *d_Jp,
					const EvalNode<T> *d_jNodes,
					const int *d_jTerms,
					const int *d_jOffsetTerms,
					T const* __restrict__ d_kp,
					T const* __restrict__ d_yp
					);
} // end of namespace cusolve 

#include <equation_system/detail/coo_jacobian.inl>
