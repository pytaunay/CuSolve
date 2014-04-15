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
	class cooJacobian : virtual public SystemJacobian<T> 
	{
		protected:
			// Host
			std::vector<int> idxI;
			std::vector<int> idxJ;
			
			// Inherited attributes
		//	std::vector<int> terms;	
		//	std::vector<T> kInds;
		//	std::vector<T>  constants;
		//	std::vector< std::map<T,T> > jFull;
		//	int maxElements;
		//	int nbElem;

			// Device wrappers
		//	EvalNode<T> *d_jNodes;
		//	int *d_jTerms;
		//	int *d_jOffsetTerms;	

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
			__host__ virtual void evaluate(
					cusp::coo_matrix<int,T,cusp::device_memory> &J,
					const cusp::array1d<T,cusp::device_memory> &Y,
					const cusp::array1d<T,cusp::device_memory> &d_kData) const;

//			__host__ virtual void setConstants(const T gamma);		
			
			/*! \brief Get terms
			 *
			 *
			 */
			__host__ std::vector<int> const & getTerms() const {
				return this->terms;
			}	

			__host__ std::vector<int> const & getIdxI() const {
				return this->idxI;
			}	
			__host__ std::vector<int> const & getIdxJ() const {
				return this->idxJ;
			}	
			__host__ std::vector<T> const & getkInds() const {
				return this->kInds;
			}	
			__host__ std::vector<T> const & getConstants() const {
				return this->constants;
			}	
			__host__ std::vector< std::map<T,T> > const & getjFull() const {
				return this->jFull;
			}	
			__host__ int const & getMaxElements() const {
				return this->maxElements;
			}	
			__host__ int const & getnbElem() const {
				return this->nbElem;
			}	
			__host__ __device__ EvalNode<T>* const & getNodes() const {
				return this->d_jNodes;
			}	
	};		
	/*! \brief Kernel  for the Jacobian evaluation
	 *
	 *
	 */
	template<typename T>
	__global__ void k_JacobianEvaluate(
					T *d_Jp,
					const EvalNode<T> *d_jNodes,
					const int *d_jTerms,
					const int *d_jOffsetTerms,
					T const* __restrict__ d_kp,
					T const* __restrict__ d_yp
					);
} // end of namespace System

#include <equation_system/detail/coojacobian.inl>
	
