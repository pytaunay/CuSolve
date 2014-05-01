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
#include <equation_system/eval_node.h>

namespace cusolve {
	/*!\class system_jacobian "system_jacobian.h" "inc/equation_system/system_jacobian.h"
	 *
	 * \brief Interface for the analytical representation of the Jacobian of a system.
	 *
	 * Each equation in the Jacobian is represented as an ensemble of "nodes", each node being the representation of the following term:
	 * \f[
	 * 	c_{i}\cdot k_{j} \cdot Y_{k}^a \cdot Y_{l}^b.
	 * \f]	
	 *
	 * \tparam T Single / Double precision
	 */
	template<typename T> 
	class system_jacobian {
		protected:
			// Host
			std::vector<int> terms;	 
			std::vector<T> kInds; /*!< Index for the k parameter in each node encountered */
			std::vector<T>  constants; /*!< Constants for each node encountered */ 
			std::vector< std::map<T,T> > jFull; /*!< Nodal representation, on the CPU */

			int maxElements;
			int nbElem; /*!< Number of stored elements in the matrix */
			
			dim3 blocks_j; /*!< Grid configuration for kernel launches */
			dim3 threads_j; /*!< Block configuration for kernel launches */

			// Device wrappers
			eval_node<T> *d_jNodes; /*!< Array of nodal representation of a simple equation term c*k*y1^e1*y2^e2 */
			int *d_jTerms; /*!< Total number of nodes for each equation */
			int *d_jOffsetTerms; /*!< Since the total number of nodes is not constant, we store also the offset in the array d_jNodes to find where the nodes for a given equation start */
			
		private:
			/**
			 * \brief set_grid method
			 *
			 * The set_grid methods determines the grid size and block size for a kernel launch.
			 * Virtual gives the flexibility of reimplementing the method for different matrix representations.
			 *
			 */
			__host__ virtual void set_grid();

		public:	

			/**
			 * \brief Default constructor
			 *
			 * Empty.
			 */
			system_jacobian() {}

			/**
			 * \brief Destructor
			 *
			 * Pure virtual to render the class abstract. An implementation of the destructor is, however, provided, as it
			 * provides a default behavior when destructors of children classes are invoked. 
			 */
			virtual ~system_jacobian() = 0;

			/**
			 * \brief Getter for terms member
			 */
			__host__ std::vector<int> const & get_terms() const {
				return this->terms;
			}	
			/**
			 * \brief Getter for kInds member
			 *
			 */
			__host__ std::vector<T> const & get_k_inds() const {
				return this->kInds;
			}	
			/**
			 * \brief Getter for constants member
			 *
			 */
			__host__ std::vector<T> const & get_constants() const {
				return this->constants;
			}	
			/**
			 * \brief Getter for jFull member
			 *
			 */
			__host__ std::vector< std::map<T,T> > const & get_jfull() const {
				return this->jFull;
			}	
			/**
			 * \brief Getter for maxElements member
			 *
			 */
			__host__ int const & get_max_elements() const {
				return this->maxElements;
			}	
			/**
			 * \brief Getter for nbElem member
			 *
			 */
			__host__ int const & get_nb_elem() const {
				return this->nbElem;
			}	
			/**
			 * \brief Getter for d_jNodes member
			 *
			 */
			__host__ __device__ eval_node<T>* const & get_jnodes() const {
				return this->d_jNodes;
			}	
			/**
			 * \brief Getter for d_jTerms member
			 *
			 */
			__host__ __device__ int* const & get_jterms() const {
				return this->d_jTerms;
			}	
			/**
			 * \brief Getter for d_jOffsetTerms member
			 *
			 */
			__host__ __device__ int* const & get_joffset_terms() const {
				return this->d_jOffsetTerms;
			}	

	};
} // End namespace cusolve 
