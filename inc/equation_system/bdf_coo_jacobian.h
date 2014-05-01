/**
 * @file bdfcoojacobian.h
 * @author Pierre-Yves Taunay (py.taunay@psu.edu)
 * @date April, 2014
 * @brief Derived class from COO Jacobian to add more functionality for the BDF solver
 *
 */

#pragma once

namespace cusolve {
	/*!\class bdf_coo_jacobian
	 *
	 * \brief COO Jacobian representation used in the BDF solver
	 *
	 * The BDF solver requires an additional COO representation since it has to set a necessary constant in the BDF method.
	 * The BDF solver solves multiple times the following equation
	 * \f[
	 * 	\left( I - \gamma J\right) \delta = -G,
	 * \f]	
	 * where \f$ J \f$ is the system Jacobian, and \f$ I \f$ is the identity matrix. The BDFCOO Jacobian is used to set the constant
	 * \f$ \gamma \f$ at each timesteps, and represents the term \f$\left( I -\gamma J\right)\f$.
	 *
	 * Virtual inheritance is used to resolve the correct methods to use at compile time.
	 *
	 * \tparam T type
	 */
	template<typename T>
	class bdf_coo_jacobian : virtual public coo_jacobian<T>, virtual public bdf_jacobian<T> {
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
		//	eval_node<T> *d_jNodes;
		//	int *d_jTerms;
		//	int *d_jOffsetTerms;	
			
			cusp::coo_matrix<int,T,cusp::device_memory> ID; /*!< COO representation of the identity matrix */

		public: 
			/**
			 * \brief Class destructor
			 *
			 * Empties vectors, deallocates GPU memory.
			 *
			 */
			~bdf_coo_jacobian();

			/** 
			 * \brief Constructor
			 *
			 * Constructor based on the total number of equations in the system, and an already created COO Jacobian.
			 * The constructors sets all inherited attributes to the input coo Jacobian, and initializes the member ID.
			 *
			 */
			bdf_coo_jacobian(const coo_jacobian<T> &J,const int &nEq); 


			/**
			 * \brief Evaluate method
			 *
			 * Evaluates the BDF Jacobian at point Y and parameters d_kData, and stores the result in J
			 *
			 * \param[in] Y The evaluation point
			 * \param[in] K The k parameters
			 * \param[out] J Numerical representation of the BDF Jacobian
			 */
 			__host__ virtual void evaluate(
					cusp::coo_matrix<int,T,cusp::device_memory> &J,
					const cusp::array1d<T,cusp::device_memory> &Y,
					const cusp::array1d<T,cusp::device_memory> &d_kData) const;


			/**
			 * \brief method to set the gamma contant
			 *
			 * set_constants multiplies all node representation on the gpu by the input parameter gamma, to represent \f$ \gamma J \f$
			 *
			 * \param[in] gamma The constant \f$ \gamma \f$
			 */
			__host__ void set_constants(const T &gamma);		

 			/**
			 * \brief Method to reset the Jacobian to its original values 
			 *
			 * reset_constants reinitializes the constants in the nodal representation of each Jacobian term to its original value stored in the user-provided Jacobian, \f$ J \f$
			 *
			 * \param[in] J User provided Jacobian 
			 */
			__host__ void reset_constants(const coo_jacobian<T> &J);
	};


	template<typename T>
	__global__ void
		k_bdf_coo_jacobian_set_constants(const T gamma, eval_node<T> *d_jNodes, const int *d_fTerms, const int *d_fOffsetTerms, const int num_leaves);

	template<typename T>
	__global__ void
		k_bdf_coo_jacobian_reset_constants(const eval_node<T> *d_jNodes, eval_node<T> *d_gNodes, const int *d_jTerms, const int *d_jOffsetTerms, const int num_leaves); 

} // end of namespace cusolve

#include <equation_system/detail/bdfcoojacobian.inl>
