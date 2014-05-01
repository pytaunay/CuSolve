/** 
 * @file bdf_jacobian.h
 * @author Pierre-Yves Taunay (py.taunay@psu.edu)
 * @date April, 2014
 * @brief Interface for the modified Jacobian classes necessary in the BDF solver
 */

#pragma once

//// CUDA
#include <cuda.h>

//// cusolve
#include <equation_system/system_jacobian.h>
#include <equation_system/coo_jacobian.h>

namespace cusolve {
	/*! \class bdf_jacobian bdf_jacobian.h "inc/equation_system/bdf_jacobian.h"
	 * \brief Interface for the BDF jacobian
	 *
	 * Abstract class which is an interface to the different Jacobian representations necessary in the BDF solver. Derives from the system_jacobian class.
	 * The representations in the BDF solver are for representing analytically the modified Jacobian \f$ I - \gamma J \f$. The class contains the definition of multiple
	 * pure virtual methods to set/reset the constant \f$ \gamma \f$ in the analytical representation of the matrix.
	 * Virtual inheritance is necessary since there is multiple inheritance for classes derived from bdf_jacobian.
	 *
	 * \tparam T Single / Double precision
	 */
	template<typename T>
	class bdf_jacobian : virtual public system_jacobian<T> {

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
	//		eval_node<T> *d_jNodes;
	//		int *d_jTerms;
	//		int *d_jOffsetTerms;	
	
		public:
			__host__ virtual void set_constants(const T &gamma) = 0;
			__host__ virtual void reset_constants(const coo_jacobian<T> &J) = 0;
	};
} // End of namespace cusolve	
	
