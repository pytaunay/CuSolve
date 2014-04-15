/**
 * @file bdffunctional.h
 * @author Pierre-Yves Taunay (py.taunay@psu.edu)
 * @date April, 2014
 * @brief Class representation of the functional G(u) for the BDF solver
 *
 * Class representation of the functional \f$ G\left(u\right) \f$, given by
 * \f[
 * 	G\left(Y,t\right) = \left(Y - Y_{n,0}\right) - \gamma \left( F\left(Y,t\right) - Y^{(1)}_{n0} \right).
 * \f]	
 *
 */
#pragma once

namespace System {
	template<typename T>
	class BDFfunctional : public SystemFunctional<T> {
		protected:
			// Inherited attributes
			//std::vector<T> h_kData; /*!< k parameter data to load */
			//std::vector<T> h_kInds; /*!< k indices for equations */ 
			//std::vector<T> constants; /*!< all constant and sign data */
			//std::vector< map<T,T> > yFull; /*!< all y data (yindex(key) & power(value)) */
			//std::vector<int> terms; /*!< Number of terms per equation evaluation */

			//int maxElements; /*!< Maximum number of terms in the equation evaluation */
			//int maxTermSize; /*!< Maximum number of elements encountered in equation term, for node size */

			//int nbEq; /*!< Number of equations parsed that were non zero*/

			// Device wrappers 
			//EvalNode<T> *d_fNodes; /*!< Nodes in the polynomial tree representation; allocated on the device*/
			//int *d_fTerms; /*!< Number of terms per equation evaluation; allocated on the device*/
			//int *d_fOffsetTerms; /*!< Offset terms; allocated on the device*/
			//cusp::array1d<T,cusp::device_memory> d_kData; /*!< k parameter data loaded; allocated on the device*/

		private:
			BDFfunctional();

		public:
			BDFfunctional(const SystemFunctional<T>& F); 

			__host__ void setConstants(const T gamma, const thrust::device_vector<T> &Yterm);
			__host__ void resetConstants(const SystemFunctional<T>& F);
		
	};

	template<typename T>
	__global__ void
		k_BDFfunctionalSetConstants(const T gamma, const T *Y,const int *d_fTerms,const int *d_fOffsetTerms,const int num_leaves,	EvalNode<T> *d_fNodes); 


	template<typename T>
	__global__ void	k_BDFfunctionalResetConstants(
						const int *d_fTerms, 
						const int *d_fOffsetTerms,
						const int *d_gTerms, 
						const int *d_gOffsetTerms,
						const int num_leaves,
						const EvalNode<T> *d_fNodes,
						EvalNode<T> *d_gNodes);
}

#include <equation_system/detail/bdffunctional.inl>


			

