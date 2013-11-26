#ifndef _SPARSEVECTOR_HPP_
#define _SPARSEVECTOR_HPP_

#include <cuda.h>
#include <linear_algebra/containers/Vector.cuh>

#include <cusp/array1d.h>

namespace LinearAlgebra {
	namespace Containers {
		
		template<class T>
		class SparseVector : public Vector<T> {
			protected:
				/* Inherited attributes:
				thrust::host_vector<T> hV; CPU holder
				int N;  Size of the vector 
				T *dV; 	 Raw device pointer 
				thrust::device_ptr<T> dpV; Thrust device_ptr to dV; allows for using thrust functionality 
				*/
//				cusp::array1d<T,cusp::device_memory> spV; /*!< Cusp array1d vector; allows for CUSP functionality */
				
			public:
				/*! \brief Empty constructor
				 *
				 *
				 */
				SparseVector() {
					this->N=0;
				}

				/*! \brief Getter of the cusp array
				 *
				 *
				 */ 
			/*	const cusp::array1d<T,cusp::device_memory> & getCuspVector() {
					return spV;
				}*/	
		};	
	}
}	

#endif

