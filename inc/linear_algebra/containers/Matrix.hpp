#ifndef _MATRIX_HPP_
#define _MATRIX_HPP_

#include <cuda.h>

#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>

namespace LinearAlgebra {
	namespace Containers {

		template<class T>
		class Matrix {

			protected:
				// CPU Holder
				thrust::host_vector<T> hM; 	// CPU Holder		
				int nbCol;			// Number of columns
				int nbRow;			// Number of rows
				
				// GPU data
				thrust::device_ptr<T> dpM;	//  Thrust device ptr to dM for Thrust and CUSP integration
				T *dM;				//  Raw GPU pointer
			
			public:
				virtual void copyFromDevice() = 0;
				virtual void copyToDevice() = 0;
				// Get number of rows
				virtual const int getNbRows() {
					return nbRow;
				}
				// Get number of columns
				virtual const int getNbCols() {
					return nbCol;
				}

		};	
	}	
}

#endif

