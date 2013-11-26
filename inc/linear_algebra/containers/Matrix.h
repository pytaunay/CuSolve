#ifndef _MATRIX_HPP_
#define _MATRIX_HPP_

#include <cuda.h>

#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>

namespace LinearAlgebra {
	namespace Containers {
		/*! \brief Matrix: an abstract parent class for all matrix types implemented
		 *
		 * Matrix contains generic information about all the matrix types, such as number of columns and rows. 
		 * It also holds information about both the host and device pointers and arrays for data storage
		 */ 
		template<class T>
		class Matrix {
			protected:
				thrust::host_vector<T> hM;	/*!< Host container */		
				int nbCol;			/*!< Number of columns */
				int nbRow;			/*!< Number of rows */	
				
				// GPU data
				thrust::device_ptr<T> dpM;	/*!<  Thrust device ptr to dM for Thrust and CUSP integrationi */
				T *dM;				/*!<  Raw GPU pointer for allocation and data access*/
			
			public:

				/*! \brief Pure virtual function to transfer data from device to host 
				 *
				 *
				 */ 
				virtual void copyFromDevice() = 0;
				/*! \brief Pure virtual function to transfer data from host to device 
				 *
				 *
				 */
				virtual void copyToDevice() = 0;
				/*!  \brief Get number of rows
				 *
				 *
				 */ 
				virtual const int getNbRows() {
					return nbRow;
				}
				/*! \brief Get number of columns
				 *
				 *
				 */ 
				virtual const int getNbCols() {
					return nbCol;
				}

		};	
	}	
}

#endif

