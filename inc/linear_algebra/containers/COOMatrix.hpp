#ifndef _COOMATRIX_HPP_
#define _COOMATRIX_HPP_

#include <cuda.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>

#include <cusp/coo_matrix.h>

#include <CppNS/linear_algebra/containers/Matrix.hpp>

namespace LinearAlgebra {
	namespace Containers {

		template<class T>
		class COOMatrix : public Matrix<T> {
			protected:
				/* Inherited attributes
				thrust::host_vector<T> hM; 	// CPU Holder		
				int nbCol;			// Number of columns
				int nbRow;			// Number of rows
				thrust::device_ptr<T> dpM;	//  Thrust device ptr to dM for Thrust and CUSP integration
				T *dM;				//  Raw GPU pointer
				*/
				cusp::coo_matrix<int, T, cusp::device_memory> cooM;	
				int nbEntry;

			public:
				//// Constructors
				// 1. Just create an empty COO Matrix
				COOMatrix() {
					cooM = coo_matrix<int, T, cusp::device_memory>();
					dpM = NULL;
					nbCol = 0;
					nbRow = 0;
					nbEntry = 0;
				}	

				// 2. Create an empty COO Matrix of size nbRow, nbCol, nbEntry
				COOMatrix(int nbRow, int nbCol, int nbEntry,thrust::host_vector<int> iIdx, thrust::host_vector<int> jIdx) {

						cooM = coo_matrix<int, T, cusp::device_memory>(nbRow,nbCol,nbEntry);
						thrust::copy(iIdx.begin(),iIdx.end(),cooM.row_indices.begin();
						thrust::copy(jIdx.begin(),jIdx.end(),cooM.column_indices.begin();

						this->nbRow = nbRow;
						this->nbCol = nbCol
						this->nbEntry = nbEntry;

						transferDataToCoo();

				}		

				// 3. Create a COO Matrix of sizer nbRol, nbCol, nbEntry, and populate with content
			/*	COOMatrix(int nbRow, int nbCol, int nbEntry,thrust::host_vector<int> iIdx, thrust::host_vector<int> jIdx, thrust::device_ptr<T> values) {

						cooM = coo_matrix<int, T, cusp::device_memory>(nbRow,nbCol,nbEntry);
						thrust::copy(iIdx.begin(),iIdx.end(),cooM.row_indices.begin();
						thrust::copy(jIdx.begin(),jIdx.end(),cooM.column_indices.begin();
						transferDataToCoo();
				}		
			*/

				//// Destructor
				~COOMatrix() {
					cudaFree(dM);
					hM.clear();
					nbCol = 0;
					nbRow = 0;
					nbEntry = 0;
				}	
				
				//// Methods	
				// Transfer data from a thrust device pointer
				void transferDataToCOO() {
						thrust::copy_n(dpM,cooM.values.size(),cooM.values.begin());
				}		
				// Transfer data from the matrix, to a device vector
				void transferDataFromCOO() {
					thrust::copy(cooM.values.begin(),cooM.values.end(),hM.begin());
				}	
				// Transfer data from the matrix to the host
				void copyFromDevice() {
					thrust::copy(cooM.values.begin(),cooM.values.end(),hM.begin());
				}	
				// Transfer data from the host to the matrix
				void copyToDevice() {
					thrust::copy(hM.begin(),hM.end(),cooM.values.begin());
				}	

				// Get number of entries
				int getNbEntry() {
					return nbEntry;
				}	
				// Get address of the matrix
				const cusp::coo_matrix<int, T, cusp::device_memory> & getMatrix() {		
					return &cooM; 
				}	

				// TODO
				// - Getter of hM, dpM, dM
				// - Setter of hM, dpM, dM


		};
	}
}

