#ifndef _COOMATRIX_HPP_
#define _COOMATRIX_HPP_

#include <cuda.h>

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>

#include <cusp/coo_matrix.h>

#include <linear_algebra/containers/SparseMatrix.cuh>

using namespace LinearAlgebra::Containers;

namespace LinearAlgebra {
	namespace Containers {
		
		/*! \brief COO Representation of a matrix
		 *
		 * COOMatrix contains additional information for storage of a sparse matrix in a COO representation. 
		 * A cusp::coo_matrix is included to have an interface with CUSP data storage
		 */ 
		template<class T>
		class COOMatrix : public SparseMatrix<T> {
			protected:
				/* Inherited attributes
				thrust::host_vector<T> hM; 	// CPU Holder		
				int nbCol;			// Number of columns
				int nbRow;			// Number of rows
				thrust::device_ptr<T> dpM;	//  Thrust device ptr to dM for Thrust and CUSP integration
				T *dM;				//  Raw GPU pointer
				*/
				cusp::coo_matrix<int, T, cusp::device_memory> cooM; /*! < COO matrix for storage, using CUSP */	
				int nbEntry; /*!< Total number of non zero entries in the sparse matrix */

			public:
				//// Constructors
				// 1. Just create an empty COO Matrix
				/*! \brief Empty COO Matrix constructor
				 *
				 * The empty constructor initializes all pointers ot NULL and all counters to zero.
				 */ 
				COOMatrix() {
					this->cooM = cusp::coo_matrix<int, T, cusp::device_memory>();
					this->dM = NULL;
					this->nbCol = 0;
					this->nbRow = 0;
					this->nbEntry = 0;
				}	

				// 2. Create an empty COO Matrix of size nbRow, nbCol, nbEntry
				/*! \brief Constructor based on the location of the non-zero entries in the matrix
				 *
				 *
				 */ 
				COOMatrix(int nbRow, int nbCol, int nbEntry,thrust::host_vector<int> iIdx, thrust::host_vector<int> jIdx) {

						cooM =cusp:: coo_matrix<int, T, cusp::device_memory>(nbRow,nbCol,nbEntry);
						thrust::copy(iIdx.begin(),iIdx.end(),cooM.row_indices.begin());
						thrust::copy(jIdx.begin(),jIdx.end(),cooM.column_indices.begin());

						this->nbRow = nbRow;
						this->nbCol = nbCol;
						this->nbEntry = nbEntry;

						this->transferDataToCoo();

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
					cudaFree(this->dM);
					this->hM.clear();
					this->nbCol = 0;
					this->nbRow = 0;
					this->nbEntry = 0;
				}	
				
				//// Methods	
				// Transfer data from a thrust device pointer
				void transferDataToCOO() {
						thrust::copy_n(this->dpM,this->cooM.values.size(),this->cooM.values.begin());
				}		
				// Transfer data from the matrix, to a device vector
				void transferDataFromCOO() {
					thrust::copy(this->cooM.values.begin(),this->cooM.values.end(),this->hM.begin());
				}	
				/*! \brief Transfer data from the matrix to the host
				 *
				 *
				 */ 
				void copyFromDevice() {
					thrust::copy(this->cooM.values.begin(),this->cooM.values.end(),this->hM.begin());
				}	
				/*! \brief Transfer data from the host to the matrix
				 *
				 *
				 */ 
				void copyToDevice() {
					thrust::copy(this->hM.begin(),this->hM.end(),this->cooM.values.begin());
				}	

				/*! \brief Get number of entries
				 *
				 *
				 */ 
				int getNbEntry() {
					return nbEntry;
				}	
				// Get address of the matrix
			/*	const cusp::coo_matrix<int, T, cusp::device_memory> & getMatrix() {		
					return cooM; 
				}	
*/
				// TODO
				// - Getter of hM, dpM, dM
				// - Setter of hM, dpM, dM
		};
	}
}
#endif

