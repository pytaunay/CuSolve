#ifndef _COOMATRIX_HPP_
#define _COOMATRIX_HPP_

#ifdef USE_GPU
#include <cuda.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#endif

#include <thrust/host_vector.h>
#include <cusp/coo_matrix.h>

#include <CppNS/linear_algebra/containers/Matrix.hpp>

namespace LinearAlgebra {
namespace Containers {

template<class T>
class COOMatrix : public Matrix {
	protected:
		#ifdef USE_GPU
		cusp::coo_matrix<int, T, cusp::device_memory> COOM;	
		#else
		cusp::coo_matrix<int, T, cusp::host_memory> COOM;	
		#endif
	public:
		COOMatrix(int nbRow, int nbCol, thrust::host_vector<int> iIdx, thrust::host_vector<int> jIdx) {


		}

		
	};
}
}

