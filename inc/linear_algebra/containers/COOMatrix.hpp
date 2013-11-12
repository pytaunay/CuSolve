#ifndef _COOMATRIX_HPP_
#define _COOMATRIX_HPP_

#include <cuda.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>

#include <thrust/host_vector.h>
#include <cusp/coo_matrix.h>

#include <CppNS/linear_algebra/containers/Matrix.hpp>

namespace LinearAlgebra {
	namespace Containers {

		template<class T>
		class COOMatrix : public Matrix {
			protected:
				cusp::coo_matrix<int, T, cusp::device_memory> COOM;	
			public:
				
			};
	}
}

