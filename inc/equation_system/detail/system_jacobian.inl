/**
 * @file system_jacobian.inl
 * @author Pierre-Yves Taunay (py.taunay@psu.edu)
 * @date May, 2014
 * @brief Method implementation for the Jacobian representation of a system of equations
 *
 */

#include <equation_system/system_jacobian.h>

namespace cusolve {

// set_grid method
template<typename T>
__host__ void system_jacobian<T>
	::set_grid() {
		int num_leaves = this->jFull.size();

		// Set up grid configuration
		blocks_j.x = this->nbElem;
		threads_j.x = (this->maxElements< 32) ? 32 : ceil((T)this->maxElements/((T)32.0))*32;
	}

// Destructor
template<typename T>
system_jacobian<T>
	::~system_jacobian() {
		// Clears out the vectors
		this->terms.clear();
		this->kInds.clear();
		this->constants.clear();
		this->jFull.clear();

		// Sets the integers to -1
		this->maxElements = -1;
		this->nbElem = -1;

		blocks_j.x = -1;
		blocks_j.y = -1;
		blocks_j.z = -1;

		threads_j.x = -1;
		threads_j.y = -1;
		threads_j.z = -1;

		// Free the GPU memory
		cudaFree(this->d_jNodes);
		cudaFree(this->d_jTerms);
		cudaFree(this->d_jOffsetTerms);
	}
	

} // end namespace cusolve	
