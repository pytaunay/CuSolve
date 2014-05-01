/**
 * @file bdfcoojacobian.inl
 * @author Pierre-Yves Taunay (py.taunay@psu.edu)
 * @date April, 2014
 * @brief Implementation of some methods for the derived class bdf_coo_jacobian 
 *
 */

//// CUDA
#include <cuda.h>
// CUSP
#include <cusp/elementwise.h>

// CuSolve
#include <equation_system/coo_jacobian.h>
#include <equation_system/bdf_coo_jacobian.h>

namespace cusolve {

	template<typename T>
	bdf_coo_jacobian<T>
		::~bdf_coo_jacobian() {
			
			this->idxI.clear();
			this->idxJ.clear();

			this->terms.clear();
			this->kInds.clear();
			this->constants.clear();
			this->jFull.clear();

			this->maxElements = 0;

			cudaFree(this->d_jNodes);
			cudaFree(this->d_jTerms);
			cudaFree(this->d_jOffsetTerms);
		}	

	template<typename T>
	bdf_coo_jacobian<T>
		::bdf_coo_jacobian(const coo_jacobian<T> &J,const int &nEq) {
		// CPU: Copy data attributes
		this->idxI = J.get_idx_i();
		this->idxJ = J.get_idx_j();
		this->terms = J.get_terms();
		this->kInds = J.get_kinds();
		this->constants = J.get_constants();
		this->jFull = J.get_jfull();

		this->maxElements = J.get_max_elements();
		this->nbElem = J.get_nb_elem();

		int num_leaves         = this->constants.size();
		int num_funcs         = this->terms.size();

		// Allocate memory
		cudaMalloc((void**)&this->d_jNodes,sizeof(eval_node<T>)*num_leaves);
		cudaCheckError("malloc, d_jNodes");
		cudaMalloc((void**)&this->d_jTerms, sizeof(int)*num_funcs);
		cudaCheckError("malloc, d_jTerms");
		cudaMalloc((void**)&this->d_jOffsetTerms, sizeof(int)*num_funcs);
		cudaCheckError("malloc, d_jOffsetTerms");

		// Device pointers
		thrust::device_ptr<eval_node<T>> dptr_jNodes = thrust::device_pointer_cast(this->d_jNodes);
		thrust::device_ptr<int> dptr_jTerms = thrust::device_pointer_cast(this->d_jTerms);
		thrust::device_ptr<int> dptr_jOffsetTerms = thrust::device_pointer_cast(this->d_jOffsetTerms);

		thrust::device_ptr<eval_node<T>> dptr_original_jNodes = thrust::device_pointer_cast(J.get_jnodes());
		thrust::device_ptr<int> dptr_original_jTerms = thrust::device_pointer_cast(J.get_jterms());
		thrust::device_ptr<int> dptr_original_jOffsetTerms = thrust::device_pointer_cast(J.get_joffset_terms());

		// Copy data from original
		thrust::copy(dptr_original_jNodes,
				dptr_original_jNodes + num_leaves,
				dptr_jNodes); 
		thrust::copy(dptr_original_jTerms,
				dptr_original_jTerms + num_funcs,
				dptr_jTerms);
		thrust::copy(dptr_original_jOffsetTerms,
				dptr_original_jOffsetTerms + num_funcs,
				dptr_jOffsetTerms);

		//// Set the identity matrix
		// Indices are 0 through Neq-1
//		thrust::host_vector<int> tmpIdx(nEq);
//		thrust::sequence(tmpIdx.begin(),tmpIdx.end());
		

//		thrust::copy(tmpIdx.begin(),tmpIdx.end(),ID.row_indices.begin());
//		thrust::copy(tmpIdx.begin(),tmpIdx.end(),ID.column_indices.begin());

		this->ID.resize(nEq,nEq,nEq);
		thrust::sequence(ID.row_indices.begin(),ID.row_indices().end());
		thrust::sequence(ID.column_indices.begin(),ID.column_indices().end());

		thrust::fill(ID.values.begin(),ID.values.end(),(T)1.0);

/*
		// LION-CODES: init.cpp
		// Number of leaves in the Jacobian matrix
		int num_leaves         = this->constants.size();
		// Number of equations representing the Jacobian (i.e. number of entries in the Jacobian)
		int num_funcs         = this->terms.size();

		eval_node<T>* tmp_nodes         = new eval_node<T>[ num_leaves ];
		int * tmp_terms                 = new int[ num_funcs ];
		int * tmp_offsets                 = new int[ num_funcs ];

		tmp_offsets[0]        = 0;
		int off         = this->terms[0];

		for (int i=1; i<num_funcs; i++){
			tmp_offsets[i] = off;
			off+=this->terms[i];
		}

		for (int i=0; i<num_funcs; i++)
			tmp_terms[i] = this->terms[i];

		for (int i=0; i<num_leaves; i++){
			tmp_nodes[i].constant     = this->constants[i];
			tmp_nodes[i].kIdx         = (int) this->kInds[i];
			tmp_nodes[i].yIdx1        = 1;
			tmp_nodes[i].yExp1        = 1.0;
			tmp_nodes[i].yIdx2        = -1;
			tmp_nodes[i].yExp2        = 1.0;

			std::map<T,T> tmp = this->jFull[i];

			tmp_nodes[i].yIdx1         = (int) tmp.begin()->first;
			tmp_nodes[i].yExp1         = tmp.begin()->second;

			if (tmp.size()>1){
				// Typename keyword required
				// See http://stackoverflow.com/questions/3184682/map-iterator-in-template-function-unrecognized-by-compiler
				typename std::map<T,T>::iterator it = tmp.begin();
				it++;
				tmp_nodes[i].yIdx2         = (int) it->first;
				tmp_nodes[i].yExp2         = it->second;
			}
		}
*/

/*
		cudaMemcpy(this->d_jNodes,tmp_nodes,sizeof(eval_node<T>)*num_leaves, cudaMemcpyHostToDevice);
		cudaCheckError("memcpy, d_jNodes");
		cudaMemcpy(this->d_jTerms, tmp_terms, sizeof(int)*num_funcs, cudaMemcpyHostToDevice);
		cudaCheckError("memcpy, d_jTerms");
		cudaMemcpy(this->d_jOffsetTerms, tmp_offsets, sizeof(int)*num_funcs, cudaMemcpyHostToDevice);
		cudaCheckError("memcpy, d_jOffsetTerms");


		delete[] tmp_terms, tmp_nodes, tmp_offsets;
*/

	}

	template<typename T>
	__host__ void bdf_coo_jacobian<T> 
		::evaluate(
			cusp::coo_matrix<int,T,cusp::device_memory> &J,
			const cusp::array1d<T,cusp::device_memory> &Y,
			const cusp::array1d<T,cusp::device_memory> &d_kData) const {

		// Fill the original J with zeros
		thrust::fill(J.values.begin(),J.values.end(),(T)(0.0));

		// Get the pointer to the data
		// J.values = array1d inside the coo_matrix
		T *d_Jp = thrust::raw_pointer_cast(J.values.data());
		const T *d_yp = thrust::raw_pointer_cast(Y.data());
		const T *d_kp = thrust::raw_pointer_cast(d_kData.data());
		
		// Set up grid configuration
		set_grid();

		// Launch kernel
		#ifdef VERBOSE
		std::cout << "Starting the Jacobian evaluation routine on the GPU with NTH=" << this->threads_j.x << " and NBL=" << this->blocks_j.x << std::endl;
		#endif 
		k_jacobian_evaluate<T> <<<this->blocks_j,this->threads_j>>> (d_Jp,this->d_jNodes,this->d_jTerms,this->d_jOffsetTerms,d_kp,d_yp);
		cudaThreadSynchronize();
		cudaCheckError("ERROR from the evaluation of the Jacobian: kernel call");

		// Add the identity matrix to get I - gamma*J
		cusp::add(J,this->ID,J);
	}


	template<typename T>
	__host__ void bdf_coo_jacobian<T>
		::set_constants(const T &gamma){

		int num_leaves = this->jFull.size();

		// Grid configuration
		set_grid();

		// Kernel launch
		k_bdf_coo_jacobian_set_constants<<<this->blocks_j,this->threads_j>>> (gamma,this->d_jNodes,this->d_jTerms,this->d_jOffsetTerms, num_leaves);
		cudaThreadSynchronize();
		cudaCheckError("ERROR from the BDF Jacobian setting the data");

	}

	template<typename T>
	__global__ void
		k_bdf_coo_jacobian_set_constants(
			const T gamma, 
			eval_node<T> *d_jNodes, 
			const int *d_jTerms, 
			const int *d_jOffsetTerms, 
			const int num_leaves) {

		T gam = gamma;
		int index = d_jOffsetTerms[blockIdx.x];
		int terms_this_function = d_jTerms[blockIdx.x];

		eval_node<T> node;
		if(index + threadIdx.x < num_leaves) {
			// ALl threads load sthg
			node = d_jNodes[index + threadIdx.x];

			// Warp divergence occurs here
			if(threadIdx.x < terms_this_function) {
				node.constant *= -(T)1.0*gam;
				d_jNodes[index+threadIdx.x] = node;
			}
		}	
	}

	template<typename T>
	__host__ void bdf_coo_jacobian<T>
		::reset_constants(const coo_jacobian<T> &J){

		int num_leaves = this->jFull.size();

		// Set up grid configuration
		set_grid();

		// Kernel launch
		k_bdf_coo_jacobian_reset_constants<<<this->blocks_j,this->threads_j>>> (J.get_nodes(),this->d_jNodes,this->d_jTerms,this->d_jOffsetTerms, num_leaves);
		cudaThreadSynchronize();
		cudaCheckError("Error from the BDF Jacobian setting the data");
	}

	template<typename T>
	__global__ void
		k_bdf_coo_jacobian_reset_constants(
			const eval_node<T> *d_jNodes, 
			eval_node<T> *d_gNodes, 
			const int *d_jTerms, 
			const int *d_jOffsetTerms, 
			const int num_leaves) {

		int index = d_jOffsetTerms[blockIdx.x];
		int terms_this_function = d_jTerms[blockIdx.x];

		eval_node<T> nodeJ,nodeG;
		if(index + threadIdx.x < num_leaves) {
			// ALl threads load sthg
			nodeJ = d_jNodes[index + threadIdx.x];
			nodeG = d_gNodes[index + threadIdx.x];

			// Warp divergence occurs here
			if(threadIdx.x < terms_this_function) {
				nodeG.constant = nodeJ.constant;
				d_gNodes[index+threadIdx.x] = nodeG;
			}
		}	
	}
		

}
	




