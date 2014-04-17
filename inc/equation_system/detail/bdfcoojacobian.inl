/**
 * @file bdfcoojacobian.inl
 * @author Pierre-Yves Taunay (py.taunay@psu.edu)
 * @date April, 2014
 * @brief Implementation of some methods for the derived class BDFcooJacobian 
 *
 */

//// CUDA
#include <cuda.h>
// CUSP
#include <cusp/elementwise.h>

// CuSolve
#include <equation_system/systemjacobian.h>
#include <equation_system/coojacobian.h>

namespace System {

	template<typename T>
	BDFcooJacobian<T>
		::~BDFcooJacobian() {
			cudaFree(this->d_jNodes);
			cudaFree(this->d_jTerms);
			cudaFree(this->d_jOffsetTerms);
		}	

// neq : total size of the functional
	template<typename T>
	BDFcooJacobian<T>
		::BDFcooJacobian(const cooJacobian<T> &J,int nEq) {
		// CPU: Copy data attributes
		this->idxI = J.getIdxI();
		this->idxJ = J.getIdxJ();
		this->terms = J.getTerms();
		this->kInds = J.getkInds();
		this->constants = J.getConstants();
		this->jFull = J.getjFull();

		this->maxElements = J.getMaxElements();
		this->nbElem = J.getnbElem();

		// LION-CODES: init.cpp
		// Number of leaves in the Jacobian matrix
		int num_leaves         = this->constants.size();
		// Number of equations representing the Jacobian (i.e. number of entries in the Jacobian)
		int num_funcs         = this->terms.size();

		EvalNode<T>* tmp_nodes         = new EvalNode<T>[ num_leaves ];
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
		cudaMalloc((void**)&this->d_jNodes,sizeof(EvalNode<T>)*num_leaves);
		cudaCheckError("malloc, d_jNodes");
		cudaMemcpy(this->d_jNodes,tmp_nodes,sizeof(EvalNode<T>)*num_leaves, cudaMemcpyHostToDevice);
		cudaCheckError("memcpy, d_jNodes");

		cudaMalloc((void**)&this->d_jTerms, sizeof(int)*num_funcs);
		cudaCheckError("malloc, d_jTerms");

		cudaMalloc((void**)&this->d_jOffsetTerms, sizeof(int)*num_funcs);
		cudaCheckError("malloc, d_jOffsetTerms");

		cudaMemcpy(this->d_jTerms, tmp_terms, sizeof(int)*num_funcs, cudaMemcpyHostToDevice);
		cudaCheckError("memcpy, d_jTerms");
		cudaMemcpy(this->d_jOffsetTerms, tmp_offsets, sizeof(int)*num_funcs, cudaMemcpyHostToDevice);
		cudaCheckError("memcpy, d_jOffsetTerms");


		delete[] tmp_terms, tmp_nodes, tmp_offsets;


		//// Set the identity matrix
		// Indices are 0 through Neq-1
		thrust::host_vector<int> tmpIdx(nEq);
		thrust::sequence(tmpIdx.begin(),tmpIdx.end());
		

		this->ID.resize(nEq,nEq,nEq);
		thrust::copy(tmpIdx.begin(),tmpIdx.end(),ID.row_indices.begin());
		thrust::copy(tmpIdx.begin(),tmpIdx.end(),ID.column_indices.begin());
		thrust::fill(ID.values.begin(),ID.values.end(),(T)1.0);
	}

	template<typename T>
	__host__ void BDFcooJacobian<T> 
		::evaluate(cusp::coo_matrix<int,T,cusp::device_memory> &J,
				const cusp::array1d<T,cusp::device_memory> &Y,
				const cusp::array1d<T,cusp::device_memory> &d_kData) const {

			// Fill the original J with zeros
			thrust::fill(J.values.begin(),J.values.end(),(T)(0.0));
			int nbJac = this->nbElem;
			int mxTermsJ = this->maxElements;

			// Get the pointer to the data
			// J.values = array1d inside the coo_matrix
			T *d_Jp = thrust::raw_pointer_cast(J.values.data());
			const T *d_yp = thrust::raw_pointer_cast(Y.data());
			const T *d_kp = thrust::raw_pointer_cast(d_kData.data());
			
			// Bind textures
			cudaBindTexture(0,kTexJ,d_kp,sizeof(T)*d_kData.size());
			cudaBindTexture(0,yTexJ,d_yp,sizeof(T)*Y.size());
			cudaThreadSynchronize();
			cudaCheckError("Error from the evaluation of the Jacobian: texture binding");

			// Set up grid configuration
			dim3 blocks_j, threads_j;
			blocks_j.x = nbJac;
			threads_j.x = (mxTermsJ< 32) ? 32 : ceil((T)mxTermsJ/((T)32.0))*32;

			std::cout << "Starting the Jacobian evaluation routine on the GPU with NTH=" << threads_j.x << " and NBL=" << blocks_j.x << std::endl;
		//	k_JacobianEvaluate<T> <<<blocks_j,threads_j>>> (d_Jp,this->d_jNodes,this->d_jTerms,this->d_jOffsetTerms);
			k_JacobianEvaluate<T> <<<blocks_j,threads_j>>> (d_Jp,this->d_jNodes,this->d_jTerms,this->d_jOffsetTerms,d_kp,d_yp);
			cudaThreadSynchronize();
			cudaCheckError("Error from the evaluation of the Jacobian: kernel call");

			cudaUnbindTexture(kTexJ);
			cudaUnbindTexture(yTexJ);

			// Add the identity matrix to get I - gamma*J
			// cusp::add
			cusp::add(J,this->ID,J);

	}




	template<typename T>
	__host__ void BDFcooJacobian<T>
		::setConstants(const T gamma){

		int nbJac = this->nbElem;
		int mxTermsJ = this->maxElements;
		int num_leaves = this->jFull.size();

		// Set up grid configuration
		dim3 blocks_j, threads_j;
		blocks_j.x = nbJac;
		threads_j.x = (mxTermsJ< 32) ? 32 : ceil((T)mxTermsJ/((T)32.0))*32;

		k_BDFcooJacobianSetConstants <<<blocks_j,threads_j>>> (gamma,this->d_jNodes,this->d_jTerms,this->d_jOffsetTerms, num_leaves);
		cudaThreadSynchronize();
		cudaCheckError("Error from the BDF Jacobian setting the data");

	}

	template<typename T>
	__global__ void
		k_BDFcooJacobianSetConstants(const T gamma, EvalNode<T> *d_jNodes, const int *d_jTerms, const int *d_jOffsetTerms, const int num_leaves) {

		T gam = gamma;
		int index = d_jOffsetTerms[blockIdx.x];
		int terms_this_function = d_jTerms[blockIdx.x];

		EvalNode<T> node;
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
	__host__ void BDFcooJacobian<T>
		::resetConstants(const cooJacobian<T> &J){

		int nbJac = this->nbElem;
		int mxTermsJ = this->maxElements;
		int num_leaves = this->jFull.size();

		// Set up grid configuration
		dim3 blocks_j, threads_j;
		blocks_j.x = nbJac;
		threads_j.x = (mxTermsJ< 32) ? 32 : ceil((T)mxTermsJ/((T)32.0))*32;

		k_BDFcooJacobianResetConstants <<<blocks_j,threads_j>>> (J.getNodes(),this->d_jNodes,this->d_jTerms,this->d_jOffsetTerms, num_leaves);
		cudaThreadSynchronize();
		cudaCheckError("Error from the BDF Jacobian setting the data");


		}

	template<typename T>
	__global__ void
		k_BDFcooJacobianResetConstants(const EvalNode<T> *d_jNodes, EvalNode<T> *d_gNodes, const int *d_jTerms, const int *d_jOffsetTerms, const int num_leaves) {

		int index = d_jOffsetTerms[blockIdx.x];
		int terms_this_function = d_jTerms[blockIdx.x];

		EvalNode<T> nodeJ,nodeG;
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
	




