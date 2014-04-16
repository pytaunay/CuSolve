/**
 * @file bdffunctional.inl
 * @author Pierre-Yves Taunay (py.taunay@psu.edu)
 * @date April, 2014
 * @brief Implementation of the methods of the class BDFfunctional
 *
 */

#include <vector>
#include <map>
#include <utility>

namespace System {

	template<typename T>
	BDFfunctional<T>
		::BDFfunctional(const SystemFunctional<T>& F) {
			// Copy data
			this->yFull = F.getyFull();
			this->terms = F.getTerms();
			this->constants = F.getConstants();
			this->h_kData = F.getkDataHost(); 
			this->h_kInds = F.getkInds();

			this->maxElements = F.getMaxElements();
			this->maxTermSize = F.getMaxTermSize();

			// Push back a bogus value of k for the new terms in the equation 
			this->h_kData.push_back(1.0f);

			// Convenience iterators
			typename std::vector< std::map<T,T> >::iterator it;
			typename std::vector<T>::iterator itCst, itK;

			int vectorPos = 0; // Insertion position
			int nterms = 0; // Number of terms in the equation
			int nk = this->h_kData.size(); // Total number of k's; last k is the bogus k, equal to 1.0

			// Add terms to all equations
			for(int e = 1; e <= this->terms.size(); e++) {
				nterms = this->terms[e-1];

				std::map<T,T> t1,t2;

				//// Insert elements in Y
				// c*k-1 * y0^0
				// Position for insertion
				itCst = this->constants.begin() + vectorPos + nterms;
				itK = this->h_kInds.begin() + vectorPos + nterms;
				it = this->yFull.begin() + vectorPos + nterms;

				// Value
				t2.insert(std::pair<T,T>(1.0,0.0)); // y1^0

				// Insertion operation
				this->constants.insert(itCst,1.0f); // 1.0
				this->h_kInds.insert(itK, nk); // k-1
				this->yFull.insert(it,t2);// y1^0


				// 1.0 * k-1 * yi^1
				// Position for insertion
				itCst = this->constants.begin() + vectorPos + nterms;
				itK = this->h_kInds.begin() + vectorPos + nterms;
				it = this->yFull.begin() + vectorPos + nterms;

				// Value
				t1.insert(std::pair<T,T>(e,1.0)); // yi^1

				// Insertion operation
				this->constants.insert(itCst,1.0f); // 1.0
				this->h_kInds.insert(itK, nk); // k-1
				this->yFull.insert(it,t1); // yi^1

				// Adjust next insertion
				vectorPos+=nterms+2;
				
				// Increment total number of terms by 2
				this->terms[e-1] +=2;

			}
			
			// Total number of equations stays the same
			this->nbEq = this->terms.size();
			// Add 2 to the max number of elements
			this->maxElements += 2;

			//// Allocate GPU memory, and transfer data
			// LION-CODES: init_f
			// Initialization of the function based on the data gathered
			int num_leaves 	= this->constants.size();
			int num_funcs 	= this->terms.size();
			
			EvalNode<T> *tmp_nodes 		= new EvalNode<T> [ num_leaves ];
			int *tmp_terms 			= new int [ num_funcs ];
			T *tmp_y	 		= new T [ num_funcs ];
			int *tmp_offsets		= new int[num_funcs];

			tmp_offsets[0] = 0;
			int off = this->terms[0];

			for (int i=0; i<num_funcs; i++){
				if (i>0) {
					tmp_offsets[i] = off;
					off += this->terms[i];
				}
				tmp_terms[i] = this->terms[i];
			}	

			for (int i=0; i<num_leaves;i++){
				tmp_nodes[i].constant 	= this->constants[i];
				tmp_nodes[i].kIdx 	= (int) this->h_kInds[i];
				tmp_nodes[i].yIdx1	= 0;
				tmp_nodes[i].yExp1	= 1.0;
				tmp_nodes[i].yIdx2	= -1;
				tmp_nodes[i].yExp2	= 1.0;
			
				std::map<T,T> tmp = this->yFull[i];

				tmp_nodes[i].yIdx1 	= (int) tmp.begin()->first;
				tmp_nodes[i].yExp1 	= tmp.begin()->second;
			
				if (tmp.size()>1){
					typename std::map<T,T>::iterator it = tmp.begin();
					it++;
					tmp_nodes[i].yIdx2 	= (int)it->first;
					tmp_nodes[i].yExp2 	= it->second;
				}
			}


			//// Allocate device memory, and copy to device
			// d_fNodes
			cudaMalloc((void**)&this->d_fNodes, sizeof(EvalNode<T>)*num_leaves);
			//cudaCheckError("malloc, f_nodes_dev");
			cudaMemcpy(this->d_fNodes,tmp_nodes, sizeof(EvalNode<T>)*num_leaves, cudaMemcpyHostToDevice);
			//cudaCheckError("memcpy, f_nodes_dev");
			
			// d_fTerms
			cudaMalloc((void**)&this->d_fTerms, sizeof(int)*num_funcs);
			//cudaCheckError("malloc, terms_dev");
			cudaMemcpy(this->d_fTerms, tmp_terms, sizeof(int)*num_funcs, cudaMemcpyHostToDevice);
			//cudaCheckError("memcpy, terms_dev");
			
			cudaMalloc((void**)&this->d_fOffsetTerms, sizeof(int)*num_funcs);
			cudaMemcpy(this->d_fOffsetTerms, tmp_offsets, sizeof(int)*num_funcs, cudaMemcpyHostToDevice);


			// d_kInds
			int num_k = this->h_kData.size();
			this->d_kData.resize(num_k);
			thrust::copy(this->h_kData.begin(),this->h_kData.end(),this->d_kData.begin());

			delete[] tmp_terms, tmp_nodes, tmp_y, tmp_offsets;
		} // End of BDFfunctional()


		template<typename T>
		__host__ void BDFfunctional<T>
			::setConstants(
				const T gamma,
				const thrust::device_vector<T> &Yterm
				) {
			// Number of equations, and maximum numbers of terms
			int nbEq = this->nbEq;
			int mxTermsF = this->maxElements;
			int num_leaves = this->yFull.size();

			const T *d_yp = thrust::raw_pointer_cast(Yterm.data());

			// Determine running configuration
			dim3 blocks_f, threads_f;
			blocks_f.x = nbEq;
			threads_f.x = ( mxTermsF < 32 ) ? 32 : ceil( (T)mxTermsF / ((T)32.0))*32;

			k_BDFfunctionalSetConstants <<< blocks_f, threads_f >>> ( gamma, d_yp , this->d_fTerms, this->d_fOffsetTerms,num_leaves,this->d_fNodes);
				
			cudaThreadSynchronize();
			cudaCheckError("Error in the setting of parameters in the BDF functional");
		} // End of BDFfunctional::setConstants()	




		// Kernel to set the constants of the BDF Functional
		// The first TERMS-2 (idx: 0 -> TERMS-3) nodes correspond to the original functional. They have to be multiplied by
		// -gammma
		// The [TERMS-1]th node (idx: TERMS-2) does not have to be changed
		// The [TERMS]th node (idx: TERMS-1) has to be set to a constant based on the input vector Y
		template<typename T>
		__global__ void
			k_BDFfunctionalSetConstants(
						const T gamma, 
						const T *Y, 
						const int *d_fTerms, 
						const int *d_fOffsetTerms,
						const int num_leaves,
						EvalNode<T> *d_fNodes) {
			
			// Constant memory LDU
			T gam = gamma;

			//could use constant mem here
			int index = d_fOffsetTerms[blockIdx.x];
			int terms_this_function = d_fTerms[blockIdx.x];

			T Yval = Y[blockIdx.x];
			
			EvalNode<T> node;

			// All threads load something
			if(index + threadIdx.x < num_leaves) {
				node = d_fNodes[index + threadIdx.x];

				// Divergence starts here
				if(threadIdx.x < terms_this_function - 2 ) {
					node = d_fNodes[index + threadIdx.x];
					node.constant *= -1.0f*gam;

				} else if (threadIdx.x == terms_this_function-1) { 	
					node = d_fNodes[index + threadIdx.x];
					node.constant = Yval;
				}	

				// Write data back
				__syncthreads();
				if(threadIdx.x < terms_this_function) {
					d_fNodes[index + threadIdx.x] = node;
				}	
			
			}
		}	

		template<typename T>
		__host__ void BDFfunctional<T>
			::resetConstants(
				const SystemFunctional<T>& F
				) {
			// Number of equations, and maximum numbers of terms
			int nbEq = this->nbEq;
			int mxTermsF = this->maxElements;
			int num_leaves = this->yFull.size();

			// Determine running configuration
			dim3 blocks_f, threads_f;
			blocks_f.x = nbEq;
			threads_f.x = ( mxTermsF < 32 ) ? 32 : ceil( (T)mxTermsF / ((T)32.0))*32;

			k_BDFfunctionalResetConstants <<< blocks_f, threads_f >>> ( F.getDevTerms(), F.getDevOffset(), this->d_fTerms, this->d_fOffsetTerms,num_leaves, F.getNodes(), this->d_fNodes);
				
			cudaThreadSynchronize();
			cudaCheckError("Error in the setting of parameters in the BDF functional");
		} // End of BDFfunctional::resetConstants()	

		template<typename T>
		__global__ void
			k_BDFfunctionalResetConstants(
						const int *d_fTerms, 
						const int *d_fOffsetTerms,
						const int *d_gTerms, 
						const int *d_gOffsetTerms,
						const int num_leaves,
						const EvalNode<T> *d_fNodes,
						EvalNode<T> *d_gNodes) {
			
			//could use constant mem here
			int indexG = d_gOffsetTerms[blockIdx.x];
			int indexF = d_fOffsetTerms[blockIdx.x];

			int terms_this_function = d_gTerms[blockIdx.x];

			EvalNode<T> nodeF,nodeG;

			// All threads load something
			if(indexG + threadIdx.x < num_leaves) {
				nodeG = d_gNodes[indexG + threadIdx.x];

				// Divergence starts here
				if(threadIdx.x < terms_this_function - 2 ) {
					nodeF = d_fNodes[indexF + threadIdx.x];
					nodeG.constant = nodeF.constant;
					d_gNodes[indexG + threadIdx.x] = nodeG;
				}	
			}
		}	
}		
