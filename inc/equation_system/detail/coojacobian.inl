/**
 * @file coojacobian.inl
 * @author Pierre-Yves Taunay (py.taunay@psu.edu)
 * @date November, 2013
 * @brief Implementation of the methods of the class cooJacobian
 *
 */

//// STD
#include <vector>
#include <map>

//// CUDA
#include <cuda.h>
// CUSP
#include <cusp/array1d.h>
#include <cusp/coo_matrix.h>
// Thrust
#include <thrust/device_ptr.h>

//// CuSolve
#include <equation_system/systemfunctional.h>
#include <equation_system/systemjacobian.h>
#include <equation_system/evalnode.h>

/*\XXX TODO: Texture representation for doubles*/
texture<float,1,cudaReadModeElementType> yTexJ;
texture<float,1,cudaReadModeElementType> kTexJ;

#define SHARED_BUF_SIZE 256 

namespace System {
	
	template<typename T>
	cooJacobian<T>
		::cooJacobian(const SystemFunctional<T> &F) {

		int displacement=0;

		this->maxElements = -1;
		
		const std::vector<T>& kInds_F = F.getkInds();
		const std::vector<T>& constants_F = F.getConstants();
		const std::vector<std::map<T,T> >& y_complete = F.getyFull();
		const std::vector<int>& terms_F = F.getTerms();


		for (int i=0; i<terms_F.size(); i++){


			int j_size=terms_F[i];

			//gather the total y for an equation (key), don't care about vals
			//we use a std::map, exploiting insertion order
			std::map<T,vector<int> > tmp;

			for (int j=0; j<j_size; j++){

				int index = j+displacement;
				for (typename map<T,T>::const_iterator itm = y_complete[index].begin(); itm != y_complete[index].end(); itm++){
					tmp[itm->first].push_back(j);
				}

			}

			// these are all the valid j indices for this row (i) in jacobian
			for (typename map<T,vector<int> >::iterator itmv = tmp.begin(); itmv != tmp.end(); itmv++){
				//convert 1 based index to 0
				this->idxJ.push_back((int) itmv->first -1);
				this->idxI.push_back(i);
			}

			//each GPU block is going to eval one i,j in jacobian
			//so as with function evaluation we'll store the number of terms in i,j evaluation
			//and thus a block will know how many nodes/structs to load

			for (typename map<T,vector<int> >::iterator itmv = tmp.begin(); itmv != tmp.end(); itmv++){

				//find this y in all the maps (one map per term)
				T y = itmv->first;
				int jac_terms=0;
				for (int j=0; j<j_size; j++){

					int index = j+displacement;
					bool found=false;
					std::map<T,T> relevant_y;
					T pow = 1.0;

					for (typename map<T,T>::const_iterator itm = y_complete[index].begin(); itm != y_complete[index].end(); itm++){
						if (itm->first==y){
							found=true;
							pow = itm->second;                                
						} else{
							relevant_y[itm->first] = itm->second;
						}
					}

					// if y appeared in this map, we have a non-zero contribution to the jacobian term
					if (found){
						jac_terms++;
						if (pow == (T)1.0)
							this->constants.push_back(constants_F[index]);
						else

							this->constants.push_back(constants_F[index]*pow);


						this->kInds.push_back(kInds_F[index]);

						if (pow != (T)1.0){
							relevant_y[y]=pow-1;
						}

						this->jFull.push_back(relevant_y);                        


					}

				}


				this->maxElements = (jac_terms > this->maxElements) ? jac_terms : this->maxElements;
				this->terms.push_back(jac_terms);
			}

			displacement += terms_F[i];

		}

		assert(this->terms.size()==this->idxJ.size());
		//
		//
		this->nbElem = this->idxI.size();

	#ifdef __VERBOSE

		displacement =0;

		for (int i=0; i<this->terms.size(); i++){


			cerr << "jac element : " << i << endl;
			cerr << "indices : " << this->idxI[i] << " " << this->idxJ[i] << endl;
			int j_size=this->terms[i];

			for (int j=0; j<j_size; j++){

				int index = j+displacement;
				for (typename map<T,T>::iterator itm = this->jFull[index].begin(); itm != this->jFull[index].end(); itm++){
					cout << "term/y_ind/pwr: " << j << " " << itm->first << " " << itm->second << endl;

				}

			}
			cerr << "k_inds" << endl;
			for (int j=0; j<j_size; j++){

				int index = j+displacement;
				cerr << this->kInds[index] << " ";

			}
			cerr << endl;
			cerr << "consts" << endl;
			for (int j=0; j<j_size; j++){

				int index = j+displacement;
				cerr << this->constants[index] << " ";

			}
			cerr << endl;

			displacement += this->terms[i];

		}

	#endif

	
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
				tmp_nodes[i].yIdx1        = 0;
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
	} // End of cooJacobian()


	template<typename T>
	__host__ void cooJacobian<T>
			::evaluate(cusp::coo_matrix<int,T,cusp::device_memory> &J,
					const cusp::array1d<T,cusp::device_memory> &Y,
					const cusp::array1d<T,cusp::device_memory> &d_kData) const {

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
			k_JacobianEvaluate<T> <<<blocks_j,threads_j>>> (d_Jp,this->d_jNodes,this->d_jTerms,this->d_jOffsetTerms);
			cudaThreadSynchronize();
			cudaCheckError("Error from the evaluation of the Jacobian: kernel call");

			cudaUnbindTexture(kTexJ);
			cudaUnbindTexture(yTexJ);

	}

		


	template<typename T>
	__global__ void k_JacobianEvaluate(
					T *d_Jp,
					const EvalNode<T> *d_jNodes,
					const int *d_jTerms,
					const int *d_jOffsetTerms
					) {

		__shared__ volatile T scratch[256];
	//	__shared__ T scratch[256];

		// Could use constant mem here
		int index = d_jOffsetTerms[blockIdx.x];
		int terms_this_function = d_jTerms[blockIdx.x];
		T fnt = (T)0.0;


		if (threadIdx.x<terms_this_function){

			EvalNode<T> node = d_jNodes[index+threadIdx.x];

			fnt                = node.constant;
			int K_index        = (node.kIdx-1);
			fnt                *= tex1Dfetch(kTexJ, K_index);
			//zero based indexing
			if (node.yExp1 != 0)
				fnt                *= pow(tex1Dfetch(yTexJ, node.yIdx1-1),node.yExp1);        
			if (node.yIdx2 != -1)
				fnt                *= pow(tex1Dfetch(yTexJ, node.yIdx2-1),node.yExp2);        
	
	//		printf("b : %i t: %i c: %f k: %i y1: %i e1: %f y2: %i e2: %f fnt : %f tr : %f y: %f\n",\
	//		blockIdx.x,threadIdx.x,node.constant,node.kIdx,node.yIdx1,node.yExp1,node.yIdx2,\
	//			node.yExp2, fnt, tex1Dfetch(kTexJ,node.kIdx-1), tex1Dfetch(yTexJ, node.yIdx1-1));

		}

		scratch[threadIdx.x] = fnt;

		__syncthreads();

		if (blockDim.x >= 256){
			if (threadIdx.x < 128){
				scratch [ threadIdx.x ]         += scratch [ threadIdx.x + 128 ];
			}
			__syncthreads();
		}

		if (blockDim.x >= 128){
			if (threadIdx.x < 64){
				scratch [ threadIdx.x ]         += scratch [ threadIdx.x + 64 ];
			}
			__syncthreads();
		}

		if (blockDim.x >= 64){
			if (threadIdx.x < 32){
				scratch [ threadIdx.x ]         += scratch [ threadIdx.x + 32 ];
			}
		}


		if (threadIdx.x < 16)               scratch [ threadIdx.x ]         += scratch [ threadIdx.x + 16 ];
		if (threadIdx.x < 8)                scratch [ threadIdx.x ]         += scratch [ threadIdx.x + 8 ];
		if (threadIdx.x < 4)                scratch [ threadIdx.x ]         += scratch [ threadIdx.x + 4 ];
		if (threadIdx.x < 2)                scratch [ threadIdx.x ]         += scratch [ threadIdx.x + 2 ];
		if (threadIdx.x < 1)                scratch [ threadIdx.x ]         += scratch [ threadIdx.x + 1 ];



		if (threadIdx.x == 0)
			d_Jp[blockIdx.x]         = scratch[0];
	}		

} // End of namespace System		

