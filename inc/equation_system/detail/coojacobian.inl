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

	
			// Jacobian
			int num_leaves         = this->constants.size();
			int num_funcs         = terms_F.size();

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
		//	cudaCheckError("malloc, jac_nodes_dev");
			cudaMemcpy(this->d_jNodes,tmp_nodes,sizeof(EvalNode<T>)*num_leaves, cudaMemcpyHostToDevice);
			//cudaCheckError("memcpy, jac_nodes_dev");

			cudaMalloc((void**)&this->d_jTerms, sizeof(int)*num_funcs);
			cudaMalloc((void**)&this->d_jOffsetTerms, sizeof(int)*num_funcs);
			//cudaCheckError("malloc, terms_jac_dev");
			cudaMemcpy(this->d_jTerms, tmp_terms, sizeof(int)*num_funcs, cudaMemcpyHostToDevice);
			cudaMemcpy(this->d_jOffsetTerms, tmp_offsets, sizeof(int)*num_funcs, cudaMemcpyHostToDevice);
			//cudaCheckError("memcpy, terms_jac_dev");


			delete[] tmp_terms, tmp_nodes, tmp_offsets;
	} // End of cooJacobian()


	template<typename T>
	__host__ void cooJacobian<T>
				::evaluate(
					cusp::coo_matrix<int,T,cusp::device_memory> &J,
					const cusp::array1d<T,cusp::device_memory> &Y) {
		
		// Get the pointer to the data
		// J.values = array1d inside the coo_matrix
		T *d_Jp = thrust::raw_pointer_cast(J.values.data());


		/*\XXX TODO: Determine nthreads, nblocks*/
		/*\XXX TODO: Fix textures*/
	//	k_evaluate<<<1,1>>>(d_Jp);

	} // End of cooJacobian::evaluate

	template<typename T>
	__device__ void cooJacobian<T> 
				::implementation(T *d_Jp) {
			__shared__ volatile T scratch[SHARED_BUF_SIZE];

			// Could use constant mem here
			int index = this->d_jOffsetTerms[blockIdx.x];
			int terms_this_function = this->d_jTerms[blockIdx.x];
			T fnt = 0.0f;

			if (threadIdx.x<terms_this_function){

				EvalNode<T> node = this->d_jNodes[index+threadIdx.x];

				fnt                = node.constant;
				int K_index        = (node.k_index-1);
				fnt                *= tex1Dfetch(kTex, K_index);
				//zero based indexing
				if (node.yExp1 != 0)
					fnt                *= powf(tex1Dfetch(yTex, node.yIdx1-1),node.yExp1);        
				if (node.yIdx2 != -1)
					fnt                *= powf(tex1Dfetch(yTex, node.yIdx2-1),node.yExp2);        

				//if (blockIdx.x==0) printf("b : %i t: %i c: %f k: %i y1: %i e1: %f y2: %i e2: %f fnt : %f tr : %f y: %f\n",\
				blockIdx.x,threadIdx.x,node.constant,node.k_index,node.y_index_1,node.y_exp_1,node.y_index_2,\
					node.y_exp_2, fnt, tex1Dfetch(k_tex,node.k_index-1), tex1Dfetch(y_tex, node.y_index_1-1));

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


			if (threadIdx.x < 16)                 scratch [ threadIdx.x ]         += scratch [ threadIdx.x + 16 ];
			if (threadIdx.x < 8)                scratch [ threadIdx.x ]         += scratch [ threadIdx.x + 8 ];
			if (threadIdx.x < 4)                scratch [ threadIdx.x ]         += scratch [ threadIdx.x + 4 ];
			if (threadIdx.x < 2)                scratch [ threadIdx.x ]         += scratch [ threadIdx.x + 2 ];
			if (threadIdx.x < 1)                scratch [ threadIdx.x ]         += scratch [ threadIdx.x + 1 ];



			if (threadIdx.x == 0)
				d_Jp[blockIdx.x]         = scratch[0];
		}		

	template<typename T>
	__global__ void J_evaluate(T *d_Jp) {
		cooJacobian<T>::implementation(d_Jp);
	} // End of cooJacobian::k_evaluate		
} // End of namespace System		

