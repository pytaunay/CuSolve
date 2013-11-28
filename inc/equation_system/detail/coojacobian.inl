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

	/*
	void jacobian(vector<float>& k_inds,                         //input: the k indices for equations
			vector<float>& constants,                         //input:         all constant/sign data
			vector<map<float,float> >& y_complete,                 //input:         all y data (yindex(key) & power(value))
			vector<int>& terms,                                 //input:         the number of terms per function evaluation
			vector<int>& terms_jac,                         //output:         the number of terms per jacobian i,j evaluation
			vector<int>& index_i,                                //output:         all the Jocobian row indices
			vector<int>& index_j,                                //output:        all the non-zero Jacobian column indices
			vector<float>& k_inds_jac,                         //output:        the k indices for each jac term
			vector<float>& constants_jac,                         //output:        all constant/sign data for each jac term
			vector<map<float,float> >& jac_complete,         //output:        all y data (yindex(key) & power(value)) for each jac term
			int& max_elements){                                //output:         the maximum number of terms in any i,j evaluation
	*/
		int displacement=0;

		this->maxElements = -1;

		for (int i=0; i<terms_F.size(); i++){


			int j_size=terms_F[i];

			//gather the total y for an equation (key), don't care about vals
			//we use a std::map, exploiting insertion order
			map<float,vector<int> > tmp;

			for (int j=0; j<j_size; j++){

				int index = j+displacement;
				for (map<float,float>::iterator itm = y_complete[index].begin(); itm != y_complete[index].end(); itm++){
					tmp[itm->first].push_back(j);

				}

			}

			// these are all the valid j indices for this row (i) in jacobian
			for (map<float,vector<int> >::iterator itmv = tmp.begin(); itmv != tmp.end(); itmv++){
				//convert 1 based index to 0
				idxJ.push_back((int) itmv->first -1);
				idxI.push_back(i);
			}

			//each GPU block is going to eval one i,j in jacobian
			//so as with function evaluation we'll store the number of terms in i,j evaluation
			//and thus a block will know how many nodes/structs to load

			for (map<float,vector<int> >::iterator itmv = tmp.begin(); itmv != tmp.end(); itmv++){

				//find this y in all the maps (one map per term)
				float y = itmv->first;
				int jac_terms=0;
				for (int j=0; j<j_size; j++){

					int index = j+displacement;
					bool found=false;
					map<float,float> relevant_y;
					float pow = 1.0f;

					for (map<float,float>::iterator itm = y_complete[index].begin(); itm != y_complete[index].end(); itm++){

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
						if (pow = 1.0f)
							constants.push_back(constants_F[index]);
						else

							constants.push_back(constants_F[index]*pow);


						kInds.push_back(k_inds_F[index]);

						if (pow != 1.0f){
							relevant_y[y]=pow-1;

						}

						jFull.push_back(relevant_y);                        


					}

				}


				maxElements = (jac_terms > maxElements) ? jac_terms : maxElements;
				terms.push_back(jac_terms);
			}

			displacement += terms_F[i];

		}

		assert(terms.size()==index_j.size());
		//
		//

	#ifdef __VERBOSE

		displacement =0;

		for (int i=0; i<terms.size(); i++){


			cerr << "jac element : " << i << endl;
			cerr << "indices : " << idxI[i] << " " << idxJ[i] << endl;
			int j_size=terms[i];

			for (int j=0; j<j_size; j++){

				int index = j+displacement;
				for (map<float,float>::iterator itm = jFull[index].begin(); itm != jFull[index].end(); itm++){
					cout << "term/y_ind/pwr: " << j << " " << itm->first << " " << itm->second << endl;

				}

			}
			cerr << "k_inds" << endl;
			for (int j=0; j<j_size; j++){

				int index = j+displacement;
				cerr << kInds[index] << " ";

			}
			cerr << endl;
			cerr << "consts" << endl;
			for (int j=0; j<j_size; j++){

				int index = j+displacement;
				cerr << constants[index] << " ";

			}
			cerr << endl;

			displacement += terms[i];

		}

	#endif

	}
			// Jacobian
			int num_leaves         = constants.size();
			int num_funcs         = terms_F.size();

			EvalNode* tmp_nodes         = new EvalNode[ num_leaves ];
			int * tmp_terms                 = new int[ num_funcs ];
			int * tmp_offsets                 = new int[ num_funcs ];

			tmp_offsets[0]        = 0;
			int off         = terms[0];

			for (int i=1; i<num_funcs; i++){
				tmp_offsets[i] = off;
				off+=terms[i];
			}

			for (int i=0; i<num_funcs; i++)
				tmp_terms[i] = terms[i];

			for (int i=0; i<num_leaves; i++){
				tmp_nodes[i].constant     = constants[i];
				tmp_nodes[i].kIdx         = (int) kInds[i];
				tmp_nodes[i].yIdx1        = 0;
				tmp_nodes[i].yExp1        = 1.0;
				tmp_nodes[i].yIdx2        = -1;
				tmp_nodes[i].yExp2        = 1.0;

				map<T,T> tmp = jFull[i];

				tmp_nodes[i].yIdx1         = (int) tmp.begin()->first;
				tmp_nodes[i].yExp1         = tmp.begin()->second;

				if (tmp.size()>1){
					map<T,T>:: iterator it = tmp.begin();
					it++;
					tmp_nodes[i].yIdx2         = (int) it->first;
					tmp_nodes[i].yExp2         = it->second;
				}
			}
			cudaMalloc((void**)&d_jNodes,sizeof(EvalNode)*num_leaves);
		//	cudaCheckError("malloc, jac_nodes_dev");
			cudaMemcpy(d_jNodes,tmp_nodes,sizeof(EvalNode)*num_leaves, cudaMemcpyHostToDevice);
			//cudaCheckError("memcpy, jac_nodes_dev");

			cudaMalloc((void**)&d_jTerms, sizeof(int)*num_funcs);
			cudaMalloc((void**)&d_jOffsetTerms, sizeof(int)*num_funcs);
			//cudaCheckError("malloc, terms_jac_dev");
			cudaMemcpy(d_jTerms, tmp_terms, sizeof(int)*num_funcs, cudaMemcpyHostToDevice);
			cudaMemcpy(d_jOffsetTerms, tmp_offsets, sizeof(int)*num_funcs, cudaMemcpyHostToDevice);
			//cudaCheckError("memcpy, terms_jac_dev");


			delete[] tmp_terms, tmp_nodes, tmp_offsets;
	} // End of cooJacobian()


	template<typename T>
	__host__ void cooJacobian<T>
				::evaluate(
					cusp::coo_matrix<T,cusp::device_memory> &J,
					const cusp::array1d<T,cusp::device_memory> &Y) {
		
		// Get the pointer to the data
		// J.values = array1d inside the coo_matrix
		T *d_Jp = thrust::raw_pointer_cast(J.values.data());


		/*\XXX TODO: Determine nthreads, nblocks*/
		/*\XXX TODO: Fix textures*/
		k_evaluate<<<1,1>>>(d_Jp);

	} // End of cooJacobian::evaluate


	template<typename T>
	__global__ void cooJacobian<T>
				::k_evaluate(T *d_Jp) {
		__shared__ volatile T scratch[SHARED_BUF_SIZE];

		// Could use constant mem here
		int index = d_jOffsetTerms[blockIdx.x];
		int terms_this_function = d_jTerms[blockIdx.x];
		T fnt = 0.0f;

		if (threadIdx.x<terms_this_function){

			EvalNode node = d_jNodes[index+threadIdx.x];

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

	} // End of cooJacobian::k_evaluate		
} // End of namespace System		

