/** 
 * @file systemfunctional.inl
 * @author Pierre-Yves Taunay (py.taunay@psu.edu)
 * @date November, 2013
 * @brief Implementation details of the methods of the class SystemFunctional
 *
 */
#include <iostream>
#include <fstream>

#include <cuda.h>
#include <cusp/array1d.h>

#include <equation_system/clientp.hpp>
#include <checks.cuh>

#define fpe(x) (isnan(x) || isinf(x))

/*\XXX TODO: Texture representation for doubles*/
texture<float,1,cudaReadModeElementType> yTex;
texture<float,1,cudaReadModeElementType> kTex;

#define SHARED_BUF_SIZE 256 

namespace System {
	using namespace clientp;
	using namespace std;
	template<typename T>
	SystemFunctional<T>
		::SystemFunctional(char *k_values,char *equations_file) {

		// LION-CODES: parser_main 
		std::ifstream ifile(k_values);
		string value;
		bool success=false;

		std::cout << "The file for the k values is ... " << k_values << std::endl;

		while(getline(ifile,value)){
			success |= clientp::parse_csv(value.begin(), value.end(), this->h_kData);
		}
		ifile.close();

		// abort;
		if (!success)
			throw std::invalid_argument( "loading of k data failed" );
		else
			std::cout << "Successfully read k values " << std::endl;

		//buffer for equations arriving on stdin & line num
		string input;
		int line_no=0;

		//what we do and don't want on stdin
		/*\XXX TODO: Add special functions to parser */
		string unsup[]={"cosh","sinh","tanh","cos","sin","tan"};
		string sup[]={"+","-"};


		//maximum elements in terms, in order to decide size of struct for node storage
		this->maxElements=INT_MIN;
		this->maxTermSize=INT_MIN;

		std::cout << "The file for the equations is ... " << equations_file << std::endl;
		ifile.open(equations_file,std::ifstream::in);
		while(getline(ifile, input)){

			unsigned count = 0;

			//c++0x
			//for (string &x : unsup)
			for (int i=0; i<6; i++)
				parse(input.c_str(),*(  str_p(unsup[i].c_str()) [ increment_a(count) ] | anychar_p ));

			// abort;
			if (count > 0){
				cerr << "Input line : " << line_no << endl; 
				cerr << "Received : " << input << endl; 
				throw std::invalid_argument( "Contains one or more unsupported functions \
						(*polynomials only please*) on stdin" );
			}


			count=0;
			//c++0x
			//for (string &x : sup)
			for (int i=0; i<2; i++)
				parse(input.c_str(),*(  str_p(sup[i].c_str()) [ increment_a(count) ] | anychar_p ));

			// abort;
			if (!(count > 0)){
				sad_little_message(line_no,input);
			}


			//tokenize just a little bit
			int index=0;
			string::iterator it;
			vector<string>::iterator its;
			string tmp_string="";
			vector<string> tokens;

			for ( it = input.begin() ; it < input.end(); it++,index++){

				if (((*it=='-') || (*it=='+')) && (index !=0)){
					tokens.push_back(tmp_string);
					tmp_string.clear();
					tmp_string+=*it;
				} else {

					tmp_string+=*it;
				}
			}

			tokens.push_back(tmp_string);

			index=0;
			int first_ele=0;
			float sign=1.0;

			for ( its = tokens.begin() ; its != tokens.end(); its++, index++){


				map<T,T> tmp;
				bool success = clientp::parse_constants(its->begin(), its->end(),this->constants);
				assert(this->constants.size());
				success |= clientp::parse_k_vals(its->begin(), its->end(), this->h_kInds);
				assert(this->h_kInds.size());
				success |= clientp::parse_y_vals(its->begin(), its->end(), tmp);



				if (!success)
					sad_little_message(line_no,*its);

				int sz = tmp.size();
				this->maxTermSize = (sz > this->maxTermSize) ? sz : this->maxTermSize;

				this->yFull.push_back(tmp);
			}

			this->maxElements = (index > this->maxElements) ? index : this->maxElements;
			//tally terms in this equation
			this->terms.push_back(index);

			line_no++;
		}
		ifile.close();


		assert ((this->yFull.size()==this->constants.size()) &&( this->h_kInds.size()==this->constants.size()));
		std::cout << "Succesfully read equations" << std::endl;

#ifdef __VERBOSE

		int displacement =0;

		for (int i=0; i<this->terms.size(); i++){


			cerr << "line : " << i << endl;
			int j_size=this->terms[i];

			for (int j=0; j<j_size; j++){

				int index = j+displacement;
				for (typename std::map<T,T>::iterator itm = this->yFull[index].begin(); itm != this->yFull[index].end(); itm++){
					cout << "term/y_ind/pwr: " << j << " " << itm->first << " " << itm->second << endl;

				}
			}

			cerr << "k_inds" << endl;
			for (int j=0; j<j_size; j++){

				int index = j+displacement;
				cerr << this->h_kInds[index] << " ";
			}
			cerr << endl;

			cerr << "consts" << endl;
			for (int j=0; j<j_size; j++){

				int index = j+displacement;
				cerr << this->constants[index] <<  " ";
			}
			cerr << endl;

			displacement += this->terms[i];
		}

#endif
		cerr << "max_term & max_term_size" << endl;
		cerr << this->maxElements << " " << this->maxTermSize<< endl;

		// LION-CODES: check_y
		std::map<T,T> indices;

		//check and possibly relabel y indices
		for (typename std::vector<std::map<T,T> >::iterator it = this->yFull.begin(); it != this->yFull.end(); it++){

			std::map<T,T> y_eqs = *it;

			for (typename std::map<T,T>::iterator it1 = y_eqs.begin(); it1 != y_eqs.end(); it1++){
				indices.insert(make_pair(it1->first,0));
			}
		}


		int index=1; bool reorder =false;
		for (typename map<T,T>::iterator it = indices.begin(); it != indices.end(); it++, index++){

			it->second = index;

			if (index != it->first){

				cerr << "CAUTION; remapping y index " << it->first << " to " << index << endl;
				reorder=true;
			}
		}

		if (reorder){
			for (typename std::vector<std::map<T,T> >::iterator it=this->yFull.begin(); it != this->yFull.end(); it++){


				std::map<T,T> eq_term = *it; //= indices[it1->first];
				std::map<T,T> new_term;
		
				for (typename std::map<T,T>::iterator it1=eq_term.begin(); it1!=eq_term.end(); it1++){
					
					T tmp = indices[it1->first];

					new_term[tmp] = it1->second;
				}
				*it = new_term;
			}
		}
		assert(this->maxTermSize<= 2);

		this->nbEq = this->terms.size();


		// LION-CODES: init_f
		// Initialization of the function based on the data gathered
		int num_leaves 	= this->constants.size();
		int num_funcs 	= this->terms.size();
		
		EvalNode<T> *tmp_nodes 		= new EvalNode<T> [ num_leaves ];
		int *tmp_terms 			= new int [ num_funcs ];
		T *tmp_y	 		= new T [ num_funcs ];
		int *tmp_offsets		= new int[num_funcs];

		tmp_offsets[0] = 0;
		int off = terms[0];

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
		
			std::map<T,T> tmp = yFull[i];

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

		

/*
		//cout << num_funcs << endl;
		//init y
		for (int i=0; i<num_funcs; i++)
			tmp_y[i] = 0.0f;

		cudaMalloc(y_dev, sizeof(T)*num_funcs);
		cout << "ydev 2 " << *y_dev << endl;
		//cudaCheckError("malloc, y_dev");
		cudaMemcpy(*y_dev, tmp_y, sizeof(T)*num_funcs, cudaMemcpyHostToDevice);
		//cudaCheckError("memcpy, y_dev");
*/

		delete[] tmp_terms, tmp_nodes, tmp_y, tmp_offsets;

	} // End of SystemFunctional() 


	template<typename T>
	__host__ void SystemFunctional<T>
		::evaluate(
			cusp::array1d<T,cusp::device_memory> &F,
			const cusp::array1d<T,cusp::device_memory> &Y) const {
			
		// Number of equations, and maximum numbers of terms
		int nbEq = this->nbEq;
		int mxTermsF = this->maxElements;

		// Size of k array
		int nbK = this->h_kData.size();
		
		// Get the device pointer
		const T *d_yp = thrust::raw_pointer_cast(Y.data());
		T *d_fp = thrust::raw_pointer_cast(F.data());
		T *d_kp = thrust::raw_pointer_cast(const_cast<SystemFunctional<T>*>(this)->d_kData.data());


		// Bind textures	
		cudaBindTexture(0,kTex,d_kp,sizeof(T)*nbK);
		cudaBindTexture(0,yTex,d_yp,sizeof(T)*nbEq);


		// Determine running configuration
		dim3 blocks_f, threads_f;
		blocks_f.x = nbEq;
		threads_f.x = ( mxTermsF < 32 ) ? 32 : ceil( (T)mxTermsF / ((T)32.0))*32;
		
		std::cout << "Starting the functional evaluation routine ... " << std::endl;

		// Run kernel
		k_FunctionalEvaluate<T> <<< blocks_f,threads_f >>> (d_fp,this->d_fNodes,this->d_fTerms,this->d_fOffsetTerms,d_kp,d_yp,this->nbEq);
		cudaThreadSynchronize();
		cudaCheckError("Error from the evaluation of the functional");

		cudaUnbindTexture(kTex);
		cudaUnbindTexture(yTex);

	} // End of SystemFunctional::evaluate()	


	/*\XXX TODO Fix the name references (e.g. function_dev)*/
	// Fields required for the kernel
	// d_fTerms
	// d_fNodes
	template<typename T>
	__global__ void 
		k_FunctionalEvaluate(
			T *d_fp,
			const EvalNode<T> *d_fNodes,
			const int *d_fTerms,
			const int *d_fOffsetTerms,
			T const* __restrict__ d_kp,
			T const* __restrict__ d_yp,
			int nbEq)	
		{

		__shared__ volatile T scratch[SHARED_BUF_SIZE];

		//could use constant mem here
		int index = d_fOffsetTerms[blockIdx.x];
		int terms_this_function = d_fTerms[blockIdx.x];

		T fnt = (T)0.0;

		if (threadIdx.x<terms_this_function){
			EvalNode<T> node = d_fNodes[index+threadIdx.x];

			fnt		=  node.constant;
		//	fnt		*= tex1Dfetch(kTex, node.kIdx-1);
			fnt		*= d_kp[node.kIdx-1];
			//zero based indexing
			//fnt		*= pow((T)tex1Dfetch(yTex, node.yIdx1-1),node.yExp1);	
			if (node.yIdx1 != 0) {
				fnt		*= pow( d_yp[node.yIdx1-1],node.yExp1);	
			} else {
				fnt *= 1.0;
			}	

			if (node.yIdx2 != -1)
				fnt		*= pow( d_yp[node.yIdx2-1],node.yExp2);	
			if( blockIdx.x == 42 || blockIdx.x == 291 || blockIdx.x == 292) {
			printf("b : %i t: %i c: %14.6e k: %i y1: %i e1: %f y2: %i e2: %f fnt : %14.6e tr : %14.6e y: %14.6e\n",\
			blockIdx.x,threadIdx.x,node.constant,node.kIdx,node.yIdx1,node.yExp1,node.yIdx2,\
				node.yExp2, fnt, d_kp[node.kIdx-1], d_yp[node.yIdx1-1]);
			}	
				//node.yExp2, fnt, tex1Dfetch(kTex,node.kIdx-1), tex1Dfetch(yTex, node.yIdx1-1));
		}

		scratch[threadIdx.x] = fnt;

		__syncthreads();

		if (blockDim.x >= 256){
			if (threadIdx.x < 128){
				scratch [ threadIdx.x ] 	+= scratch [ threadIdx.x + 128 ];
			}
			__syncthreads();
		}

		if (blockDim.x >= 128){
			if (threadIdx.x < 64){
				scratch [ threadIdx.x ] 	+= scratch [ threadIdx.x + 64 ];
			}
			__syncthreads();
		}

		if (blockDim.x >= 64){
			if (threadIdx.x < 32){
				scratch [ threadIdx.x ] 	+= scratch [ threadIdx.x + 32 ];
			}
		}


		if (threadIdx.x < 16) 		scratch [ threadIdx.x ] 	+= scratch [ threadIdx.x + 16 ];
		if (threadIdx.x < 8)		scratch [ threadIdx.x ] 	+= scratch [ threadIdx.x + 8 ];
		if (threadIdx.x < 4)		scratch [ threadIdx.x ] 	+= scratch [ threadIdx.x + 4 ];
		if (threadIdx.x < 2)		scratch [ threadIdx.x ] 	+= scratch [ threadIdx.x + 2 ];
		if (threadIdx.x < 1)		scratch [ threadIdx.x ] 	+= scratch [ threadIdx.x + 1 ];
/*
		if(threadIdx.x == 0 && blockIdx.x == 0) {
		
		//	printf("Number of equations = %d\n",nbEq);
			for(int i=0;i<nbEq;i++) {
				printf("NODE %d:\n",i);
				printf("Cst = %f\n",(d_fNodes)[i].constant);
				printf("kIdx = %f\n",d_fNodes[i].kIdx);
				printf("yIdx1 = %f\n",d_fNodes[i].yIdx1);
				printf("yIdx2 = %f\n",d_fNodes[i].yIdx2);
				printf("yExp1 = %f\n",d_fNodes[i].yExp1);
				printf("yExp2 = %f\n",d_fNodes[i].yExp2);
			}
		}	
*/
		if (threadIdx.x == 0){
			d_fp[blockIdx.x] 	= scratch[0];
		}
	} // End of SystemFunctional::k_evaluate()
}	
