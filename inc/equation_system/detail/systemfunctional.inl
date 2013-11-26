
/* \file systemfunctional.inl
 * \brief Inline file for systemfunctional.h
 */


texture<float,1,cudaReadModeElementType> yTex;
texture<float,1,cudaReadModeElementType> kTex;


namespace System {
	template<typename T>
	SystemFunctional<T>
		::SystemFunctional(string filename) {

		ifstream ifile(filename);
		string value;
		bool success=false;

		while(getline(ifile,value)){
			success |= clientp::parse_csv(value.begin(), value.end(), k);
		}

		ifile.close();

		// abort;
		if (!success)
			throw std::invalid_argument( "loading of k data failed" );

		//buffer for equations arriving on stdin & line num
		string input;
		int line_no=0;

		//what we do and don't want on stdin
		string unsup[]={"cosh","sinh","tanh","cos","sin","tan"};
		string sup[]={"+","-"};


		//maximum elements in terms, in order to decide size of struct for node storage
		max_elements=INT_MIN;
		max_term_size=INT_MIN;

		while(getline(cin, input)){

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
				bool success = clientp::parse_constants(its->begin(), its->end(), constants);
				assert(constants.size());
				success |= clientp::parse_k_vals(its->begin(), its->end(), kInds);
				assert(kInds.size());
				success |= clientp::parse_y_vals(its->begin(), its->end(), tmp);



				if (!success)
					sad_little_message(line_no,*its);

				int sz = tmp.size();
				max_term_size = (sz > max_term_size) ? sz : max_term_size;

				yFull.push_back(tmp);
			}

			max_elements = (index > max_elements) ? index : max_elements;
			//tally terms in this equation
			terms.push_back(index);

			line_no++;
		}


		assert ((yFull.size()==constants.size()) &&( kInds.size()==constants.size()));

#ifdef __VERBOSE

		int displacement =0;

		for (int i=0; i<terms.size(); i++){


			cerr << "line : " << i << endl;
			int j_size=terms[i];

			for (int j=0; j<j_size; j++){

				int index = j+displacement;
				for (map<float,float>::iterator itm = yFull[index].begin(); itm != yFull[index].end(); itm++){
					cout << "term/y_ind/pwr: " << j << " " << itm->first << " " << itm->second << endl;

				}

			}
			cerr << "kInds" << endl;
			for (int j=0; j<j_size; j++){

				int index = j+displacement;
				cerr << kInds[index] << " ";

			}
			cerr << endl;
			cerr << "consts" << endl;
			for (int j=0; j<j_size; j++){

				int index = j+displacement;
				cerr << constants[index] <<  " ";

			}
			cerr << endl;



			displacement += terms[i];

		}

#endif

		// Initialization of the function based on the data gathered
		int num_leaves 	= constants.size();
		int num_funcs 	= terms.size();
		
		EvalNode *tmp_nodes 		= new EvalNode [ num_leaves ];
		int *tmp_terms 			= new int [ num_funcs ];
		T *tmp_y	 		= new T [ num_funcs ];

		for (int i=0; i<num_funcs; i++)
			tmp_terms[i] = terms[i];

		for (int i=0; i<num_funcs; i++){
			tmp_nodes[i].constant 	= constants[i];
			tmp_nodes[i].kIdx 	= (int) k_inds[i];
			tmp_nodes[i].yIdx1	= 0;
			tmp_nodes[i].yExp1	= 1.0;
			tmp_nodes[i].yIdx2	= 0;
			tmp_nodes[i].yExp2	= 1.0;
		
			map<T,T> tmp = yFull[i];

			tmp_nodes[i].yIdx1 	= (int) tmp.begin()->first;
			tmp_nodes[i].yExp1 	= tmp.begin()->second;
		
			if (tmp.size()>1){
			tmp_nodes[i].yIdx2 	= (int) (tmp.begin()++)->first;
			tmp_nodes[i].yExp2 	= (tmp.begin()++)->second;

			}
		}


		cudaMalloc((void**)&d_fNodes, sizeof(EvalNode)*num_leaves);

		//cudaCheckError("malloc, f_nodes_dev");
		cudaMemcpy(d_fNodes,tmp_nodes, sizeof(EvalNode)*num_leaves, cudaMemcpyHostToDevice);
		//cudaCheckError("memcpy, f_nodes_dev");
		cudaMalloc((void**)&d_fTerms, sizeof(int)*num_funcs);
		//cudaCheckError("malloc, terms_dev");
		cudaMemcpy(d_fTerms, tmp_terms, sizeof(int)*num_funcs, cudaMemcpyHostToDevice);
		//cudaCheckError("memcpy, terms_dev");
		
		// Create a new Vector object

		//cudaMalloc(function_dev, sizeof(float)*num_funcs);
		//cudaCheckError("malloc, function_dev");
		//cudaMalloc(delta, sizeof(float)*num_funcs);
		//cudaCheckError("malloc, delta");
		

		//cout << num_funcs << endl;
		//init y
		for (int i=0; i<num_funcs; i++)
			tmp_y[i] = 0.0f;

		cudaMalloc(y_dev, sizeof(T)*num_funcs);
		cout << "ydev 2 " << *y_dev << endl;
		//cudaCheckError("malloc, y_dev");
		cudaMemcpy(*y_dev, tmp_y, sizeof(T)*num_funcs, cudaMemcpyHostToDevice);
		//cudaCheckError("memcpy, y_dev");

		delete[] tmp_terms, tmp_nodes, tmp_y;

	} // End of SystemFunctional() 

	/*\XXX TODO Fix the textures !*/
	template<typename T>
	void SystemFunctional<T>
		::eval(const Vector<T> Y) {
			
		// Size of Y
		int nbEq = Y.size();
		


		// Bind textures	
//		cudaBindTexture(0,kTex, ,sizeof(T)*);
//		cudaBindTexture(0,yTex,Y.getDevMem(),sizeof(T)*nbEq);



		/*\XXX TODO Determine best nthreads and nblocks*/
		k_eval<T> <<< 1,1 >>> ();

//		cudaUnbindTexture(kTex);
//		cudaUnbindTexture(yTex);

	}	

	/*\XXX TODO Fix the name references (e.g. function_dev)*/
	template<typename T>
	__global__ void SystemFunctional<T>
		::k_eval(){

		__shared__ volatile T scratch[SHARED_BUF_SIZE];


		int index = blockIdx.x*blockDim.x + threadIdx.x;
		int terms_this_function = d_fTerms[blockIdx.x];

		T fnt = 0.0f;

		if (threadIdx.x<terms_this_function){
			EvalNode node = d_fNodes[index];

			fnt		=  node.constant;
			fnt		*= tex1Dfetch(k_tex, node.k_index-1);
			//zero based indexing
			fnt		*= pow(tex1Dfetch(y_tex, node.y_index_1-1),node.y_exp_1);	
			if (node.y_index_2 != -1)
				fnt		*= pow(tex1Dfetch(y_tex, node.y_index_2-1),node.y_exp_2);	


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

		//we solve J*delta = -f
		if (threadIdx.x == 0){
			function_dev[blockIdx.x] 	= -scratch[0];
		}
	}
}	
