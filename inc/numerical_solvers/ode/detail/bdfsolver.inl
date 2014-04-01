/**
 * @file bdfsolver.inl
 * @author Pierre-Yves Taunay (py.taunay@psu.edu)
 * @date March 2014
 * @brief Implementation of the methods of the class BDF solver
 *
 */

///// CUDA
#include <cuda.h>

namespace NumericalSolver {
	
	template<typename T>
	struct scalar_inv_functor : public thrust::binary_function<T,T,T>
	{
		const T a;

		scalar_inv_functor(T _a): a(_a) {}

		__host__ __device__
		T operator()(const float&x) const {
			return a/x;
		}
	};	
	


	// Constructors: allocate memory on GPU, etc...
	template<typename T>
	BDFsolver<T>::
		BDFsolver() {
			
			this->nist = 0;
			this->N = 1;
			this->neq = 1;
			this->dt = (T)1e-3;
			this->t = (T)1.0;
			this->q = 5;
			this->tol = 1e-6;
			
			

			// Allocate Nordsieck array: ZN = [ N x L ], for each ODE. L = Q+1
			cudaMalloc( (void**)&this->d_ZN,sizeof(T)*constants::L_MAX*N);
			// L polynomial for correction. l = [ 1 x L ], for each ODE
			cudaMalloc( (void**)&this->d_lpoly,sizeof(T)*constants::L_MAX );
			// Create the CUBLAS handle
			cublasCreate(&handle);

//			for(int i = 0; i< constants:: L_MAX; i++)
//				lpolyColumns.push_back( thrust::device_pointer_cast(&d_lpoly[i*neq]));
			lpolyColumns = thrust::device_pointer_cast(d_lpoly);

			// Create previous q+1 successful step sizes
			cudaMalloc( (void**)&this->d_pdt,sizeof(T)*constants::L_MAX);
			cudaMalloc( (void**)&this->d_dtSum,sizeof(T));
			cudaMalloc( (void**)&this->d_xiInv,sizeof(T));

		}

	//etc		
	

	// Compute
	template<typename T>
	void BDFsolver<T>::
		compute(const SystemFunctional<T> &F,
			const cooJacobian<T> &J,
			cusp::array1d<T,cusp::device_memory> &Fv,
			cusp::coo_matrix<int,T,cusp::device_memory> &Jv,
			cusp::array1d<T,cusp::device_memory> &d,
			cusp::array1d<T,cusp::device_memory> &Y
		) {

			// Reset and check weights
			// Check for too many time steps
			// Check for too much accuracy
			// Check for h below roundoff
			//

			/*
			 * Take a step
			 */

			//// 1. Make a prediction
			// 1.1 Update current time
			this->t += this->dt;
			// 1.2 Apply Nordsieck prediction : ZN_0 = ZN_n-1 * A(q)
			// Use the logic from CVODE
			for(int k = 1; k <= q; k++) { 
				for(int j = q; j >= k; j--) {
					// ZN[j-1] = ZN[j] + ZN[j-1]
					// Use CUBLAS
					axpyWrapper(handle, N, (const T*)(&constants::ONE) , &d_ZN[j], 1, &d_ZN[j-1],1);
				}
			}	
							

			//// 2. Calculate L polynomial and other data
			// L[0] = L[1] = 1; L[i>=2] = 0
			/*
			for( int i = 0; i<lpolyColumns.size(); i++) { 
				if( i < 2 ) {
					thrust::fill(lpolyColumns.at(i), lpolyColumns.at(i) + neq, (T)1.0);
				} else {
					thrust::fill(lpolyColumns.at(i), lpolyColumns.at(i) + neq, (T)0.0);
				}	
			}*/	
			for(int i = 0; i < constants::L_MAX; i++) {
				if( i < 2 ) {
					lpolyColumns[i] = (T) 1.0;
				} else {	
					lpolyColumns[i] = (T) 0.0;
				}	
			}	

			thrust::device_ptr<T> dptr_dtSum = thrust::device_pointer_cast(d_dtSum);
			thrust::device_ptr<T> dptr_xiInv = thrust::device_pointer_cast(d_xiInv);
			thrust::fill(dptr_dtSum,dptr_dtSum+neq,dt);

			if( q > 1 ) {
				for(int j = 2; j < q ; j++) {
					// hsum <- hsum + tau[j-1]
					axpyWrapper(handle,1,(const T*)(&constants::ONE), &d_pdt[j-2], 1, d_dtSum, 1);
					
					// xi_inv <- h/hsum
					thrust::transform(	dptr_dtSum,
								dptr_dtSum + 1, 
								dptr_xiInv,
								scalar_inv_functor<T>(dt)
							);

					for(int i = j; i>=1; i--) {
						// l[i] <- l[i] + l[i-1] * xi_inv
						// For multiple (read large amount of) equations, just use zip iterators
						lpolyColumns[i] = lpolyColumns[i] + lpolyColumns[i-1] * dptr_xiInv[0];	
					}

				}
				// TODO j = q
					
			}

			// gamma
			


			// 3. Non linear solver



			// 4. Check results


			/*
			 * Complete step
			 */

			//// 1. Update data
			this->nist++;

			//// 2. Apply correction

			//// 3. Manage order q

			// Prepare next step

		}

	template<>
	void axpyWrapper<float>(cublasHandle_t handle,
			int n,
			const float *alpha,
			const float *X,int incx,
			float *Y, int incy) {
				cublasSaxpy(handle,n,alpha,X,incx,Y,incy);
			}	

	template<>
	void axpyWrapper<double>(cublasHandle_t handle,
			int n,
			const double *alpha,
			const double *X,int incx,
			double *Y, int incy) {
				cublasDaxpy(handle,n,alpha,X,incx,Y,incy);
			}	
}
