/**
 * @file bdfsolver.inl
 * @author Pierre-Yves Taunay (py.taunay@psu.edu)
 * @date March 2014
 * @brief Implementation of the methods of the class BDF solver
 *
 */

///// STD
#include <algorithm>
#include <cmath>

///// CUDA
#include <cuda.h>
// Thrust
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>


///// CuSolve
#include <equation_system/bdffunctional.h>
#include <equation_system/bdfcoojacobian.h>

namespace NumericalSolver {
	
	// Static members
	/*
	template<typename T>
	static const T BDFsolver<T>::DT_LB_FACTOR = (T)100.0;
	template<typename T>
	static const T BDFsolver<T>::DT_UB_FACTOR = (T)0.1;
	template<typename T>
	static const int BDFsolver<T>::QMAX = 5;
	template<typename T>
	static const int BDFsolver<T>::LMAX=BDFsolver<T>::QMAX + 1;
	template<typename T>
	static const T BDFsolver<T>::ONE = (T)1.0; 
	template<typename T>
	static const T BDFsolver<T>::TWO = (T)2.0; 
	template<typename T>
	static const int BDFsolver<T>::MAX_DT_ITER = 4; 
	*/

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

	template<typename T>
	struct square : public thrust::binary_function<T,T,T>
	{
		__host__ __device__
		T operator()(const T &x) {
			return x*x;
		}
	};	


	template<typename T>
	struct scalar_functor : public thrust::binary_function<T,T,T>
	{
		const T a;
		scalar_functor(T _a) : a(_a) {}

		__host__ __device__
		T operator()(const T &x) const {
			return a*x;
		}
	};	
	
	template<typename T>
	struct dt0_upper_bound : public thrust::unary_function<thrust::tuple<T,T,T>,T>
	{
		const T relTol;
		dt0_upper_bound(T _relTol) : relTol(_relTol) {}

		__host__ __device__
		T operator()(const thrust::tuple<T,T,T>& t) const {
			return (abs ( thrust::get<1>(t) ) / ( (relTol+0.1) * abs(thrust::get<0>(t)) + thrust::get<2>(t)));
		}
	};	

	template<typename T>
	struct eval_weights_functor : public thrust::unary_function<thrust::tuple<T,T>,T>
	{
		const T relTol;
		eval_weights_functor(T _relTol) : relTol(_relTol) {}
			
		__host__ __device__
		T operator()(const thrust::tuple<T,T>& t) const {
			return ( (T)1.0 / ( relTol * abs( thrust::get<0>(t) ) + thrust::get<1>(t) ) );
		}
	};	

	template<typename T>
	struct weighted_rms_functor : public thrust::unary_function<thrust::tuple<T,T>,T>
	{
		__host__ __device__
		T operator()(const thrust::tuple<T,T>&t) const {
			return ( thrust::get<0>(t)*thrust::get<0>(t)*thrust::get<1>(t)*thrust::get<1>(t) );
		}
	};	
			


	// Constructors: allocate memory on GPU, etc...
	template<typename T>
	BDFsolver<T>::
		BDFsolver(const SystemFunctional<T> &F, const cooJacobian<T> &J, NonLinearSolver<T> *nlsolve, const cusp::array1d<T,cusp::device_memory> &Y0, const cusp::array1d<T,cusp::device_memory> &absTol) {
			
			this->nist = 0;
			this->N = 1;
			//this->neq = 1;
			this->dt = (T)1e-3;
			this->t = (T)0.0;
			this->q = 5;
			this->relTol = 1e-4;

			this->nEq = F.getTerms().size();
			
			// Allocate Nordsieck array: ZN = [ N x L ], for each ODE. L = Q+1
			cudaMalloc( (void**)&this->d_ZN,sizeof(T)*BDFsolver<T>::LMAX()*N);
			// L polynomial for correction. l = [ 1 x L ], for each ODE
			cudaMalloc( (void**)&this->d_lpoly,sizeof(T)*BDFsolver<T>::LMAX() );
			// Create the CUBLAS handle
			cublasCreate(&handle);


//			for(int i = 0; i< constants:: L_MAX; i++)
//				lpolyColumns.push_back( thrust::device_pointer_cast(&d_lpoly[i*neq]));
			lpolyColumns = thrust::device_pointer_cast(d_lpoly);

			// Create previous q+1 successful step sizes
			cudaMalloc( (void**)&this->d_pdt,sizeof(T)*BDFsolver<T>::LMAX());
			cudaMalloc( (void**)&this->d_dtSum,sizeof(T));
			cudaMalloc( (void**)&this->d_xiInv,sizeof(T));

			//// Create G and H based on F and J
			// G(U) = (U - YN0) - GAM*(F(U,t) - YdN0) ; GAM = h/l1
			// G(U) = (U - YN0) - GAM*F(U,t) + ZN1/l1
			// H = I - GAM*J
			this->G = new BDFfunctional<T>(F);
			this->H = new BDFcooJacobian<T>(J,this->nEq);
			this->nlsolve = nlsolve; 

			YTMP.resize(this->nEq,(T)0.0);


			//// Initialization of the Nordsieck array
			// FTMP = YN0_dot
			this->dptr_ZN = thrust::device_pointer_cast(d_ZN);
			cusp::array1d<T,cusp::device_memory> FTMP(this->nEq);
			F.evaluate(FTMP,Y0);
			
			thrust::copy(FTMP.begin(),FTMP.end(),this->dptr_ZN + this->nEq);
			
			// ZN[0] = YN0
			thrust::copy(Y0.begin(),Y0.end(),this->dptr_ZN);

			//// Initialization of the integration tolerances
			cudaMalloc((void**)&this->d_absTol,sizeof(T)*this->nEq);
			this->dptr_absTol = thrust::device_pointer_cast(d_absTol);
			thrust::copy(absTol.begin(),absTol.end(),this->dptr_absTol);

			//// Weights
			cudaMalloc((void**)&this->d_weight,sizeof(T)*this->nEq);
			this->dptr_weight = thrust::device_pointer_cast(d_weight);
			evalWeights(Y0,absTol,dptr_weight);

		}


	// Compute
	template<typename T>
	void BDFsolver<T>::
		compute(const SystemFunctional<T> &F,
			const cooJacobian<T> &J,
			cusp::array1d<T,cusp::device_memory> &Fv,
			cusp::coo_matrix<int,T,cusp::device_memory> &Jv,
			cusp::array1d<T,cusp::device_memory> &d,
			cusp::array1d<T,cusp::device_memory> &Y,
			const T &tmax
		) {
			// Thrust device pointers for convenience
			thrust::device_ptr<T> dptr_dtSum = thrust::device_pointer_cast(d_dtSum);
			thrust::device_ptr<T> dptr_xiInv = thrust::device_pointer_cast(d_xiInv);
			thrust::device_ptr<T> dptr_pdt = thrust::device_pointer_cast(d_pdt);


			if(this->nist == 0) {

				// Initialize the time step
				initializeTimeStep(tmax,F);

				// Scale zn[1] by the new time step
				thrust::transform( dptr_ZN + this->nEq, dptr_ZN + 2*this->nEq, dptr_ZN+this->nEq,scalar_functor<T>(this->dt));




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
			for(int i = 0; i < BDFsolver<T>::LMAX(); i++) {
				if( i < 2 ) {
					lpolyColumns[i] = (T) 1.0;
				} else {	
					lpolyColumns[i] = (T) 0.0;
				}	
			}	

			//thrust::fill(dptr_dtSum,dptr_dtSum+neq,dt);

			T alpha0 = (T)(-1.0);
			T dtSum = dt;
			T xistar_inv = (T)1.0;
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

					alpha0 -= 1.0/(T)j;

					for(int i = j; i>=1; i--) {
						// l[i] <- l[i] + l[i-1] * xi_inv
						// For multiple (read large amount of) equations, just use zip iterators
						lpolyColumns[i] = lpolyColumns[i] + lpolyColumns[i-1] * dptr_xiInv[0];	
					}

				}
				
				// j = q
				alpha0 -= 1.0/(T)q;
				xistar_inv = -lpolyColumns[1]-alpha0;
				dtSum += d_pdt[q-2];
				xistar_inv = dt/dtSum;
				for(int i = q; i>=1; i--) {
					// l[i] <- l[i] + l[i-1] * xi_inv
					// For multiple (read large amount of) equations, just use zip iterators
					lpolyColumns[i] = lpolyColumns[i] + lpolyColumns[i-1] * xistar_inv;	
				}



					
			}

			// gamma = h/l1
			T gamma = dt/lpolyColumns[1]; 


			// 3. Non linear solver
			// Set delta to 0
			thrust::fill(d.begin(),d.end(),(T)0.0);	
			// Copy Nordsieck array in Y	
			// The first N values correspond to Yn predicted, Yn0
			thrust::copy(this->dptr_ZN, this->dptr_ZN + N, Y.begin());

			// Set constants in G and H	
			thrust::transform(this->dptr_ZN + N, this->dptr_ZN + 2*N, YTMP.begin(),scalar_inv_functor<T>(lpolyColumns[1])); // YTMP = ZN[1] / l1 
			thrust::transform(YTMP.begin(),YTMP.end(),this->dptr_ZN,YTMP.begin(),thrust::minus<T>()); // YTMP = YTMP - ZN[0]
		//	this->G->setConstants(gamma,YTMP);
		//	this->H->setConstants(gamma);

			// Call non linear solver
			//this->nlsolve->compute(this->G,this->H,Gv,Hv,d,Y);

			// 4. Check results
			// TODO: use exceptions here from 3.


			/*
			 * Complete step
			 */
			//// 1. Update data
			// Increment total step number
			this->nist++;
			// Update data from previous successful steps "tau"
			for(int i = q; i >= 2; i--) 
				dptr_pdt[i-1] = dptr_pdt[i-2]; 
			if( (q == 1) && (nist > 1) )
				dptr_pdt[1] = dptr_pdt[0];
			dptr_pdt[0] = dt;	

			//// 2. Apply correction
			// ZN[j] = ZN[j] + en * l[j], w/ en being Yn - Yn0
			T *tmp_ptr = thrust::raw_pointer_cast(d.data());
			T pj;
			for(int j = 0; j <= q; j++) { 
				pj = lpolyColumns[j];
				axpyWrapper(	handle, 
						N, 
						&pj, 
						tmp_ptr, 1,
						d_ZN, 1); 
			}			
			

			/*
			 * Prepare next step
			 */

			//// 1. Manage order q

		}

	// See http://sundials.wikidot.com/cvode-init-stepsize 
	template<typename T>
	void BDFsolver<T>
		::initializeTimeStep(const T tout, const SystemFunctional<T> &F) {
			
			int nEq = F.getTerms().size();

			short sgn = ( tout - t > 0.0 ) ? 1 : -1;
			T tround = std::numeric_limits<T>::epsilon()*std::max( abs(this->t), abs(tout));

			//// Define the lower bound and the upper bound
			T dt_lb, dt_ub, dt_geom;

			// Lower bound: h_L = 100*epsilon*max(|t0|,|tout|)
			dt_lb = BDFsolver<T>::DT_LB_FACTOR() * tround;

			// Upper bound: h_U = 0.1*|tout-t0|
			// In CVODE, h_U is adjusted so that h_U * |ydot0| < 0.1*|y0| + atol for all components
			// h_U is chosen as the minimum between the adjusted value of h_U, and FAC*|t-tout|
			dt_ub = BDFsolver<T>::DT_UB_FACTOR() * abs(this->t - tout); 

			// Adjusted h_U: 1/h_U = max (  | zn[1] | / (FAC*|zn[0]| + relTol*|zn[0]| + absTol) )
			T tmp = thrust::transform_reduce(
				thrust::make_zip_iterator(thrust::make_tuple(dptr_ZN, dptr_ZN + nEq, dptr_absTol)),
				thrust::make_zip_iterator(thrust::make_tuple(dptr_ZN + nEq, dptr_ZN + 2*nEq, dptr_absTol + nEq)),
				dt0_upper_bound<T>(this->relTol),
				(T)0,
				thrust::maximum<T>());
			
			dt_ub = min( (T)1.0/tmp, dt_ub );


			dt_geom = sqrt( dt_lb * dt_ub ); 	

			if( dt_ub < dt_lb ) {
				this->dt = (T)sgn * dt_geom;
				return;
			}	

			// FIXME CUSP array to interoperate with our temporary array
			cusp::array1d<T,cusp::device_memory> FTMP(nEq);
			cusp::array1d<T,cusp::device_memory> HOLDERTMP(nEq);

			T yddrms, dt_save,dt_ratio,dt_new = 0.0;
			bool ACCEPT=false;
			// Try MAX_DT_ITER times to find a "correct" h
			for(int i=0; i < BDFsolver<T>::MAX_DT_ITER(); i++) {
				// If stopping critera
				if( ACCEPT ) {
					dt_new = dt_geom;
					break;
				} else {	
					//// Calculate the RMS norm of Ydd
					// Ydd is approximated by ( F(t + dt_geom, Y0+ dt_geom*Yd0) - Yd0 )/dt_geom
					// dt_geom*Yd0
					thrust::transform(dptr_ZN+nEq,dptr_ZN+2*nEq,HOLDERTMP.begin(),scalar_functor<T>(dt_geom));
					// Y0 + dt_geom*Yd0
					thrust::transform(dptr_ZN,dptr_ZN+nEq,HOLDERTMP.begin(),HOLDERTMP.begin(),thrust::plus<T>());
					F.evaluate(FTMP,HOLDERTMP);
					// F - Yd0
					thrust::transform(FTMP.begin(),FTMP.end(),dptr_ZN+nEq, YTMP.begin(), thrust::minus<T>());
					
					// Transform reduce w/ RMS 
					// RMS = sqrt ( 1/N * sum( (Y*e)^2 ) ), where Y = vector, e = error weight
					thrust::copy(YTMP.begin(),YTMP.end(),HOLDERTMP.begin());
					yddrms = weightedRMSnorm(HOLDERTMP,this->dptr_weight);

					// Divide by dt_geom
					yddrms /= (dt_geom);

					//// Save proposed step size
					dt_save = dt_geom;

					//// Propose a new step size
					dt_new = (yddrms*dt_ub*dt_ub > 2.0) ? sqrt(2.0/yddrms) : sqrt(dt_geom*dt_ub);  

					//// Calculate ratio
					dt_ratio = dt_new/dt_geom;
					if ( dt_ratio > 0.5 && dt_ratio < 2.0) {
						ACCEPT=true;
					} 
					if( i > 0 && dt_ratio > 2.0 ) {
						dt_new = dt_geom;
						ACCEPT=true;
					}	

					//// Send value back into F to calculate RMS again
					dt_geom = dt_new;
				}	
			}

			//// Apply bounds, bias factor (0.5), and sign
			dt_geom = 0.5*dt_new;
			if(dt_geom < dt_lb) { 
				dt_geom = dt_lb;
			} else if (dt_geom > dt_ub) {
				dt_geom = dt_ub;
			}
			dt_geom *= (T)sgn;

			// Save time step
			this->dt = dt_geom;
		}
	template<typename T>
	void BDFsolver<T>
		::evalWeights(
			const cusp::array1d<T,cusp::device_memory> &Y,
			const cusp::array1d<T,cusp::device_memory> &absTol,
			thrust::device_ptr<T> dptr_w) {

			thrust::transform(
				thrust::make_zip_iterator(thrust::make_tuple(Y.begin(),absTol.begin())),
				thrust::make_zip_iterator(thrust::make_tuple(Y.end(),absTol.end())),
				dptr_w,
				eval_weights_functor<T>(this->relTol));
			}

	template<typename T>
	T BDFsolver<T>
		::weightedRMSnorm(	
			cusp::array1d<T,cusp::device_memory> &Y,
			const thrust::device_ptr<T> dptr_w) {
				T tmp;

				// RMS = sqrt ( 1/N * sum( (Y*e)^2 ) ), where Y = vector, e = error weight
				tmp = thrust::transform_reduce(
					thrust::make_zip_iterator(thrust::make_tuple(Y.begin(),dptr_w)),
					thrust::make_zip_iterator(thrust::make_tuple(Y.end(),dptr_w+Y.size())),
					weighted_rms_functor<T>(),
					(T)0.0,
					thrust::plus<T>());
				
				tmp /= Y.size();
				tmp = sqrt(tmp);

				return tmp;
			}


/*
	template<typename T>
	T BDFsolver<T>
		::upperBoundFirstTimeStep(int nEq) {

			T test;
			test = thrust::transform_reduce(
				thrust::make_zip_iterator(thrust::make_tuple(dptr_ZN, dptr_ZN + nEq, dptr_absTol)),
				thrust::make_zip_iterator(thrust::make_tuple(dptr_ZN + nEq, dptr_ZN + 2*nEq, dptr_absTol + nEq)),
				dt0_upper_bound<T>(this->relTol),
				(T)0,
				thrust::maximum<T>());
			

							
			return test;				
			//dt_ub = BDFsolver<T>::DT_UB_FACTOR() * abs(this->t - tout); 

		}
*/

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
