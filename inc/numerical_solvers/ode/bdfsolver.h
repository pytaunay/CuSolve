/**
 * @file bdfodesolver.h
 * @author Pierre-Yves Taunay (py.taunay@psu.edu)
 * @date March, 2014
 * @brief Class representation of a backwards differentiation formula (BDF) solver
 *
 */

#pragma once

//// CUDA
#include <cuda.h>
// CUBLAS
#include <cublas_v2.h>


#include <numerical_solvers/ode/odesolver.h>

#include <equation_system/systemjacobian.h>
#include <equation_system/systemfunctional.h>


namespace NumericalSolver {

	namespace constants {
		const double ONE = 1.0;
	}	


	template<typename T>
	class BDFsolver : public ImplicitODESolver, LMMODESolver {


		private:
			//BDFfunctional<T> *G; /*!< Modified functional for the non linear solver in the BDF method: \f$G(u) = (u - ZN^{p}_{0}) - \gamma[F(u)-ZN^{p}_{1})\f$ */	
			SystemFunctional<T> *G; /*!< Modified functional for the non linear solver in the BDF method: \f$G(u) = (u - ZN^{p}_{0}) - \gamma[F(u)-ZN^{p}_{1})\f$ */	
			SystemJacobian<T> *H; /*!< Modified Jacobian for the non linear solver in the BDF method: \f$ H(u) = I -\gamma J	$\f */	

		protected:
			NonLinearSolver<T> *nlsolve; /*!< Non linear solver */
			T relTol; /*!< Tolerance */

			short q; /*!< BDF order */
			T dt; /*!< Current time step */
			T t; /*!< Current time */


			int nist; /*!< Number of internal steps taken */
			int N; /*!< Size of the system of ODE */
			int nEq; /*!< Number of equations in the system */

				

			// Stored on device
			// Nordsieck array -- should be invisible for user
			T *d_ZN; /*!< Nordsieck array */

			// L polynomial -- should be invisible for user
			T *d_lpoly; /*!< L polynomial for updates */
			//std::vector< thrust::device_ptr<T> > lpolyColumns; /*!< Array of thrust device pointers pointing to location in memory of columns of d_lpoly */

			thrust::device_ptr<T> lpolyColumns; /*!< Thrust device wrapper for d_lpoly */

			// CUBLAS handle
			cublasHandle_t handle; /*!< CUBLAS handle */

			T *d_pdt; /*!< Previous q+1 successful step sizes */
			T *d_dtSum; /*!< Used to build the L polynomial */
			T *d_xiInv; /*!< Used to build the L polynomial */
			T *d_absTol; /*!< Absolute tolerance */
			T *d_weight; /*!< Weights for RMS calculations */


			thrust::device_vector<T> YTMP; /*!< Holder for temporary operations */
			thrust::device_ptr<T> dptr_ZN; /*!< Thrust device wrapper for the Nordsieck array */
			thrust::device_ptr<T> dptr_absTol; /*!< Thrust wrapper for absolute tolerances */
			thrust::device_ptr<T> dptr_weight; 

			static const T DT_LB_FACTOR() { return 100.0; }
			static const T DT_UB_FACTOR() { return 0.1; } 
			// Maximum order for the BDF
			static const int QMAX() { return (int)5; } 
			static const int LMAX() { return (QMAX()+1);}
			static const int MAX_DT_ITER() { return 4; }

		public:
			BDFsolver();
		//	BDFsolver(const SystemFunctional<T> &F, const cooJacobian<T> &J); 
			BDFsolver(const SystemFunctional<T> &F, const cooJacobian<T> &J, NonLinearSolver<T> *nlsolve, const cusp::array1d<T,cusp::device_memory> &Y0,const cusp::array1d<T,cusp::device_memory> &absTol); 
			~BDFsolver() {
				delete G;
				delete H;
				delete nlsolve;
			}	


		void compute(const SystemFunctional<T> &F,
			const cooJacobian<T> &J,
			cusp::array1d<T,cusp::device_memory> &Fv,
			cusp::coo_matrix<int,T,cusp::device_memory> &Jv,
			cusp::array1d<T,cusp::device_memory> &d,
			cusp::array1d<T,cusp::device_memory> &Y,
			const T &tmax);

		private:
			void initializeTimeStep(const T tmax, const SystemFunctional<T> &F);
			T upperBoundFirstTimeStep(int nEq);

			void evalWeights(	
					const cusp::array1d<T,cusp::device_memory> &Y,
					const cusp::array1d<T,cusp::device_memory> &absTol,
					thrust::device_ptr<T> dptr_w);
					

			T weightedRMSnorm(
					cusp::array1d<T,cusp::device_memory> &Y,
					const thrust::device_ptr<T> dptr_w);
				

	};



	// AXPY wrapper
	template<typename T>
	void axpyWrapper(cublasHandle_t handle,
			int n,
			const T *alpha,
			const T *X,int incx,
			T *Y, int incy);

	template<>
	void axpyWrapper<float>(cublasHandle_t handle,
			int n,
			const float *alpha,
			const float *X,int incx,
			float *Y, int incy);
	template<>
	void axpyWrapper<double>(cublasHandle_t handle,
			int n,
			const double *alpha,
			const double *X,int incx,
			double *Y, int incy);



}

#include <numerical_solvers/ode/detail/bdfsolver.inl>
