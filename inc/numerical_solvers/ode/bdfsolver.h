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

namespace NumericalSolver {

	namespace constants {
		// Maximum order for the BDF
		const int Q_MAX = 5; 
		const int L_MAX = Q_MAX + 1;
		const double ONE = 1.0;
	}

	template<typename T>
	class BDFsolver : public ImplicitODESolver, LMMODESolver {

		protected:
			NonLinearSolver<T> *nlsolve; /*!< Non linear solver */
			T tol; /*!< Tolerance */

			short q; /*!< BDF order */
			T dt; /*!< Current time step */
			T t; /*!< Current time */

			int nist; /*!< Number of internal steps taken */
			int neq; /*!< Number of independent SYSTEM of ODE */
			int N; /*!< Size of the system of ODE */

				

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




		public:
			BDFsolver();
			~BDFsolver();


		void compute(const SystemFunctional<T> &F,
			const cooJacobian<T> &J,
			cusp::array1d<T,cusp::device_memory> &Fv,
			cusp::coo_matrix<int,T,cusp::device_memory> &Jv,
			cusp::array1d<T,cusp::device_memory> &d,
			cusp::array1d<T,cusp::device_memory> &Y );


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
