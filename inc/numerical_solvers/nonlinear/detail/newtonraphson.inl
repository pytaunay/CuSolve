/**
 * @file newtonraphson.inl
 * @author Pierre-Yves Taunay (py.taunay@psu.edu)
 * @date November, 2013
 * @brief Implementations of the methods of the class NewtonRaphson
 *
 */
//// STD
#include <cmath>

//// CUDA
#include <cuda.h>

// Thrust
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>

// CUSP
#include <cusp/detail/matrix_base.h>
#include <cusp/array1d.h>
#include <cusp/format.h>
#include <cusp/print.h>

//// CuSolve
// Equation system
//#include <equation_system/systemjacobian.h>
#include <equation_system/systemfunctional.h>

namespace NumericalSolver {

template<typename T>
class square_functor : public thrust::unary_function<T,T> {
	public:
	__host__ __device__
	T operator()(const T &x) {
		return x*x;
	}
};	

template<typename T>
class abs_functor : public thrust::unary_function<T,T> {
	public:
	__host__ __device__
	T operator()(const T &x) {
		return (x < 0.0 ? -x : x);
	}
};	


		template<typename T>
		NewtonRaphson<T>::
			NewtonRaphson() {
				this->maxIter = 50;
				this->tol = (T)1e-6;
				//this->lsolve = new LUDirect();
			}	

		template<typename T>
		NewtonRaphson<T>::
			NewtonRaphson(const int maxIter) {
				this->tol = (T)1e-6;
				this->maxIter = maxIter;
				//this->lsolve = new LUDirect()
			}

		template<typename T>
		NewtonRaphson<T>::
			NewtonRaphson(LinearSolver<T> *lsolve, int maxIter, T tol) {
				this->tol = tol;
				this->maxIter = maxIter;
				this->lsolve = lsolve;
			}	

		template<typename T>
		NewtonRaphson<T>::
			~NewtonRaphson() {}

		template<typename T>
		void NewtonRaphson<T>::
			compute(const SystemFunctional<T> &F,
				const cooJacobian<T> &J,
				cusp::array1d<T,cusp::device_memory> &Fv,
				cusp::coo_matrix<int,T,cusp::device_memory> &Jv,
				cusp::array1d<T,cusp::device_memory> &d,
				cusp::array1d<T,cusp::device_memory> &Y
				)
			{
				
				T tol = (T)1.0;
				// Evaluate the Jacobian and the functional for the first iteration; stores results in Fv and Jv
				std::cout << std::endl;
				for(int i = 0;i < Fv.size();i++)
					std::cout << "Y[" << i << "] = " << Y[i] << std::endl;
				F.evaluate(Fv,Y);
				std::cout << std::endl;
				for(int i = 0;i < Fv.size();i++)
					std::cout << "F[" << i << "] = " << Fv[i] << std::endl;

				thrust::transform(Fv.begin(),Fv.end(),Fv.begin(),thrust::negate<T>());


				J.evaluate(Jv,Y,F.getkData());
				cusp::print(Jv);

				T scale = (T)1.0;
//				cusp::array1d<T,cusp::device_memory> tmp(F.getTerms().size(),(T)(1.0)/scale);


				for(int N = 0; N < this->maxIter; N++) {
					// Solve J*d = -F
					this->lsolve->compute(Jv,d,Fv);	

					std::cout << "Solution of linear system J*d = -F" << std::endl;

					for(int i = 0; i<d.size();i++)
						std::cout << "d[" << i << "] = "<< d[i] << std::endl;

					// Update Y from delta: Y = Y + d/scale
					// Apply scaling to d, in place
				/*	thrust::transform(	d.begin(),
								d.end(),
								tmp.begin(),
								d.begin(),
								thrust::multiplies<T>());
*/	
					// Y = Y+d/s
					thrust::transform(	Y.begin(), 
								Y.end(), 
								d.begin(), 
								Y.begin(), 
								thrust::plus<T>()); 
								
					// Calculate the tolerance: tol = scale*sqrt(sum( d*d ))
					F.evaluate(Fv,Y);
					thrust::transform(Fv.begin(),Fv.end(),Fv.begin(),thrust::negate<T>());

					std::cout << "Iteration " << N+1 << "Functional: " << std::endl;
					for(int i = 0; i<Fv.size();i++)
						std::cout << "F[" << i << "]= " << Fv[i] << std::endl;
				
			//		tol = thrust::transform_reduce(Fv.begin(),Fv.end(),square_functor<T>(),(T)0.0,thrust::plus<T>());
					tol = thrust::transform_reduce(Fv.begin(),Fv.end(),abs_functor<T>(),(T)0.0,thrust::maximum<T>());


//					tol = scale*std::sqrt(tol);
					//tol = std::sqrt(tol);
					std::cout << "Iteration " << N+1 << "\t Tolerance: " << tol << endl;

					// Break if tolerance is attained
					if( tol < this->tol ) {
						std::cout << "INFO Newton-Raphson converged ! Tolerance " << tol << " < " << this->tol << std::endl;
						std::cout << "INFO Solution " << std::endl;
						cusp::print(Y);
						std::cout << "INFO Final functional" << std::endl;
						F.evaluate(Fv,Y);
						cusp::print(Fv);
						break;
					}	
					
					// Update the Jacobian and the functional
					J.evaluate(Jv,Y,F.getkData());
				}	
			} // End of NewtonRaphson :: compute




/*
		template<typename T>
		void NewtonRaphson<T>::
			compute(
				const SystemFunctional<T> &F,
				const SystemJacobian<T,Format> &J,
				cusp::array1d<T,cusp::device_memory> &Fv,
				cusp::detail::matrix_base<int,T,cusp::device_memory,cusp::known_format> &Jv,
				cusp::array1d<T,cusp::device_memory> &d,
				cusp::array1d<T,cusp::device_memory> &Y
				) 
			{
				// Evaluate the Jacobian and the functional for the first iteration; stores results in Fv and Jv
				F.evaluate(Fv,Y);
				J.evaluate(Jv,Y);

				for(int N = 0; N < this->maxIter; N++) {
					// Solve J*d = -F
					this->lsolve->compute(Jv,d,Fv);	

					// Update Y from delta: Y = Y + d
					thrust::transform(	Y.begin(), 
								Y.end(), 
								d.begin(), 
								Y.begin(), 
								thrust::plus<T>()) 
					
					// Calculate the tolerance

					// Break if tolerance is attained
					
					// Update the Jacobian and the functional
					F.evaluate(Fv,Y);
					J.evaluate(Jv,Y);
				}	
			} // End of NewtonRaphson :: compute
*/			

} // End of namespace NumericalSolver			
