/**
 * @file bdfsolver.inl
 * @author Pierre-Yves Taunay (py.taunay@psu.edu)
 * @date March 2014
 * @brief Implementation of the methods of the class BDF solver
 *
 */


namespace NumericalSolver {
	
	// Constructors: allocate memory on GPU, etc...
	template<typename T>
	BDFsolver<T>::
		BDFsolver() {

		}

	//etc		
	

	// Compute
	template<typename T>
	BDFsolver<T>::
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

			// Take a step

			// 1. Make a prediction
			// 1.1 Update current time
			this->t += this->dt;
			// 1.2 Apply Nordsieck prediction : ZN_0 = ZN_n-1 * A(q)


			// 2. Calculate L polynomial and other data
			// 3. Non linear solver
			// 4. Check results


			// Complete step

			// 1. Update data
			this->nist++;

			// 2. Apply correction

			// 3. Manage order q

			// Prepare next step



		}



