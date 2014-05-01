/**
 * @file thrust_functors.h
 * @author Pierre-Yves Taunay (py.taunay@psu.edu)
 * @date April 2014
 * @brief Various functors used in Thrust operations
 *
 */

namespace cusolve {
namespace functor {
	/**
	 *
	 * \brief Functor to inverse elements of an array and multiply them by a scalar
	 * \tparam T Single/double precision
	 *
	 *
	 */
	template<typename T>
	struct scalar_inv_functor : public thrust::binary_function<T,T,T>
	{
		const T a; /*!< Scalar value */

		/**
		 * \brief Constructor 
		 *
		 * Initializes scalar value
		 *
		 */
		scalar_inv_functor(T _a): a(_a) {}


		/** 
		 * \brief Overloaded operator()
		 *
		 * Operation:
		 * \f$ X_{i} = a/X_{i} \f$
		 * 
		 */ 
		__host__ __device__
		T operator()(const float&x) const {
			return a/x;
		}
	};	

	/**
	 *
	 * \brief Functor to square each element of an array
	 * \tparam T Single/double precision
	 *
	 *
	 */
	template<typename T>
	struct square_functor : public thrust::binary_function<T,T,T>
	{
		/**
		 * \brief Overloaded operator()
		 *
		 * Operation:
		 * \f$ X_i = \left( X_i^2 \right ) \f$
		 *
		 */
		__host__ __device__
		T operator()(const T &x) {
			return x*x;
		}
	};	


	/**
	 *
	 * \brief Functor to multiply each element of an array by a scalar
	 * \tparam T Single/double precision
	 *
	 */
	template<typename T>
	struct scalar_functor : public thrust::binary_function<T,T,T>
	{
		const T a; /*!< Scalar value */

		/**
		 * \brief Constructor 
		 *
		 * Initializes scalar value
		 * 
		 */
		scalar_functor(T _a) : a(_a) {}

		/**
		 * \brief Overloaded operator()
		 *
		 * Operation:
		 * \f$ X_i = a\cdot X_i \f$
		 *
		 */
		__host__ __device__
		T operator()(const T &x) const {
			return a*x;
		}
	};	
	
	/**
	 * 
	 * \brief Functor to calculate the upper bound of the first time step in the BDF method
	 * \tparam T Single/double precision
	 *
	 * The functor takes a tuple with three arguments created with a zip iterator as an input argument:
	 * - 
	 * -
	 * - 
	 *
	 */
	template<typename T>
	struct dt0_upper_bound : public thrust::unary_function<thrust::tuple<T,T,T>,T>
	{
		const T relTol; /*!< Relative tolerance required for the calculation */

		/**
		 * \brief Constructor
		 *
		 * Initializes relative tolerance
		 *
		 */
		dt0_upper_bound(T _relTol) : relTol(_relTol) {}

		/**
		 * \brief Overloaded operator()
		 *
		 * Operation:
		 *
		 */
		__host__ __device__
		T operator()(const thrust::tuple<T,T,T>& t) const {
			return (abs ( thrust::get<1>(t) ) / ( (relTol+0.1) * abs(thrust::get<0>(t)) + thrust::get<2>(t)));
		}
	};	


	/**
	 * \brief Functor to calculate the weights necessary for error evaluations
	 * \tparam T Single/double precision
	 *
	 * The functor takes a tuple with two arguments, created with a zip iterator, as an input argument:
	 * -
	 * -
	 *
	 */ 
	template<typename T>
	struct eval_weights_functor : public thrust::unary_function<thrust::tuple<T,T>,T>
	{
		const T relTol; /*!< Relative tolerance required for the calculation */

		 /**
		 * \brief Constructor
		 *
		 * Initializes relative tolerance
		 *
		 */

		eval_weights_functor(T _relTol) : relTol(_relTol) {}
			
		/**
		 * \brief Overloaded operator()
		 *
		 * Operation:
		 *
		 */
		__host__ __device__
		T operator()(const thrust::tuple<T,T>& t) const {
			return ( (T)1.0 / ( relTol * abs( thrust::get<0>(t) ) + thrust::get<1>(t) ) );
		}
	};	

	/** 
	 * \brief Functor to evaluate the weighted RMS error
	 * \tparam T Single/double precision
	 *
	 * The function takes a tuple with two arguments, created with a zip iterator, as an input argument:
	 * -
	 * -
	 *
	 */
	template<typename T>
	struct weighted_rms_functor : public thrust::unary_function<thrust::tuple<T,T>,T>
	{
		/**
		 * \brief Overloaded operator()
		 *
		 * Operation:
		 *
		 */
		__host__ __device__
		T operator()(const thrust::tuple<T,T>&t) const {
			return ( thrust::get<0>(t)*thrust::get<0>(t)*thrust::get<1>(t)*thrust::get<1>(t) );
		}
	};	

} // functor namespace
} // cusolve namespace
