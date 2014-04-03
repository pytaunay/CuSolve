/**
 * @file systemfunctional.h
 * @author Pierre-Yves Taunay (py.taunay@psu.edu)
 * @date November, 2013
 * @brief Class representation of the functional of a system of equations
 *
 */
#pragma once


// STD
#include <vector>
#include <map>
#include <string>

// CUSP
#include <cusp/array1d.h>

// CuSolve
#include <equation_system/clientp.hpp>
#include <equation_system/evalnode.h>

namespace System {

	/*!\class SystemFunctional
	 *
	 *
	 *
	 * \tparam T Type
	 */
	template<typename T>
	class SystemFunctional {
		protected:
			// Host 
			std::vector<T> h_kData; /*!< k parameter data to load */
			std::vector<T> h_kInds; /*!< k indices for equations */ 
			std::vector<T> constants; /*!< all constant and sign data */
			std::vector< map<T,T> > yFull; /*!< all y data (yindex(key) & power(value)) */
			std::vector<int> terms; /*!< Number of terms per equation evaluation */

			int maxElements; /*!< Maximum number of terms in the equation evaluation */
			int maxTermSize; /*!< Maximum number of elements encountered in equation term, for node size */

			int nbEq; /*!< Number of equations parsed that were non zero*/

			// Device wrappers 
			EvalNode<T> *d_fNodes; /*!< Nodes in the polynomial tree representation; allocated on the device*/
			int *d_fTerms; /*!< Number of terms per equation evaluation; allocated on the device*/
			int *d_fOffsetTerms; /*!< Offset terms; allocated on the device*/
			cusp::array1d<T,cusp::device_memory> d_kData; /*!< k parameter data loaded; allocated on the device*/

		private:
			/*!\brief Default constructor. Private so user can not use it
			 *
			 */

		public:
			SystemFunctional() {
				maxElements = 0;
				maxTermSize = 0;
				d_fNodes= NULL;
				d_fTerms= NULL;
				d_fOffsetTerms = NULL;
			}	
			/*!\brief Constructor with filename
			 *
			 *
			 * @param[in] filename location of the k data
			 */
			SystemFunctional(char *k_values, char *equations_file); 

			/*!\brief Assignment operator
			 *
			 */
			SystemFunctional<T>& operator=( SystemFunctional<T> tmp ) {
				std::swap( h_kData, tmp.h_kData );
				std::swap( h_kInds, tmp.h_kInds );
				std::swap( constants, tmp.constants );
				std::swap( yFull , tmp.yFull );
				std::swap( terms, tmp.terms );
				maxElements = tmp.maxElements;
				maxTermSize = tmp.maxTermSize;
				nbEq = tmp.nbEq;

				return *this;
			}	

			/*!\brief Evaluation of the system functional, based on the data stored in the device memory
			 *
			 *
			 * @param[inout] F vector where the evaluation is stored
			 * @param[in] Y vector at which we want to evaluate the functional
			 */ 
			__host__ void evaluate(
					cusp::array1d<T,cusp::device_memory> &F,
					const cusp::array1d<T,cusp::device_memory> &Y) const;


			/*!\brief k Indices getter
			 *
			 *
			 * @param[out] constant reference to h_kInds
			 */
			__host__ std::vector<T> const & getkInds() const {	
				return h_kInds;
			}	

			/*!\brief constants getter
			 *
			 *
			 * @param[out] constant reference to constant
			 */
			__host__ std::vector<T> const & getConstants() const {	
				return constants;
			}	

			/*!\brief yFull getter
			 *
			 *
			 * @param[out] constant reference to yFull
			 */
			__host__ std::vector< std::map<T,T> > const & getyFull() const {
				return yFull;
			}	
			/*\brief terms getter
			 *
			 *
			 * @param[out] constant reference to terms
			 */
			__host__ std::vector<int> const & getTerms() const {
				return terms;
			} 	

			__host__ cusp::array1d<T,cusp::device_memory> const & getkData() const {
				return d_kData;
			}	

			__host__ std::vector<T> const & getkDataHost() const {
				return h_kData;
			}	

			__host__ int const & getMaxElements() const {
				return maxElements;
			}	
			__host__ int const & getMaxTermSize() const {
				return maxTermSize;
			}	

		public:
			__device__ void evaluateImplementation(T *d_fp); 
	};
	
	/*!\brief Kernel for the evaluation of the system functional
	 *
	 *
	 * @param[in] d_fp device function pointer, which is obtained from a raw pointer cast of a cusp array1d
	 */ 
	template<typename T>
	__global__ void k_FunctionalEvaluate(T *d_fp,const EvalNode<T> *d_fNodes, const int *d_fTerms,const int *d_fOffsetTerms,int nbEq); 
} // end of System	

#include <equation_system/detail/systemfunctional.inl>
