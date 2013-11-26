/**
 * @file gmres.h
 * @author Pierre-Yves Taunay (py.taunay@psu.edu)
 * @date November, 2013
 * @brief Class representation of a GMRES solver
 *
 */
#pragma once

#include <cusolve.h>

#include <cuda.h>
#include <cusp/krylov/gmres.h>


using namespace LinearAlgebra::Containers;

namespace NumericalSolver {
	class GMRES : public LinearSolver {
		protected:
			int restartIter; 	
			int maxIter;		
			double tol;		

		public:
			GMRES() {
				this->restartIter = 50;
				this->maxIter = 500;
				this->tol = 1e-6;
			}	

/*
			void compute(SparseMatrix<float> &A, SparseVector<float> &b, SparseVector<float> &x); 
			void compute(SparseMatrix<double> &A, SparseVector<double> &b, SparseVector<double> &x); 
			void compute(FullMatrix<float> &A, Vector<float> &b, Vector<float> &x); 
			void compute(FullMatrix<double> &A, Vector<double> &b, Vector<double> &x); 
*/
			// Sparse matrix implementation
			void compute	(	SparseMatrix<float> &A, 
						cusp::array1d<float,cusp::device_memory> &b, 
						cusp::array1d<float,cusp::device_memory> &x
					); 

			void compute	(	SparseMatrix<double> &A, 
						cusp::array1d<double,cusp::device_memory> &b, 
						cusp::array1d<double,cusp::device_memory> &x
					); 

			// Full matrix implementation
			void compute	(	float *A, 
						float *b,
						float *x
					); 

			void compute	(	double *A, 
						double *b,
						double *x
					); 




	};		
}

