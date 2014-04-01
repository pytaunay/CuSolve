#include <iostream>
#include <ctime>

#include <cuda.h>

#include <numerical_solvers/linear/gmres.h>
#include <numerical_solvers/nonlinear/newtonraphson.h>

#include <cusp/coo_matrix.h>
#include <cusp/array1d.h>

#include <cusp/format.h>
#include <cusp/detail/matrix_base.h>
#include <cusp/print.h>

#include <equation_system/systemfunctional.h>
#include <equation_system/coojacobian.h>

#include <numerical_solvers/ode/bdfsolver.h>

using namespace NumericalSolver;

int main() {


	// Create a linear numerical solver
	GMRES<float> *myLinearSolver = new GMRES<float>();

	// Create containers
	cusp::array1d<float,cusp::device_memory> b;
	cusp::array1d<float,cusp::device_memory> X;
	cusp::coo_matrix<int,float,cusp::device_memory> A; 

//	myLinearSolver->compute(A,b,X);



	//// Solving a non-linear system
	// Create a linear solver
	GMRES<float> *myLinearSolverD = new GMRES<float>();
	// Create a non linear solver
	NewtonRaphson<float> *myNonLinearSolver = new NewtonRaphson<float>(myLinearSolverD,500,1e-5);
	
	// Set up the analytical forms

	// Set up the numerical containers	
	// array1d Fv
	// cooMatrix Jv
	// array1d dv
	// array1d Y

	// Solve



	/* Testing the analytical containers and evaluation
	 */
	SystemFunctional<float> *myFunctional = new SystemFunctional<float>("/gpfs/work/pzt5044/Github/CuSolve/res/k_values_new.csv","/gpfs/work/pzt5044/Github/CuSolve/res/newer_equations.txt");
	cooJacobian<float> *myCooJacobian = new cooJacobian<float>(*myFunctional);

	// Number of equations
	int nEq = myFunctional->getTerms().size();
	int nJac = myCooJacobian->getTerms().size();
	std::cout << nEq << " equations were parsed " << std::endl;
	std::cout << nJac << " non zeros for the jacobian " << std::endl;

	// Set up numerical arrays
	cusp::array1d<float,cusp::device_memory> Fv(nEq,0);
	cusp::array1d<float,cusp::device_memory> Y(nEq,1);
	cusp::array1d<float,cusp::device_memory> d(nEq,0);
	cusp::array1d<float,cusp::host_memory> Yh(nEq,0);

	srand(1024);
//	srand(time(NULL));
	for (int i=0; i<nEq; i++)
		Yh[i] = 0.00001f*(float)rand() / (float) RAND_MAX;

	thrust::copy(Yh.begin(),Yh.end(),Y.begin());

	printf("\n");
	cusp::print(Y);
	printf("\n");

	cusp::coo_matrix<int,float,cusp::device_memory> Jv(nEq,nEq,nJac);

	thrust::copy(myCooJacobian->getIdxI().begin(),
			myCooJacobian->getIdxI().end(), 
			Jv.row_indices.begin());
	thrust::copy(myCooJacobian->getIdxJ().begin(),
			myCooJacobian->getIdxJ().end(), 
			Jv.column_indices.begin());


//	myFunctional->evaluate(Fv,Y);
//	myCooJacobian->evaluate(Jv,Y,myFunctional->getkData());
		
//	cusp::print(Jv);

	    // print contents of D
//        for(int i = 0; i < Y.size(); i++)
//	        std::cout << "F[" << i << "]= " << Fv[i] << std::endl;



//	myNonLinearSolver->compute(*myFunctional,*myCooJacobian,Fv,Jv,d,Y);

	BDFsolver<float> *myBdfSolver = new BDFsolver<float>();	
	myBdfSolver->compute(*myFunctional,*myCooJacobian,Fv,Jv,d,Y);


	delete myCooJacobian;
	delete myFunctional;
	delete myNonLinearSolver;
	delete myLinearSolverD;
	delete myLinearSolver;

	return 0;
}
