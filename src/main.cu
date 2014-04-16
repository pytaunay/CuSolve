#include <iostream>
#include <ctime>

#include <cuda.h>
#include <thrust/functional.h>


#include <numerical_solvers/linear/gmres.h>
#include <numerical_solvers/nonlinear/newtonraphson.h>

#include <cusp/coo_matrix.h>
#include <cusp/array1d.h>

#include <cusp/format.h>
#include <cusp/detail/matrix_base.h>
#include <cusp/print.h>

#include <equation_system/systemfunctional.h>
#include <equation_system/coojacobian.h>
//#include <equation_system/bdffunctional.h>
//#include <equation_system/bdfcoojacobian.h>

#include <numerical_solvers/ode/bdfsolver.h>

using namespace NumericalSolver;

template<typename T>
struct abs_functor : public thrust::unary_function<T,T>
{
	__host__ __device__
	T operator()(const T &X) const {
		return ( X < (T)(0.0) ? -X : X );
	}
};	

int main() {


/*
	// 04/07/14 Ali
	//// Create functional
	SystemFunctional<double> *myFunctional = new SystemFunctional<double>("/gpfs/home/pzt5044/work/Github/CuSolve/res/040714-eqs/kvalues.txt","/gpfs/home/pzt5044/work/Github/CuSolve/res/040714-eqs/equations.txt");
	int nEq = myFunctional->getTerms().size();
	cusp::array1d<double,cusp::device_memory> Fv(nEq,0);

	//// Initial values
	vector<double> Yhost;
	bool success=false;
	string value;
	ifstream ivfile("/gpfs/home/pzt5044/work/Github/CuSolve/res/040714-eqs/initial.txt");
	if (ivfile.good()){
		success=false;

		while(getline(ivfile,value)){
			success |= clientp::parse_csv(value.begin(), value.end(), Yhost);
		}

		ivfile.close();

		// abort;
		if (!success)
			throw std::invalid_argument( "loading of inital value data failed" );

	}

	// Copy initial to device
	cusp::array1d<double,cusp::device_memory> Y(nEq,1);
	thrust::copy(Yhost.begin(),Yhost.end(),Y.begin());


	//// Create Jacobian
	cooJacobian<double> *myCooJacobian = new cooJacobian<double>(*myFunctional);
	int nJac = myCooJacobian->getTerms().size();
	cusp::coo_matrix<int,double,cusp::device_memory> Jv(nEq,nEq,nJac);

	// Set up the COO matrix
	thrust::copy(myCooJacobian->getIdxI().begin(),
			myCooJacobian->getIdxI().end(), 
			Jv.row_indices.begin());
	thrust::copy(myCooJacobian->getIdxJ().begin(),
			myCooJacobian->getIdxJ().end(), 
			Jv.column_indices.begin());


	//// Create solvers
	GMRES<double> *myLinearSolver = new GMRES<double>(1e-5,1e-5);
	NewtonRaphson<double> *myNonLinearSolver = new NewtonRaphson<double>(myLinearSolver,500,1e-4);

	//// Solve !
	cusp::array1d<double,cusp::device_memory> d(nEq,0);
	myNonLinearSolver->compute(*myFunctional,*myCooJacobian,Fv,Jv,d,Y);

	std::cout << "FINAL SOLUTION" << std::endl;
	for(int i=0;i<nEq;i++) 
		std::cout << "Y[" << i << "] = " << Y[i] << std::endl;


		
	delete myNonLinearSolver;
	delete myLinearSolver;
	delete myCooJacobian;
	delete myFunctional;
*/	


	//// Solving a non-linear system
	// Create a linear solver
//	GMRES<float> *myLinearSolver = new GMRES<float>();
	// Create a non linear solver
//	NewtonRaphson<float> *myNonLinearSolver = new NewtonRaphson<float>(myLinearSolver,500,1e-5);
	
	// Set up the analytical forms
//	SystemFunctional<float> *myFunctional = new SystemFunctional<float>("/gpfs/work/pzt5044/Github/CuSolve/res/k_values_new.csv","/gpfs/work/pzt5044/Github/CuSolve/res/newer_equations.txt");
//	SystemFunctional<float> *myFunctional = new SystemFunctional<float>("/gpfs/work/pzt5044/Github/CuSolve/res/040414-kvals.txt","/gpfs/work/pzt5044/Github/CuSolve/res/040414-equations.txt");
//	SystemFunctional<float> *myFunctional = new SystemFunctional<float>("/gpfs/work/pzt5044/Github/CuSolve/res/roberts/kvals.txt","/gpfs/work/pzt5044/Github/CuSolve/res/roberts/equations.txt");
//	cooJacobian<float> *myCooJacobian = new cooJacobian<float>(*myFunctional);

	// Set up the numerical containers	
	// array1d Fv
	// cooMatrix Jv
	// array1d dv
	// array1d Y

	// Number of equations
	/*
	int nEq = myFunctional->getTerms().size();
	int nJac = myCooJacobian->getTerms().size();
	std::cout << nEq << " equations were parsed " << std::endl;
	std::cout << nJac << " non zeros for the jacobian " << std::endl;
*/
	// Set up numerical arrays
//	cusp::array1d<float,cusp::device_memory> Fv(nEq,0);
/*	cusp::array1d<float,cusp::device_memory> Y(nEq,1);
	cusp::array1d<float,cusp::device_memory> d(nEq,0);
	cusp::array1d<float,cusp::host_memory> Yh(nEq,0);
	vector<float> Yhost;
	cusp::coo_matrix<int,float,cusp::device_memory> Jv(nEq,nEq,nJac);
	*/
/*
	cusp::array1d<float,cusp::device_memory> absTol(nEq,0);
	absTol[0] = 1e-8;
	absTol[1] = 1e-14;
	absTol[2] = 1e-6;
*/


	// Set up initial guess
//	srand(1024);
//	for (int i=0; i<nEq; i++)
//		Yh[i] = 0.00001f*(float)rand() / (float) RAND_MAX;


//	myNonLinearSolver->compute(*myFunctional,*myCooJacobian,Fv,Jv,d,Y);

//	BDFsolver<float> *myBdfSolver = new BDFsolver<float>(*myFunctional,*myCooJacobian,myNonLinearSolver,Y);	



	///// DOUBLE PRECISION
	SystemFunctional<double> *myFunctionalD = new SystemFunctional<double>("/gpfs/work/pzt5044/Github/CuSolve/res/roberts/kvals.txt","/gpfs/work/pzt5044/Github/CuSolve/res/roberts/equations.txt");
	cooJacobian<double> *myCooJacobianD = new cooJacobian<double>(*myFunctionalD);

	// Number of equations
	int nEq = myFunctionalD->getTerms().size();
	int nJac = myCooJacobianD->getTerms().size();
	std::cout << nEq << " equations were parsed " << std::endl;
	std::cout << nJac << " non zeros for the jacobian " << std::endl;

	// Set up numerical arrays
	cusp::array1d<double,cusp::device_memory> FvD(nEq,0);
	cusp::array1d<double,cusp::device_memory> YD(nEq,1);
	cusp::array1d<double,cusp::device_memory> dD(nEq,0);
	cusp::array1d<double,cusp::host_memory> YhD(nEq,0);
	vector<double> Yhost;

	// Tolerances
	cusp::array1d<double,cusp::device_memory> absTolD(nEq,0);
	absTolD[0] = 1e-8;
	absTolD[1] = 1e-14;
	absTolD[2] = 1e-6;

	cusp::coo_matrix<int,double,cusp::device_memory> JvD(nEq,nEq,nJac);


	thrust::copy(myCooJacobianD->getIdxI().begin(),
			myCooJacobianD->getIdxI().end(), 
			JvD.row_indices.begin());
	thrust::copy(myCooJacobianD->getIdxJ().begin(),
			myCooJacobianD->getIdxJ().end(), 
			JvD.column_indices.begin());

	// Initial values
	bool success=false;
	string value;
	ifstream ivfile("/gpfs/work/pzt5044/Github/CuSolve/res/roberts/init.txt");
	if (ivfile.good()){
		success=false;

		while(getline(ivfile,value)){
			success |= clientp::parse_csv(value.begin(), value.end(), Yhost);
		}

		ivfile.close();

		// abort;
		if (!success)
			throw std::invalid_argument( "loading of inital value data failed" );

	}
	thrust::copy(Yhost.begin(),Yhost.end(),YD.begin());



	// Linear solver
	GMRES<double> *myLinearSolverD = new GMRES<double>(1e-30,1e-30);

	// Create a non linear solver
	NewtonRaphson<double> *myNonLinearSolverD = new NewtonRaphson<double>(myLinearSolverD,500,1e-8);
	BDFsolver<double> *myBdfSolverD = new BDFsolver<double>(*myFunctionalD,*myCooJacobianD,myNonLinearSolverD,YD,absTolD);	

	std::cout << "STARTING SOLVER FOR T = 0.4" << std::endl;
	myBdfSolverD->compute(*myFunctionalD,*myCooJacobianD,FvD,JvD,dD,YD,0.4);

	double tlim;
	for(int i = 0;i<11;i++) {
		tlim = 4.0*pow(10.0,(double)i);
		std::cout << "STARTING SOLVER FOR T = " << tlim << std::endl;
		myBdfSolverD->compute(*myFunctionalD,*myCooJacobianD,FvD,JvD,dD,YD,tlim);
	}	
		
		

	delete myBdfSolverD;
	delete myCooJacobianD;
	delete myFunctionalD;
	delete myNonLinearSolverD;
	delete myLinearSolverD;


	
	return 0;
}
