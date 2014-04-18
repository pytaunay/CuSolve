#include <numerical_solvers/linear/gmres.h>
#include <numerical_solvers/nonlinear/newtonraphson.h>
#include <numerical_solvers/ode/bdfsolver.h>

using namespace NumericalSolver;

void testCase_ali() {

	// 04/07/14 Ali
	//// Create functional
	SystemFunctional<double> *myFunctional = new SystemFunctional<double>("/gpfs/home/pzt5044/work/Github/CuSolve/res/040714-eqs/kvalues.txt","/gpfs/home/pzt5044/work/Github/CuSolve/res/040714-eqs/equations.txt");
	int nEq = myFunctional->getTerms().size();
	cusp::array1d<double,cusp::device_memory> Fv(nEq,0);

	//// Initial values
	vector<double> Yhost;
	bool success=false;
	string value;
	ifstream ivfile("/gpfs/home/pzt5044/work/Github/CuSolve/res/040714-eqs/initial-bu.txt");
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

	// Tolerances
	cusp::array1d<double,cusp::device_memory> absTolD(nEq,1e-8);

	//// Create solvers
	GMRES<double> *myLinearSolver = new GMRES<double>(1e-8,1e-8);
	NewtonRaphson<double> *myNonLinearSolver = new NewtonRaphson<double>(myLinearSolver,500,1e-8);

	//// Solve !
	cusp::array1d<double,cusp::device_memory> d(nEq,0);
//	myNonLinearSolver->compute(*myFunctional,*myCooJacobian,Fv,Jv,d,Y);
	BDFsolver<double> *myBdfSolverD = new BDFsolver<double>(*myFunctional,*myCooJacobian,myNonLinearSolver,Y,absTolD);	

//	std::cout << "FINAL SOLUTION" << std::endl;
//	for(int i=0;i<nEq;i++) 
//		std::cout << "Y[" << i << "] = " << Y[i] << std::endl;

	std::cout << "STARTING SOLVER" << std::endl;
	double  tlim = 1.0;
	
	for(int i = 0;i<4;i++) {
		tlim *= pow(10.0,(double)i);
		std::cout << "STARTING SOLVER FOR T = " << tlim << std::endl;
		myBdfSolverD->compute(*myFunctional,*myCooJacobian,Fv,Jv,d,Y,tlim);
	}	

		
	delete myNonLinearSolver;
	delete myLinearSolver;
	delete myCooJacobian;
	delete myFunctional;
	delete myBdfSolverD;
}	
