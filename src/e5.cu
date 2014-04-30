//// CuSolve
#include <numerical_solvers/linear/gmres.h>
#include <numerical_solvers/nonlinear/newtonraphson.h>
#include <numerical_solvers/ode/bdfsolver.h>

using namespace NumericalSolver;


void testCase_e5() {
	SystemFunctional<double> *myFunctionalD = new SystemFunctional<double>("/gpfs/work/pzt5044/Github/CuSolve/res/e5/kvals.txt","/gpfs/work/pzt5044/Github/CuSolve/res/e5/equations.txt");
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
	cusp::array1d<double,cusp::device_memory> absTolD(nEq,1.11e-24);

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
	ifstream ivfile("/gpfs/work/pzt5044/Github/CuSolve/res/e5/init.txt");
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
	GMRES<double> *myLinearSolverD = new GMRES<double>(1e-40,1e-40);

	// Create a non linear solver
	NewtonRaphson<double> *myNonLinearSolverD = new NewtonRaphson<double>(myLinearSolverD,500,1e-10);
	BDFsolver<double> *myBdfSolverD = new BDFsolver<double>(*myFunctionalD,*myCooJacobianD,myNonLinearSolverD,YD,absTolD);	

	std::cout << "STARTING SOLVER FOR T = 0.1" << std::endl;
	myBdfSolverD->compute(*myFunctionalD,*myCooJacobianD,FvD,JvD,dD,YD,0.1);

	double tlim;
	for(int i = 0;i<14;i++) {
		tlim = 1.0*pow(10.0,(double)i);
		std::cout << "STARTING SOLVER FOR T = " << tlim << std::endl;
		myBdfSolverD->compute(*myFunctionalD,*myCooJacobianD,FvD,JvD,dD,YD,tlim);
	}	



/*
	delete myBdfSolverD;
	delete myCooJacobianD;
	delete myFunctionalD;
	delete myNonLinearSolverD;
	delete myLinearSolverD;
*/	
}
