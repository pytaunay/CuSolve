//// CuSolve
#include <numerical_solvers/linear/gmres.h>
#include <numerical_solvers/nonlinear/newtonraphson.h>
#include <numerical_solvers/ode/bdfsolver.h>

using namespace NumericalSolver;

void testCase_hires() {
	SystemFunctional<double> *myFunctionalD = new SystemFunctional<double>("/gpfs/work/pzt5044/Github/CuSolve/res/hires/kvals.txt","/gpfs/work/pzt5044/Github/CuSolve/res/hires/equations.txt");
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
	cusp::array1d<double,cusp::device_memory> absTolD(nEq,1e-10);

	cusp::coo_matrix<int,double,cusp::device_memory> JvD(nEq,nEq,nJac);


	thrust::copy(myCooJacobianD->getIdxI().begin(),
			myCooJacobianD->getIdxI().end(), 
			JvD.row_indices.begin());
	thrust::copy(myCooJacobianD->getIdxJ().begin(),
			myCooJacobianD->getIdxJ().end(), 
			JvD.column_indices.begin());

	// Initial values
/*	bool success=false;
	string value;
	ifstream ivfile("/gpfs/work/pzt5044/Github/CuSolve/res/hires/init.txt");
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
*/	
	Yhost.push_back(1.0);
	Yhost.push_back(0.0);
	Yhost.push_back(0.0);
	Yhost.push_back(0.0);
	Yhost.push_back(0.0);
	Yhost.push_back(0.0);
	Yhost.push_back(0.0);
	Yhost.push_back(0.0057);

	thrust::copy(Yhost.begin(),Yhost.end(),YD.begin());

	// Linear solver
	GMRES<double> *myLinearSolverD = new GMRES<double>(1e-30,1e-30);

	// Create a non linear solver
	NewtonRaphson<double> *myNonLinearSolverD = new NewtonRaphson<double>(myLinearSolverD,500,1e-10);
	BDFsolver<double> *myBdfSolverD = new BDFsolver<double>(*myFunctionalD,*myCooJacobianD,myNonLinearSolverD,YD,absTolD);	

	std::cout << "STARTING SOLVER" << std::endl;
	myBdfSolverD->compute(*myFunctionalD,*myCooJacobianD,FvD,JvD,dD,YD,321.8122);

	delete myBdfSolverD;
	delete myCooJacobianD;
	delete myFunctionalD;
	delete myNonLinearSolverD;
	delete myLinearSolverD;
}
