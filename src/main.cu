#include <iostream>
/*
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
#include <equation_system/bdffunctional.h>
#include <equation_system/bdfcoojacobian.h>

#include <numerical_solvers/ode/bdfsolver.h>
*/
#include "test_cases.h"


//using namespace NumericalSolver;



int main() {


	std::cout << "Starting Roberts test case ... " << std::endl;
//	testCase_roberts();
	std::cout << "Starting HIRES test case ..." << std::endl;
	testCase_hires();


	
	return 0;
}
