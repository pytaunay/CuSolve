#include <cuda.h>

#include <iostream>
#include <type_traits>

#include <CppNS.hpp>

using namespace LinearAlgebra::Containers;
using namespace NumericalSolver;

template<class T>
class COOMatrix : public SparseMatrix<T> {
	protected:

	public:

};	

int main() {

	Vector<double> v1(10);
	Vector<double> v2(10);


/*	GMRES <SparseMatrix<double>,Vector<double>> GMRESInstance(500,(double)1.0);

	GMRESInstance(SM,v1,v2);
*/



/*
	SparseMatrix<double> SM;
	GMRES<double,SparseMatrix> GMRESInstance;
	GMRESInstance(SM);
*/
	SparseMatrix<float> SM2;
	GMRES<float,SparseMatrix> GMRESInstance2;
	GMRESInstance2(SM2);

	COOMatrix<double> COOSM;
	GMRES<double,COOMatrix> GMRESInstance3;
	GMRESInstance3(COOSM);

	GMRES<double,Vector> GMRESInstance4;
//	GMRESInstance4(v1);


	std::cout << "SM of SM ?\t" << std::is_base_of<SparseMatrix<double>,SparseMatrix<double>>::value << std::endl;
	std::cout << "SM of CM ?\t" << std::is_base_of<COOMatrix<double>,SparseMatrix<double>>::value << std::endl;
	std::cout << "CM of SM ?\t" << std::is_base_of<SparseMatrix<double>,COOMatrix<double>>::value << std::endl;
	std::cout << "Is COO same as SM?\t" << std::is_same<SparseMatrix<double>, COOMatrix<double>>::value<<std::endl;

	return 0;
}	
