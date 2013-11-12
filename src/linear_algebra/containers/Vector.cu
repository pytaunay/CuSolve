#include <cuda.h>
#include <stdio.h>

#include <CppNS.hpp>

using namespace LinearAlgebra::Containers;


__global__ void kernel_test(Vector<int> v,int size) {
	printf("Hello from GPU ! (tdx = %d, bdx = %d)\n",threadIdx.x, blockIdx.x);
	printf("Content of v:\n");
	for(int i=0; i<size; i++) {
		printf("v[%d]: %d\n",i,v.at_gpu(i));
		v[i] = v[i]*i;
	}	
	printf("Ended\n");
}	

void CUDA_Test(Vector<int> v) {
	
//	printf("Size of v = %d\n", v.N);
	kernel_test <<< 1 , 1 >>> (v,v.size());
}	
