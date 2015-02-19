#include <iostream>

#include "popconfig.h"
#include "popcuda.h"
#include "cuda_test.h"
#include "c_neural_net.h"

int main(){
#if defined(HAVE_CUDA)
	std::cout << "You have CUDA support.";

	if (popcuda::isCudaAvailable()) {
		std::cout << " And you have a CUDA device." << std::endl << std::endl;
		//cuda_test();
		std::cout << "\n\n ********** CPU **********\n\n";
		test_neural_net_cpu();
		std::cout << "\n\n ********** GPU **********\n\n";
		test_neural_net_gpu();
		//test_cublas();
	} else {
		std::cout << " But you don't have a CUDA device." << std::endl;
	}

#else
	std::cout << "You do not have CUDA support :(" << std::endl;
#endif

	return 0;
}
