#include <iostream>

#include "popconfig.h"
#include "popcuda.h"
#include "cuda_test.h"
#include "c_neural_net.h"

int main(void) {
#if defined(HAVE_CUDA)
	std::cout << "You have CUDA support.";

	if (popcuda::isCudaAvailable()) {
		std::cout << " And you have a CUDA device." << std::endl << std::endl;

		//cuda_test();
		//test_cublas();
		//test_convolution();

		const int nb_epoch = 1;
		const int max_files_per_folder = 30;
		//test_neural_net_cpu(1000/*nb_epoch*/);
		//test_neural_net_gpu(1000/*nb_epoch*/);
		//test_neural_net_cpu_mnist(nb_epoch);
		//test_neural_net_gpu_mnist(nb_epoch);
		//test_neural_net_gpu_augmented_database(max_files_per_folder, 1, "/media/pl/shared/PL/neural_nets_samples/ANV_light/data_base_augmented", "/media/pl/shared/PL/neural_nets_samples/ANV_light/data_base", nb_epoch);
		//bench_propagate_front_gpu_augmented_database(max_files_per_folder, "/home/pl/Documents/alphanumericvision/deep_big_simple_neural_net/gpu/1/network.bin", "/media/pl/shared/PL/neural_nets_samples/ANV_light/data_base_augmented", "/media/pl/shared/PL/neural_nets_samples/ANV_light/data_base", nb_epoch);
		//test_neural_net_conv_cpu(500/*nb_epoch*/);
		//test_neural_net_conv_gpu(500/*nb_epoch*/);
		//test_neural_net_conv_cpu_mnist(nb_epoch);
		test_neural_net_conv_gpu_mnist(nb_epoch);
	} else {
		std::cout << " But you don't have a CUDA device." << std::endl;
	}

#else
	std::cout << "You do not have CUDA support :(" << std::endl;
#endif

	return 0;
}
