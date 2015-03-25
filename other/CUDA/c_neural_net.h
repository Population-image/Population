#ifndef C_NEURAL_NET_H
#define C_NEURAL_NET_H

#include <vector>

#include "popconfig.h"

#include "data/typeF/TypeF.h"
#include "data/neuralnetwork/NeuralNetwork.h"

void test_neural_net_cpu(const int nb_epoch);
void test_neural_net_cpu_mnist(const int nb_epoch);

void loadDatabase(std::string directory, const int max_per_folder, pop::Vec<pop::VecF32> &v_neuron_in, pop::Vec<pop::VecF32> &v_neuron_out);

#if defined(HAVE_CUDA)
void test_neural_net_gpu(const int nb_epoch);
void test_neural_net_gpu_mnist(const int nb_epoch);
void test_neural_net_gpu_augmented_database(const int max_files_per_folder, const int network_for_training, std::string database_training, std::string database_test, const int nb_epoch);
void bench_propagate_front_gpu_augmented_database(const int max_files_per_folder, std::string network_path, std::string database_training, std::string database_test, const int nb_epoch);
#endif

struct neural_network;

class GPUNeuralNetwork {
public:
	GPUNeuralNetwork();
	GPUNeuralNetwork(std::vector<unsigned int> v_layer, double eta);
	~GPUNeuralNetwork();

	void propagateFront(const pop::VecF32& in , pop::VecF32 &out);
	void propagateBackFirstDerivate(const pop::VecF32& desired_output);
	void displayNetwork();

	void save(std::string filename);
	void load(std::string filename);

	void setEta(const double eta);
	double getEta() const;

#if defined(HAVE_CUDA)
	void copyNetworkToGPU();
	void copyNetworkFromGPU();
	static pop::F32* gpu_copyDataToGPU(pop::Vec<pop::VecF32> h_data, const unsigned int min, unsigned int n, std::vector<int> shuffle = std::vector<int>());

	void gpu_propagateFront(pop::F32* in_set, unsigned int in_elt_size, unsigned int idx, pop::F32* out_computed);
	void gpu_propagateBackFirstDerivate(pop::F32* out_set, unsigned int out_elt_size, unsigned int idx);
	void gpu_computeError(pop::F32* out_set, pop::F32* out_computed, unsigned int out_elt_size, unsigned int idx, int* error);
	void gpu_displayNetwork();

	void gpu_learn(pop::Vec<pop::VecF32>& vtraining_in, pop::Vec<pop::VecF32>& vtraining_out, pop::Vec<pop::VecF32>& vtest_in, pop::Vec<pop::VecF32>& vtest_out, bool final_cpu_test, const int nb_epoch);
	void gpu_propagate(pop::Vec<pop::VecF32>& vtraining_in, pop::Vec<pop::VecF32>& vtest_in, const int nb_epoch);
#endif

private:
	void createNetwork(std::vector<unsigned int> v_layer, double eta);
	void deleteNetwork();

	static void printNeuronsVector(pop::F32* V, unsigned int size, std::string label);
	static void printWeightMatrix(pop::F32* M, unsigned int height, unsigned int width, std::string label);

	float sigmoid(float x) { return 1.7159f*tanh(0.66666667f*x); }
	float derived_sigmoid(float S) { return 0.666667f/1.7159f*(1.7159f*1.7159f-S*S); }

	struct neural_network* h_network; // neural network on the CPU

#if defined(HAVE_CUDA)
	void deleteNetworkOnGPU();

	struct neural_network* d_network; // neural network on the GPU
#endif
};

#endif
