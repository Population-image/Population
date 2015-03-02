#ifndef C_NEURAL_NET_H
#define C_NEURAL_NET_H

#include "popconfig.h"

#include "data/typeF/TypeF.h"
#include "data/neuralnetwork/NeuralNetwork.h"

void test_neural_net_cpu(void);
void test_neural_net_cpu_mnist(void);

#if defined(HAVE_CUDA)
void test_neural_net_gpu(void);
void test_neural_net_gpu_mnist(void);
void test_cublas(void);
#endif

void printNetwork(struct neural_network* network);

#if defined(HAVE_CUDA)
static const int MAX_NB_THREADS = 1024; // GPU dependent
#endif

struct neural_network;

class GPUNeuralNetwork {
public:
	GPUNeuralNetwork(std::vector<unsigned int> v_layer, double eta);
	~GPUNeuralNetwork();

	void propagateFront(const pop::VecF32& in , pop::VecF32 &out);
	void propagateBackFirstDerivate(const pop::VecF32& desired_output);
	void displayNetwork();

	void setEta(const double eta);
	double getEta() const;

#if defined(HAVE_CUDA)
	void copyNetworkToGPU();
	void copyNetworkFromGPU();
	static pop::F32* gpu_copyDataToGPU(pop::Vec<pop::VecF32> h_data);

	void gpu_propagateFront(pop::F32* in_set, unsigned int in_elt_size, unsigned int idx, pop::F32* out_computed);
	void gpu_propagateBackFirstDerivate(pop::F32* out_set, unsigned int out_elt_size, unsigned int idx);
	void gpu_computeError(pop::F32* out_set, pop::F32* out_computed, unsigned int out_elt_size, unsigned int idx, int* error);
	void gpu_displayNetwork();
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
