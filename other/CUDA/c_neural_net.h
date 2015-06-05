#ifndef C_NEURAL_NET_H
#define C_NEURAL_NET_H

#include <vector>

#include "popconfig.h"
#include "data/typeF/TypeF.h"
#include "data/neuralnetwork/NeuralNetwork.h"

enum TypeLayer {
    LAYER_INPUT,
    LAYER_INPUT_MATRIX,
    LAYER_FULLY_CONNECTED,
    LAYER_CONVOLUTIONAL
};

// used to create a neural network
struct layer_representation {
	TypeLayer type;
	int nb_neurons;    			// for fully connected and input
	pop::I32 sizei_map;			// for input matrix
	pop::I32 sizej_map;			// for input matrix
	int nbr_map;				// for input matrix and convolutional
	int radius_kernel;			// for convolutional
	int sub_resolution_factor;	// for convolutional
};

struct neural_network;

const float GPU_MEMORY_PRESSURE = .95; // we use at most GPU_MEMORY_PRESSURE percent of the total gpu memory for the datasets

class GPUNeuralNetwork {
public:
	GPUNeuralNetwork();
	GPUNeuralNetwork(std::vector<struct layer_representation> v_layer, double eta);
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
	void createNetwork(std::vector<struct layer_representation> v_layer, double eta);
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
