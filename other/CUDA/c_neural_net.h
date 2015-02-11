#ifndef C_NEURAL_NET_H
#define C_NEURAL_NET_H

#include "popconfig.h"

#include "data/typeF/TypeF.h"
#include "data/neuralnetwork/NeuralNetwork.h"

void test_neural_net(void);

struct layer {
	unsigned int _X_size; // size of _X and _d_E_X
	unsigned int _Y_size; // size of _Y and _d_E_Y
	unsigned int _W_width; // width of _W and _d_E_W
	unsigned int _W_height; // height of _W and _d_E_W
	pop::F32* _X;
	pop::F32* _Y;
	pop::F32** _W;
	pop::F32* _d_E_X;
	pop::F32* _d_E_Y;
	pop::F32** _d_E_W;
};

struct neural_network {
	double _eta;
	unsigned int _nb_layers;
	struct layer* layers;
};

static float sigmoid(float x){ return 1.7159f*tanh(0.66666667f*x); }
static float derived_sigmoid(float S){ return 0.666667f/1.7159f*(1.7159f*1.7159f-S*S); }

struct neural_network* createNetwork(std::vector<unsigned int> v_layer, double eta);
void propagateFront(struct neural_network* network, const pop::VecF32& in , pop::VecF32 &out);
void propagateBackFirstDerivate(struct neural_network* network, const pop::VecF32& desired_output);
void deleteNetwork(struct neural_network* network);

struct neural_network* copyNetworkToGPU(struct neural_network* h_net);
struct neural_network* copyNetworkFromGPU(struct neural_network* d_net);
void deleteNetworkOnGPU(struct neural_network* network);

void printNetwork(struct neural_network* network);

#if defined(HAVE_CUDA)
#endif

#endif
