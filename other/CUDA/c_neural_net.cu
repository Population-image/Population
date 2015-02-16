#include "popconfig.h"

#if defined(HAVE_CUDA)

#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "c_neural_net.h"
#include "popcuda.h"
#include "Population.h"
#include "microtime.h"

const int MAX_NB_THREADS = 1024; // GPU dependent

struct neural_network* createNetwork(std::vector<unsigned int> v_layer, double eta) {
	struct neural_network* network = new struct neural_network;

	network->_nb_layers = v_layer.size();
	network->_layers = new struct layer[network->_nb_layers];
	network->_eta = eta;

	for(unsigned int i=0;i<v_layer.size();i++){
		int size_layer = v_layer[i];
		struct layer& l = network->_layers[i];

		if(i != v_layer.size()-1) {
			// add a bias neuron with constant value 1
			l._X_size = size_layer+1;
		} else {
			// except for the last layer
			l._X_size = size_layer;
		}
		l._X = new pop::F32[l._X_size];
		for (unsigned int j=0; j<l._X_size; j++) {
			l._X[j] = 1;
		}
		l._d_E_X = NULL;

		l._Y_size = size_layer;
		l._Y = new pop::F32[l._Y_size];
		for (unsigned int j=0; j<l._Y_size; j++) {
			l._Y[j] = 0;
		}
		l._d_E_Y = NULL;

		if (i != 0) {
			unsigned int size_layer_previous = network->_layers[i-1]._X_size;
			pop::DistributionNormal n(0,1./std::sqrt(size_layer_previous));

			l._W_height = size_layer;
			l._W_width = size_layer_previous;
			l._W = new pop::F32[l._W_height * l._W_width];
			for (unsigned int j=0; j<l._W_height * l._W_width; j++) {
				l._W[j] = n.randomVariable();
			}
		} else {
			l._W_height = 0;
			l._W_width = 0;
			l._W = NULL;
		}
		l._d_E_W = NULL;

		l._errors_initialized = false;
	}

	//XOR network qui fonctionne
	{
		struct layer& l = network->_layers[0];
		l._X[0] = 1;
		l._X[1] = 1;
		l._X[2] = 1;

		l._Y[0] = 0;
		l._Y[1] = 0;
	}
	{
		struct layer& l = network->_layers[1];
		l._X[0] = -1.20591;
		l._X[1] = 1.68757;
		l._X[2] = -1.15248;
		l._X[3] = 1;

		l._Y[0] = -1.309180;
		l._Y[1] = 3.59145;
		l._Y[2] = -1.22061;

		l._W[0] = -0.836141;
		l._W[1] = -1.23238;
		l._W[2] = .75938;
		l._W[3] = 1.4026;
		l._W[4] = 1.09406;
		l._W[5] = 1.09479;
		l._W[6] = -.624209;
		l._W[7] = .542668;
		l._W[8] = -1.13907;
	}
	{
		struct layer& l = network->_layers[2];
		l._X[0] = -.997085;

		l._Y[0] = -.99615;

		l._W[0] = 1.2475;
		l._W[1] = 1.23825;
		l._W[2] = .886464;
		l._W[3] = -.559775;
	}

	return network;
}

void printNeuronsVector(pop::F32* V, unsigned int size, std::string label) {
	if (V == NULL) {
		std::cout << label << " = NULL" << std::endl;
	} else {
		std::cout << label << "(" << size << ") = [";
		for (unsigned int i=0; i<size; i++) {
			std::cout << "\t" << V[i];
		}
		std::cout << "\t]" << std::endl;
	}
}

void printWeightMatrix(pop::F32* M, unsigned int height, unsigned int width, std::string label) {
	if (M == NULL) {
		std::cout << label << " = NULL" << std::endl;
	} else {
		std::cout << label << "(" << height << ", " << width << ") = [" << std::endl;
		for (unsigned int i=0; i<height; i++) {
			for (unsigned int j=0; j<width; j++) {
				std::cout << "\t" << M[i*width + j];
			}
			std::cout << std::endl;
		}
		std::cout << "]" << std::endl;
	}
}

void printNetwork(struct neural_network* network) {
	std::cout << "Number of layers: " << network->_nb_layers << std::endl;
	std::cout << "Eta: " << network->_eta << std::endl;

	for (unsigned int l=0; l<network->_nb_layers; l++) {
		struct layer& layer = network->_layers[l];

		std::cout << "\n-- Layer " << l << ", error_initialized = " << (layer._errors_initialized ? "true" : "false") << ":" << std::endl;
		printNeuronsVector(layer._X, layer._X_size, "_X");
		printNeuronsVector(layer._Y, layer._Y_size, "_Y");
		printNeuronsVector(layer._d_E_X, layer._X_size, "_d_E_X");
		printNeuronsVector(layer._d_E_Y, layer._Y_size, "_d_E_Y");
		printWeightMatrix(layer._W, layer._W_height, layer._W_width, "_W");
		printWeightMatrix(layer._d_E_W, layer._W_height, layer._W_width, "_d_E_W");
	}
}

void propagateFront(struct neural_network* network, const pop::VecF32& in , pop::VecF32 &out) {
	std::copy(in.begin(),in.end(), network->_layers[0]._X);

	for (unsigned int l=0; l<network->_nb_layers-1; l++) {
		struct layer& prev_layer = network->_layers[l];
		struct layer& layer = network->_layers[l+1];

		// _Y[l+1] = _W[l+1] * _X[l]
		for (unsigned int i=0; i<layer._Y_size; i++) {
			layer._Y[i] = 0;
			for (unsigned int j=0; j<prev_layer._X_size; j++) {
				layer._Y[i] += layer._W[i*prev_layer._X_size+j] * prev_layer._X[j];
			}
		}

		// _X[l+1] = sigmoid(_Y[l+1])
		for (unsigned int i=0; i<layer._Y_size; i++) {
			layer._X[i] = sigmoid(layer._Y[i]);
		}
	}

	struct layer& last_layer = network->_layers[network->_nb_layers-1];
	if (out.size() != last_layer._X_size) {
		out.resize(last_layer._X_size);
	}
	std::copy(last_layer._X, last_layer._X+last_layer._X_size,out.begin());
}

void propagateBackFirstDerivate(struct neural_network* network, const pop::VecF32& desired_output) {
	for (unsigned int l=0; l<network->_nb_layers; l++) {
		struct layer& layer = network->_layers[l];
		if (!layer._errors_initialized) {
			layer._d_E_X = new pop::F32[layer._X_size];
			memcpy(layer._d_E_X, layer._X, sizeof(layer._X[0]) * layer._X_size);

			layer._d_E_Y = new pop::F32[layer._Y_size];
			memcpy(layer._d_E_Y, layer._Y, sizeof(layer._X[0]) * layer._Y_size);

			if (layer._W != NULL) {
				layer._d_E_W = new pop::F32[layer._W_height*layer._W_width];
				memcpy(layer._d_E_W, layer._W, sizeof(layer._W[0]) * layer._W_height*layer._W_width);
			}

			layer._errors_initialized = true;
		}
	}

	for (unsigned int l=network->_nb_layers-1; l>0; l--) {
		struct layer& layer = network->_layers[l];
		struct layer& prev_layer = network->_layers[l-1];

		// _d_E_X[l] = _X[l] - desired_output
		if (l == network->_nb_layers-1){
			for (unsigned int j=0; j<layer._X_size; j++) {
				layer._d_E_X[j] = layer._X[j] - desired_output[j];
			}
		}

		// _d_E_Y[l] = _d_E_X[l] * derived_sigmoid(_X[l])
		for (unsigned int j=0; j<layer._Y_size; j++) {
			layer._d_E_Y[j] = layer._d_E_X[j] * derived_sigmoid(layer._X[j]);
		}

		// _d_E_W[l-1] = _d_E_Y[l] * _X[l-1]
		// _W[l-1] = _W[l-1] - _eta * _d_E_W[l-1]
		for(unsigned int j=0; j<layer._W_width; j++){
			for (unsigned int i=0; i<layer._W_height; i++) {
				layer._d_E_W[i*layer._W_height+j] = layer._d_E_Y[i] * prev_layer._X[j];
				layer._W[i*layer._W_height+j] = layer._W[i*layer._W_height+j] - network->_eta*layer._d_E_W[i*layer._W_height+j];
			}
		}

		// _d_E_X[l-1][j] = sum_{i=0}^{_W[l-1].sizeI()}{_W[l](i, j) * _d_E_Y[l](i)}, j=0 to _X[l].size()
		for(unsigned int j=0; j<prev_layer._X_size; j++){
			prev_layer._d_E_X[j] = 0;
			for (unsigned int i=0; i<layer._W_height; i++) {
				prev_layer._d_E_X[j] += layer._W[i*layer._W_height+j] * layer._d_E_Y[i];
			}
		}
	}
}

void deleteNetwork(struct neural_network* network) {
	for (unsigned int i=0; i<network->_nb_layers; i++) {
		struct layer& l = network->_layers[i];

		delete[] l._X;
		if (l._d_E_X != NULL) {
			delete[] l._d_E_X;
		}

		delete[] l._Y;
		if (l._d_E_Y != NULL) {
			delete[] l._d_E_Y;
		}

		if (l._W != NULL) {
			delete[] l._W;
		}
		if (l._d_E_W != NULL) {
			delete[] l._d_E_W;
		}
	}
	delete[] network->_layers;
	delete network;
}

struct neural_network* copyNetworkToGPU(struct neural_network* h_net) {
	struct neural_network* d_net;

	// * in-memory representation on the gpu *
	// We allocate a big continuous array that will contain all the structures + values
	// [struct neural_network | struct layer 1 | struct layer 2 | ... | struct layer n | *_X | *_Y | *_W | *_d_E_X | *_d_E_Y | *_d_E_W |	 		  ...			 ]
	//																				   [  				for layer 1					   ][ for layer 2 ] [ for others ]

	unsigned int size = sizeof(*h_net) + h_net->_nb_layers * sizeof(h_net->_layers[0]);
	for (unsigned int i=0; i<h_net->_nb_layers; i++) {
		struct layer& layer = h_net->_layers[i];
		size += (layer._X_size + layer._Y_size) * 2 * sizeof(layer._X[0]);
		if (i!=0) {
			size += (layer._W_height + layer._W_width) * 2 * sizeof(layer._W[0]);
		}
	}
	cudaMalloc(&d_net, size);

	struct layer* p_layers =  h_net->_layers;
	h_net->_layers = (struct layer*)(d_net+1);
	cudaMemcpy(d_net, h_net, sizeof(*h_net), cudaMemcpyHostToDevice);
	h_net->_layers = p_layers;

	p_layers = (struct layer*)(d_net+1);
	pop::F32* start = (pop::F32*)((char*)d_net + sizeof(*d_net) + h_net->_nb_layers * sizeof(*p_layers));
	for (unsigned int i=0; i<h_net->_nb_layers; i++) {
		struct layer& layer = h_net->_layers[i];

		pop::F32* p_X = layer._X;
		pop::F32* p_Y = layer._Y;
		pop::F32* p_W = layer._W;
		pop::F32* p_d_E_X = layer._d_E_X;
		pop::F32* p_d_E_Y = layer._d_E_Y;
		pop::F32* p_d_E_W = layer._d_E_W;

		layer._X = start;
		layer._Y = layer._X + layer._X_size;
		layer._W = layer._Y + layer._Y_size;

		layer._d_E_X = layer._W + layer._W_height*layer._W_width;
		layer._d_E_Y = layer._d_E_X + layer._X_size;
		layer._d_E_W = layer._d_E_Y + layer._Y_size;

		// Note: we do not need to copy the errors vectors (i.e., d_E_*), as they will be initialized during the propagateBack algorithm
		cudaMemcpy(layer._X, p_X, sizeof(*p_X) * layer._X_size, cudaMemcpyHostToDevice);
		cudaMemcpy(layer._Y, p_Y, sizeof(*p_Y) * layer._Y_size, cudaMemcpyHostToDevice);
		if (i!=0) {
			cudaMemcpy(layer._W, p_W, sizeof(*p_W) * layer._W_height*layer._W_width, cudaMemcpyHostToDevice);
		} else {
			layer._W = NULL;
		}
		cudaMemcpy(p_layers, &layer, sizeof(*p_layers), cudaMemcpyHostToDevice);

		start = layer._d_E_W + layer._W_height*layer._W_width;

		layer._X = p_X;
		layer._Y = p_Y;
		layer._W = p_W;
		layer._d_E_X = p_d_E_X;
		layer._d_E_Y = p_d_E_Y;
		layer._d_E_W = p_d_E_W;

		p_layers++;
	}

	return d_net;
}

struct neural_network* copyNetworkFromGPU(struct neural_network* d_net) {
	struct neural_network* h_net = new struct neural_network;

	cudaMemcpy(h_net, d_net, sizeof(*h_net), cudaMemcpyDeviceToHost);

	struct layer* p_layers =  h_net->_layers;
	h_net->_layers = new struct layer[h_net->_nb_layers];

	pop::F32* start = (pop::F32*)((char*)d_net + sizeof(*d_net) + h_net->_nb_layers * sizeof(*p_layers));
	for (unsigned int i=0; i<h_net->_nb_layers; i++) {
		struct layer& layer = h_net->_layers[i];

		cudaMemcpy(&layer, &p_layers[i], sizeof(*p_layers), cudaMemcpyDeviceToHost);

		layer._X = new pop::F32[layer._X_size];
		cudaMemcpy(layer._X, start, sizeof(layer._X[0])*layer._X_size, cudaMemcpyDeviceToHost);
		start += layer._X_size;

		layer._Y = new pop::F32[layer._Y_size];
		cudaMemcpy(layer._Y, start, sizeof(layer._Y[0])*layer._Y_size, cudaMemcpyDeviceToHost);
		start += layer._Y_size;

		if (i!=0) {
			layer._W = new pop::F32[layer._W_height * layer._W_width];
			cudaMemcpy(layer._W, start, sizeof(layer._W[0])*layer._W_height*layer._W_width, cudaMemcpyDeviceToHost);
			start += layer._W_height * layer._W_width;
		} else {
			layer._W_height = 0;
			layer._W_width = 0;
			layer._W = NULL;
		}

		// Note: we do not need to copy the errors vectors (i.e., d_E_*), as they will be initialized during the propagateBack algorithm
		layer._d_E_X = NULL;
		layer._d_E_Y = NULL;
		layer._d_E_W = NULL;
		layer._errors_initialized = false;

		start += layer._X_size + layer._Y_size + layer._W_height*layer._W_width;
	}

	return h_net;

}

void deleteNetworkOnGPU(struct neural_network* network) {
	cudaFree(network);
}

const char* cublasGetErrorString(cublasStatus_t status)
{
    switch(status)
    {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
    }
    return "unknown error";
}

__device__ void printVectorOnGPU(pop::F32* V, unsigned int size, char* label) {
	printf("%s = [", label);
	for (unsigned int i=0; i<size; i++) {
		printf(" %f", V[i]);
	}
	printf("]\n");
}

__device__ void printMatrixOnGPU(pop::F32* M, unsigned int height, unsigned int width, char* label) {
	printf("%s = [\n", label);
	for (unsigned int i=0; i<height; i++) {
		for (unsigned int j=0; j<width; j++) {
			printf(" %f", M[i*width + j]);
		}
		printf("\n");
	}
	printf("]\n");
}

__global__ void printNetworkOnGPU(struct neural_network *network) {
	printf("Number of layers: %d, eta: %f\n", network->_nb_layers, network->_eta);

	for (unsigned int l=0; l<network->_nb_layers; l++) {
		struct layer& layer = network->_layers[l];
		printf("Layer %d, _X_size = %d, _Y_size = %d, _W_height = %d, _W_width = %d, _error_initialized = %s\n", l, layer._X_size, layer._Y_size, layer._W_height, layer._W_width, (layer._errors_initialized ? "true" : "false"));

		printVectorOnGPU(layer._X, layer._X_size, (char*)"_X");
		printVectorOnGPU(layer._Y, layer._Y_size, (char*)"_Y");
		printMatrixOnGPU(layer._W, layer._W_height, layer._W_width, (char*)"_W");
		printVectorOnGPU(layer._d_E_X, layer._X_size, (char*)"_d_E_X");
		printVectorOnGPU(layer._d_E_Y, layer._Y_size, (char*)"_d_E_Y");
		printMatrixOnGPU(layer._d_E_W, layer._W_height, layer._W_width, (char*)"_d_E_W");
	}
}

__global__ void propagateFrontGPUSetInput(struct neural_network *network, pop::F32* in_set, unsigned int in_elt_size, unsigned int idx) {
	int tid = blockDim.x*blockIdx.x + threadIdx.x;
	if (tid < in_elt_size) {
		network->_layers[0]._X[tid] = in_set[idx*in_elt_size+tid];
		printf("tid = %d, idx = %d, in = %.2f\n", tid, idx, in_set[idx*in_elt_size+tid]);
	}
}

__global__ void propagateFrontGPUComputeSigmoid(struct neural_network *network, int l) {
	int tid = blockDim.x*blockIdx.x + threadIdx.x;
	if (tid < network->_layers[l]._Y_size) {
		network->_layers[l]._X[tid] = 1.7159f*tanhf(0.66666667f*network->_layers[l]._Y[tid]);
	}
}

__global__ void propagateFrontGPUSetOutput(struct neural_network *network, pop::F32* out_computed) {
	int tid = blockDim.x*blockIdx.x + threadIdx.x;
	if (tid < network->_layers[network->_nb_layers-1]._X_size) {
		out_computed[tid] = network->_layers[network->_nb_layers-1]._X[tid];
		printf("tid = %d, out = %.2f\n", tid, out_computed[tid]);
	}
}

/*
 * Propagate in_set[idx] in out_computed using network h_network (CPU) or d_network (GPU)
 * in_set: set of all the inputs. Each element's size is in_elt_size
 * out_computed: the output element, of size equal to the number of neurons in the last layer
 */
void propagateFrontGPU(struct neural_network *h_network, struct neural_network *d_network, pop::F32* in_set, unsigned int in_elt_size, unsigned int idx, pop::F32* out_computed) {
	int block, grid;

	block = (h_network->_layers[0]._X_size < MAX_NB_THREADS ? h_network->_layers[0]._X_size : MAX_NB_THREADS);
    grid = h_network->_layers[0]._X_size / MAX_NB_THREADS + (h_network->_layers[0]._X_size%MAX_NB_THREADS ? 1 : 0);
	propagateFrontGPUSetInput<<<grid, block>>>(d_network, in_set, in_elt_size, idx);
	cudaDeviceSynchronize();

	cublasStatus_t	stat;
	cublasHandle_t handle;
	cublasCreate(&handle);
	float alpha = 1.0f;
	float beta = 0.0f;
	pop::F32* start = (pop::F32*)((char*)d_network + sizeof(*d_network) + h_network->_nb_layers * sizeof(h_network->_layers[0]));

	for (unsigned int l=0; l<h_network->_nb_layers-1; l++) {
		struct layer& prev_layer = h_network->_layers[l];
		struct layer& layer = h_network->_layers[l+1];

		pop::F32* d_X = start;
		start += (prev_layer._X_size + prev_layer._Y_size + prev_layer._W_height*prev_layer._W_width)*2; // d_Y and d_W are from the next layer
		pop::F32* d_Y = start + layer._X_size;
		pop::F32* d_W = d_Y + layer._Y_size;

		// _Y[l+1] = _W[l+1] * _X[l]
		stat = cublasSgemv_v2(handle, CUBLAS_OP_T, layer._W_width, layer._W_height, &alpha, d_W, prev_layer._X_size, d_X, 1, &beta, d_Y, 1);
		if (stat != CUBLAS_STATUS_SUCCESS) {
			std::cout << "Cublas error in _Y[l+1] = _W[l+1] * _X[l] for layer l = " << l << ", cublas status: " << cublasGetErrorString(stat) << std::endl;
		}

		// _X[l+1] = sigmoid(_Y[l+1])
		block = (h_network->_layers[l+1]._X_size < MAX_NB_THREADS ? h_network->_layers[l+1]._X_size : MAX_NB_THREADS);
	    grid = h_network->_layers[l+1]._X_size / MAX_NB_THREADS + (h_network->_layers[l+1]._X_size%MAX_NB_THREADS ? 1 : 0);
		propagateFrontGPUComputeSigmoid<<<grid, block>>>(d_network, l+1);
		cudaDeviceSynchronize();
	}

	cublasDestroy(handle);

	block = (h_network->_layers[h_network->_nb_layers-1]._X_size < MAX_NB_THREADS ? h_network->_layers[h_network->_nb_layers-1]._X_size : MAX_NB_THREADS);
    grid = h_network->_layers[h_network->_nb_layers-1]._X_size / MAX_NB_THREADS + (h_network->_layers[h_network->_nb_layers-1]._X_size%MAX_NB_THREADS ? 1 : 0);
	propagateFrontGPUSetOutput<<<grid, block>>>(d_network, out_computed);
}

#if 0
/*
 * Propagate back diff(out_set[idx], out_computed) using network network
 * out_set: set of all the inputs. Size = out_set_size. Each element's size is out_elt_size
 * out_computed: the output element computed previously (using propagateFrontGPU), of size out_elt_size
 */
void propagateBackFirstDerivativeGPU(struct neural_network *network, pop::F32* out_set, unsigned int out_set_size, unsigned int out_elt_size, unsigned int idx, pop::F32* out_computed) {
	//TODO
}

/*
 * increase *error if out_set != out_computed[idx]
 */
__global__ void computeErrorGPU(pop::F32* out_set, pop::F32* out_computed, unsigned int out_set_size, unsigned int out_elt_size, unsigned int idx, int* error) {
	//TODO
}
#endif

void test_neural_net(void) {
	struct neural_network* network;

	std::vector<unsigned int> v_layer;
	v_layer.push_back(2);
	v_layer.push_back(3);
	v_layer.push_back(1);
	network = createNetwork(v_layer, 0.01);

	struct neural_network* d_network = copyNetworkToGPU(network);

	std::cout << "the neural network is:" << std::endl;
	printNetworkOnGPU<<<1, 1>>>(d_network);

	//create the training set
	pop::Vec<pop::VecF32> v_in(4,pop::VecF32(2));//4 vector of two scalar values
	v_in(0)(0)=-1;v_in(0)(1)=-1; // (-1,-1)
	v_in(1)(0)= 1;v_in(1)(1)=-1; // ( 1,-1)
	v_in(2)(0)=-1;v_in(2)(1)= 1; // (-1, 1)
	v_in(3)(0)= 1;v_in(3)(1)= 1; // ( 1, 1)
	pop::Vec<pop::VecF32> v_out(4,pop::VecF32(1));//4 vector of one scalar value
	v_out(0)(0)=-1;// -1
	v_out(1)(0)= 1;//  1
	v_out(2)(0)= 1;//  1
	v_out(3)(0)=-1;// -1

	size_t total_size_sets = (v_in.size()*v_in(0).size() + v_out.size()*v_out(0).size()) * sizeof(v_in(0)(0));
	size_t free, total;
	cudaMemGetInfo(&free, &total);
	if (total_size_sets > .9*free) { // 90% of the free memory
		std::cerr << "Not enough memory on the GPU to process the whole sets at once. You need to copy the sets pieces by pieces" << std::endl;
		deleteNetworkOnGPU(d_network);
		return;
	}

	//use the backpropagation algorithm with first order method
	std::vector<int> v_global_rand(v_in.size());
	for(unsigned int i=0;i<v_global_rand.size();i++)
		v_global_rand[i]=i;
	std::cout<<"iter_epoch\t error_train"<<std::endl;

	pop::F32* d_in_set;
	cudaMalloc(&d_in_set, v_in.size()*v_in(0).size() * sizeof(v_in(0)(0)));
	pop::F32* start = d_in_set;
	for (int i=0; i<v_in.size(); i++) {
		for (int j=0; j<v_in(i).size(); j++) {
			cudaMemcpy(start, &v_in(i)(j), sizeof(*d_in_set), cudaMemcpyHostToDevice);
			start++;
		}
	}

	pop::F32* d_out_set;
	cudaMalloc(&d_out_set, v_out.size()*v_out(0).size() * sizeof(v_in(0)(0)));
	start = d_out_set;
	for (int i=0; i<v_out.size(); i++) {
		for (int j=0; j<v_out(i).size(); j++) {
			cudaMemcpy(start, &v_out(i)(j), sizeof(*d_out_set), cudaMemcpyHostToDevice);
			start++;
		}
	}

	pop::F32* d_out;
	cudaMalloc(&d_out, v_out(0).size() * sizeof(v_in(0)(0)));

	int error;
	int* d_error;
	cudaMalloc(&d_error, sizeof(error));

	unsigned int nbr_epoch = 1;
	for(unsigned int i=0;i<nbr_epoch;i++){
		std::random_shuffle ( v_global_rand.begin(), v_global_rand.end() , pop::Distribution::irand());

		error = 0;
		cudaMemcpy(d_error, &error, sizeof(error), cudaMemcpyHostToDevice);

		for(unsigned int j=0;j<v_global_rand.size();j++){
			propagateFrontGPU(network, d_network, d_in_set, v_in(0).size(), v_global_rand[j], d_out);
#if 0
			propagateBackFirstDerivativeGPU(d_network, d_out_set, v_out.size(), v_out(0).size(), v_global_rand[j], d_out);
			computeErrorGPU<<<1, 1>>>(d_out_set, d_out, v_out(0).size(), v_global_rand[j], d_error);
#endif
		}

		cudaMemcpy(&error, d_error, sizeof(error), cudaMemcpyDeviceToHost);
		std::cout<<i<<"\t"<<error*1.0/v_global_rand.size()<<std::endl;

		cudaFree(d_error);
		cudaFree(d_out);
		cudaFree(d_in_set);
		cudaFree(d_out_set);
	}

	deleteNetwork(network);
	network = copyNetworkFromGPU(d_network);

	//TODO: print? save?

	deleteNetworkOnGPU(d_network);
	deleteNetwork(network);

	return;
}

void test_neural_net_cpu(void) {
	struct neural_network* network;

	std::vector<unsigned int> v_layer;
	v_layer.push_back(2);
	v_layer.push_back(3);
	v_layer.push_back(1);
	network = createNetwork(v_layer, 0.01);

	//create the training set
	// (-1,-1)->-1
	// ( 1,-1)-> 1
	// (-1, 1)-> 1
	// ( 1, 1)->-1
	pop::Vec<pop::VecF32> v_in(4,pop::VecF32(2));//4 vector of two scalar values
	v_in(0)(0)=-1;v_in(0)(1)=-1; // (-1,-1)
	v_in(1)(0)= 1;v_in(1)(1)=-1; // ( 1,-1)
	v_in(2)(0)=-1;v_in(2)(1)= 1; // (-1, 1)
	v_in(3)(0)= 1;v_in(3)(1)= 1; // ( 1, 1)

	pop::Vec<pop::VecF32> v_out(4,pop::VecF32(1));//4 vector of one scalar value
	v_out(0)(0)=-1;// -1
	v_out(1)(0)= 1;//  1
	v_out(2)(0)= 1;//  1
	v_out(3)(0)=-1;// -1

	//use the backpropagation algorithm with first order method
	std::vector<int> v_global_rand(v_in.size());
	for(unsigned int i=0;i<v_global_rand.size();i++)
		v_global_rand[i]=i;
	std::cout<<"iter_epoch\t error_train"<<std::endl;

	unsigned int nbr_epoch = 1000;
	for(unsigned int i=0;i<nbr_epoch;i++){
		std::random_shuffle ( v_global_rand.begin(), v_global_rand.end() , pop::Distribution::irand());
		int error=0;
		for(unsigned int j=0;j<v_global_rand.size();j++){
			pop::VecF32 vout;
			propagateFront(network, v_in(v_global_rand[j]),vout);
			propagateBackFirstDerivate(network, v_out(v_global_rand[j]));

			int label1 = std::distance(vout.begin(),std::max_element(vout.begin(),vout.end()));
			int label2 = std::distance(v_out(v_global_rand[j]).begin(),std::max_element(v_out(v_global_rand[j]).begin(),v_out(v_global_rand[j]).end()));
			if(label1!=label2)
				error++;
		}

		std::cout<<i<<"\t"<<error*1.0/v_global_rand.size()<<std::endl;
	}

	//test the training
	for(int j=0;j<4;j++){
		pop::VecF32 vout;
		propagateFront(network, v_in(j), vout);
		std::cout<<vout<<std::endl;// we obtain the expected value -1 , 1 , 1 , -1
	}

	printNetwork(network);

	deleteNetwork(network);

	//MNIST neural net
#if 0
	pop::Vec<pop::Vec<pop::Mat2UI8> > number_training =  pop::TrainingNeuralNetwork::loadMNIST("/media/pl/shared/PL/neural_nets_samples/MNIST/train-images-idx3-ubyte","/media/pl/shared/PL/neural_nets_samples/MNIST/train-labels-idx1-ubyte");
	pop::Vec<pop::Vec<pop::Mat2UI8> > number_test =  pop::TrainingNeuralNetwork::loadMNIST("/media/pl/shared/PL/neural_nets_samples/MNIST/t10k-images-idx3-ubyte","/media/pl/shared/PL/neural_nets_samples/MNIST/t10k-labels-idx1-ubyte");

	double size_in= number_training(0)(0).getDomain()(0) * number_training(0)(0).getDomain()(1);
	std::cout << "size trainings: " << number_training(0).size() << std::endl;

	std::vector<unsigned int> v_layer;
	v_layer.push_back(size_in);
	v_layer.push_back(1000);
	v_layer.push_back(1000);
	v_layer.push_back(number_training.size());
	network = createNetwork(v_layer, 0.001);

	pop::Vec<pop::VecF32> vtraining_in;
	pop::Vec<pop::VecF32> vtraining_out;

	double ratio = 1;
	pop::TrainingNeuralNetwork::convertMatrixToInputValueNeuron(vtraining_in,vtraining_out,number_training,number_training(0)(0).getDomain(),pop::NNLayerMatrix::Mass,pop::NNLayerMatrix::MinusOneToOne,ratio);

	pop::Vec<pop::VecF32> vtest_in;
	pop::Vec<pop::VecF32> vtest_out;
	pop::TrainingNeuralNetwork::convertMatrixToInputValueNeuron(vtest_in,vtest_out,number_test,number_training(0)(0).getDomain(),pop::NNLayerMatrix::Mass,pop::NNLayerMatrix::MinusOneToOne,1);

	number_training.clear();
	number_test.clear();

	std::vector<int> v_global_rand(vtraining_in.size());
	for(unsigned int i=0;i<v_global_rand.size();i++)
		v_global_rand[i]=i;

	std::cout<<"iter_epoch\t error_train\t error_test\t learning rate"<<std::endl;

	for(unsigned int i=0;i<100;i++){
		std::random_shuffle ( v_global_rand.begin(), v_global_rand.end() ,pop::Distribution::irand());
		int error_training=0,error_test=0;

		for(unsigned int j=0;j<v_global_rand.size();j++){
			pop::VecF32 vout;
			propagateFront(network, vtraining_in(v_global_rand[j]),vout);
			int label1 = std::distance(vout.begin(),std::max_element(vout.begin(),vout.end()));
			propagateBackFirstDerivate(network, vtraining_out(v_global_rand[j]));
			int label2 = std::distance(vtraining_out(v_global_rand[j]).begin(),std::max_element(vtraining_out(v_global_rand[j]).begin(),vtraining_out(v_global_rand[j]).end()));
			if(label1!=label2){
				error_training++;
			}
		}
		for(unsigned int j=0;j<vtest_in.size();j++){
			pop::VecF32 vout;
			propagateFront(network, vtest_in(j),vout);
			int label1 = std::distance(vout.begin(),std::max_element(vout.begin(),vout.end()));
			int label2 = std::distance(vtest_out(j).begin(),std::max_element(vtest_out(j).begin(),vtest_out(j).end()));
			if(label1!=label2){
				error_test++;
			}
		}
		network->_eta *=0.9;
		std::cout<<i<<"\t"<<error_training*1./v_global_rand.size()<<"\t"<<error_test*1./vtest_in.size() <<"\t"<<network->_eta <<std::endl;
	}

	deleteNetwork(network);
#endif
}

void test_cublas(void) {
	cublasStatus_t	stat;
	cublasHandle_t handle;

	const int width = 3;
	const int height = 2;
	float* W = new float[width*height];
	for (int i=0; i<width*height; i++) {
		W[i] = i;
	}

	std::cout << "W = [";
	for (int i=0; i<width*height; i++) {
		std::cout << " " << W[i];
	}
	std::cout << " ]" << std::endl;

	float* X = new float[width];
	X[0] = 2;
	for (int i=1; i<width; i++) {
		X[i] = 1;
	}

	std::cout << "X = [";
	for (int i=0; i<width; i++) {
		std::cout << " " << X[i];
	}
	std::cout << " ]" << std::endl;

	float* Y = new float[height];
	for (int i=1; i<height; i++) {
		Y[i] = 0;
	}

	float* d_W;
	cudaMalloc(&d_W, width*height*sizeof(*d_W));
	cudaMemcpy(d_W, W, width*height*sizeof(*d_W), cudaMemcpyHostToDevice);

	float* d_X;
	cudaMalloc(&d_X, width*sizeof(*d_X));
	cudaMemcpy(d_X, X, width*sizeof(*d_X), cudaMemcpyHostToDevice);

	float* d_Y;
	cudaMalloc(&d_Y, height*sizeof(*d_Y));

	cublasCreate(&handle);

	float alpha = 1.0f;
	float beta = 0.0f;
	//  Y = α op(W) X + β Y
	stat = cublasSgemv_v2(handle, CUBLAS_OP_T, width, height, &alpha, d_W, width, d_X, 1, &beta, d_Y, 1);	// Y = [ 3 15 ]
	std::cout << "cublas status: " << cublasGetErrorString(stat) << std::endl;

	cudaMemcpy(Y, d_Y, height*sizeof(*d_Y), cudaMemcpyDeviceToHost);

	std::cout << "Y = [";
	for (int i=0; i<height; i++) {
		std::cout << " " << Y[i];
	}
	std::cout << " ]" << std::endl;

	cublasDestroy(handle);
	cudaFree(d_W);
	cudaFree(d_X);
	cudaFree(d_Y);

	delete[] Y;
	delete[] X;
	delete[] W;
}

#endif
