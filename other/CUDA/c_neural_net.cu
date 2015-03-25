#include "popconfig.h"

#include <iostream>
#include <time.h>

#if defined(HAVE_CUDA)
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#endif

#include "c_neural_net.h"
#include "popcuda.h"
#include "Population.h"
#include "microtime.h"
#include "data/utility/BasicUtility.h"

// slower when using cublas
//#define SET_W_ERROR_CUBLAS

// if defined, then loadDatabase() will process a batch of images instead of 1 image at a time. May improve disk access
#define BATCH_LOADING

struct layer {
	unsigned int _X_size; // size of _X and _d_E_X
	unsigned int _Y_size; // size of _Y and _d_E_Y
	unsigned int _W_width; // width of _W and _d_E_W
	unsigned int _W_height; // height of _W and _d_E_W
	pop::F32* _X;
	pop::F32* _Y;
	pop::F32* _W;
	pop::F32* _d_E_X;
	pop::F32* _d_E_Y;
	pop::F32* _d_E_W;
};

struct neural_network {
	double _eta;
	unsigned int _nb_layers;
	struct layer* _layers;
};

const float GPU_MEMORY_PRESSURE = .95; // we use at most GPU_MEMORY_PRESSURE percent of the total gpu memory for the datasets

static std::string getCurrentTime() {
	time_t rawtime;
	time(&rawtime);
	struct tm * timeinfo = localtime(&rawtime);
	std::string s = asctime(timeinfo);
	s.erase(s.length()-1);
	return s;
}

GPUNeuralNetwork::GPUNeuralNetwork() {
	h_network = NULL;
#if defined(HAVE_CUDA)
	d_network = NULL;
#endif
}

GPUNeuralNetwork::GPUNeuralNetwork(std::vector<unsigned int> v_layer, double eta) {
	createNetwork(v_layer, eta);
#if defined(HAVE_CUDA)
	copyNetworkToGPU();
#endif
}

GPUNeuralNetwork::~GPUNeuralNetwork() {
	deleteNetwork();
#if defined(HAVE_CUDA)
	deleteNetworkOnGPU();
#endif
}


void GPUNeuralNetwork::createNetwork(std::vector<unsigned int> v_layer, double eta) {
	h_network = new struct neural_network;

	h_network->_nb_layers = v_layer.size();
	h_network->_layers = new struct layer[h_network->_nb_layers];
	h_network->_eta = eta;

	for(unsigned int i=0;i<v_layer.size();i++){
		int size_layer = v_layer[i];
		struct layer& l = h_network->_layers[i];

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
			unsigned int size_layer_previous = h_network->_layers[i-1]._X_size;
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
	}
}

void GPUNeuralNetwork::deleteNetwork() {
	if (h_network == NULL) {
		return;
	}

	for (unsigned int i=0; i<h_network->_nb_layers; i++) {
		struct layer& l = h_network->_layers[i];

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
	delete[] h_network->_layers;
	delete h_network;
}


void GPUNeuralNetwork::printNeuronsVector(pop::F32* V, unsigned int size, std::string label) {
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

void GPUNeuralNetwork::printWeightMatrix(pop::F32* M, unsigned int height, unsigned int width, std::string label) {
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

void GPUNeuralNetwork::displayNetwork() {
	std::cout << "Number of layers: " << h_network->_nb_layers << ", eta: " << h_network->_eta << std::endl;

	for (unsigned int l=0; l<h_network->_nb_layers; l++) {
		struct layer& layer = h_network->_layers[l];

		std::cout << "\n-- Layer " << l << ", _X_size = " << layer._X_size << ", Y_size = " << layer._Y_size << ", _W_height = " << layer._W_height << ", _W_width = " << layer._W_width << std::endl;
		printNeuronsVector(layer._X, layer._X_size, "_X");
		printNeuronsVector(layer._Y, layer._Y_size, "_Y");
		printWeightMatrix(layer._W, layer._W_height, layer._W_width, "_W");
		printNeuronsVector(layer._d_E_X, layer._X_size, "_d_E_X");
		printNeuronsVector(layer._d_E_Y, layer._Y_size, "_d_E_Y");
		printWeightMatrix(layer._d_E_W, layer._W_height, layer._W_width, "_d_E_W");
	}
}

// save the cpu version of the neural network to the file filename
void GPUNeuralNetwork::save(std::string filename) {
	std::ofstream out(filename.c_str(), std::ios::out | std::ios::binary);
	out.write((char*)&h_network->_nb_layers,sizeof(h_network->_nb_layers));
	out.write((char*)&h_network->_eta,sizeof(h_network->_eta));

	for (unsigned int l=0; l<h_network->_nb_layers; l++) {
		struct layer& layer = h_network->_layers[l];
		out.write((char*)&l,sizeof(l));
		out.write((char*)&layer._X_size,sizeof(layer._X_size));
		out.write((char*)&layer._Y_size,sizeof(layer._Y_size));
		out.write((char*)&layer._W_height,sizeof(layer._W_height));
		out.write((char*)&layer._W_width,sizeof(layer._W_width));
		for (unsigned int i=0; i<layer._W_height*layer._W_width; i++) {
			out.write((char*)&layer._W[i],sizeof(layer._W[i]));
		}
	}

	out.flush();
	out.close();
}

// load the cpu version of the neural network from the file filename.
// If you use the GPU, then you need to copy it to the GPU by calling copyNetworkToGPU()
void GPUNeuralNetwork::load(std::string filename) {
	std::vector<unsigned int> layers;
	std::vector<pop::VecF32> weights;

	std::ifstream in(filename.c_str(), std::ios::in | std::ios::binary);
	if (!in.is_open()) {
		std::cerr << "GPUNeuralNetwork::load(): unable to open file " << filename << std::endl;
		exit(-1);
	}

	unsigned int nb_layers;
	double eta;
	in.read((char*)&nb_layers, sizeof(nb_layers));
	in.read((char*)&eta, sizeof(eta));

	for (unsigned int l=0; l<nb_layers; l++) {
		unsigned int ll, X_size, Y_size, W_height, W_width;
		in.read((char*)&ll, sizeof(ll));
		in.read((char*)&X_size, sizeof(X_size));
		in.read((char*)&Y_size, sizeof(Y_size));
		in.read((char*)&W_height, sizeof(W_height));
		in.read((char*)&W_width, sizeof(W_width));
		if (ll != l) {
			std::cerr << "GPUNeuralNetwork::load(): wrong layer: " << ll << " instead of " << l << std::endl;
			exit(-1);
		}
		layers.push_back(Y_size);

		pop::VecF32	v_w;
		for (unsigned int i=0; i<W_height*W_width; i++) {
			pop::F32 w;
			in.read((char*)&w, sizeof(w));
			v_w.push_back(w);
		}
		weights.push_back(v_w);
	}
	in.close();

	createNetwork(layers, eta);

	for (int l=1; l<nb_layers; l++) {
		for (int i=0; i<weights[l].size(); i++) {
			h_network->_layers[l]._W[i] = weights[l](i);
		}
	}
}

void GPUNeuralNetwork::setEta(const double eta) {
	h_network->_eta = eta;

#if defined(HAVE_CUDA)
	int eta_offset = (char*)&(h_network->_eta) - (char*)h_network;
	cudaMemcpy(d_network + eta_offset, &(h_network->_eta), sizeof(eta), cudaMemcpyHostToDevice);
#endif
}

double GPUNeuralNetwork::getEta() const {
	return h_network->_eta;
}

void GPUNeuralNetwork::propagateFront(const pop::VecF32& in , pop::VecF32 &out) {
	std::copy(in.begin(),in.end(), h_network->_layers[0]._X);

	for (unsigned int l=0; l<h_network->_nb_layers-1; l++) {
		struct layer& prev_layer = h_network->_layers[l];
		struct layer& layer = h_network->_layers[l+1];

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

	struct layer& last_layer = h_network->_layers[h_network->_nb_layers-1];
	if (out.size() != last_layer._X_size) {
		out.resize(last_layer._X_size);
	}
	std::copy(last_layer._X, last_layer._X+last_layer._X_size,out.begin());
}

void GPUNeuralNetwork::propagateBackFirstDerivate(const pop::VecF32& desired_output) {
	for (unsigned int l=0; l<h_network->_nb_layers; l++) {
		struct layer& layer = h_network->_layers[l];
		if (layer._d_E_X == NULL) {
			layer._d_E_X = new pop::F32[layer._X_size];
			memcpy(layer._d_E_X, layer._X, sizeof(layer._X[0]) * layer._X_size);
		}
		if (layer._d_E_Y == NULL) {
			layer._d_E_Y = new pop::F32[layer._Y_size];
			memcpy(layer._d_E_Y, layer._Y, sizeof(layer._X[0]) * layer._Y_size);
		}
		if (layer._W != NULL && layer._d_E_W == NULL) {
			layer._d_E_W = new pop::F32[layer._W_height*layer._W_width];
			memcpy(layer._d_E_W, layer._W, sizeof(layer._W[0]) * layer._W_height*layer._W_width);
		}
	}

	for (unsigned int l=h_network->_nb_layers-1; l>0; l--) {
		struct layer& layer = h_network->_layers[l];
		struct layer& prev_layer = h_network->_layers[l-1];

		// _d_E_X[l] = _X[l] - desired_output
		if (l == h_network->_nb_layers-1){
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
				int idx = i*layer._W_width+j;
				layer._d_E_W[idx] = layer._d_E_Y[i] * prev_layer._X[j];
				layer._W[idx] = layer._W[idx] - h_network->_eta*layer._d_E_W[idx];
			}
		}

		// _d_E_X[l-1][j] = sum_{i=0}^{_W[l-1].sizeI()}{_W[l](i, j) * _d_E_Y[l](i)}, j=0 to _X[l].size()
		for(unsigned int j=0; j<prev_layer._X_size; j++){
			prev_layer._d_E_X[j] = 0;
			for (unsigned int i=0; i<layer._W_height; i++) {
				prev_layer._d_E_X[j] += layer._W[i*layer._W_width+j] * layer._d_E_Y[i];
			}
		}
	}
}

void loadDatabase(std::string directory, const int max_per_folder, pop::Vec<pop::VecF32> &v_neuron_in, pop::Vec<pop::VecF32> &v_neuron_out) {
	pop::Vec2I32 domain(29,29);

	std::vector<std::string> content = pop::BasicUtility::getFilesInDirectory(directory);
	int nb_dirs = 0;
	for (int i=0; i<content.size(); i++) {
		pop::Vec<pop::Mat2UI8> images;
		if (pop::BasicUtility::isDirectory(directory + pop::BasicUtility::getPathSeparator() + content[i])) {
			nb_dirs++;
		}
	}

	for (int i=0; i<content.size(); i++) {
		if (pop::BasicUtility::isDirectory(directory + pop::BasicUtility::getPathSeparator() + content[i])) {
			std::vector<std::string> inner_content = pop::BasicUtility::getFilesInDirectory(directory + pop::BasicUtility::getPathSeparator() + content[i]);
			//std::cout << directory + pop::BasicUtility::getPathSeparator() + content[i] << ": " << inner_content.size() << " images" << std::endl;
			std::cout << content[i] << std::flush;
#ifdef BATCH_LOADING
			pop::Vec<pop::Mat2UI8> mat_tmp;
#endif
			int nb_images = 0;
			for (int j=0; j<inner_content.size() && nb_images < max_per_folder; j++) {
				std::string filename = directory + pop::BasicUtility::getPathSeparator() + content[i] + pop::BasicUtility::getPathSeparator() + inner_content[j];
				if (!pop::BasicUtility::isFile(filename)) {
					continue;
				}

				//std::cout << "Reading file " << content[i] << pop::BasicUtility::getPathSeparator() << inner_content[j] << std::endl;
				std::string ext = pop::BasicUtility::getExtension(inner_content[j]);
				if ((ext == ".pgm" || ext == ".png") && inner_content[j].find("grey") == std::string::npos) {
					pop::Mat2UI8 m;
					m.load(filename);
#ifdef BATCH_LOADING
					mat_tmp.push_back(m);
#else
					pop::VecF32 vin = pop::NNLayerMatrix::inputMatrixToInputNeuron(m, domain, pop::NNLayerMatrix::Mass, pop::NNLayerMatrix::ZeroToOne);
					v_neuron_in.push_back(vin);
					pop::VecF32 v_out(nb_dirs,-1);
					v_out(i)=1;
					v_neuron_out.push_back(v_out);
#endif
					nb_images++;
				}
			}

#ifdef BATCH_LOADING
			for (int j=0; j<mat_tmp.size(); j++) {
				pop::VecF32 vin = pop::NNLayerMatrix::inputMatrixToInputNeuron(mat_tmp(j), domain, pop::NNLayerMatrix::Mass, pop::NNLayerMatrix::ZeroToOne);
				v_neuron_in.push_back(vin);
				pop::VecF32 v_out(nb_dirs,-1);
				v_out(i)=1;
				v_neuron_out.push_back(v_out);
			}

			mat_tmp.clear();
#endif
		}
	}

	std::cout << std::endl;
}

#if defined(HAVE_CUDA)
void GPUNeuralNetwork::copyNetworkToGPU() {
	// * in-memory representation on the gpu *
	// We allocate a big continuous array that will contain all the structures + values
	// [struct neural_network | struct layer 1 | struct layer 2 | ... | struct layer n | *_X | *_Y | *_W | *_d_E_X | *_d_E_Y | *_d_E_W |	 		  ...			 ]
	//																				   [  				for layer 1					   ][ for layer 2 ] [ for others ]

	unsigned int size = sizeof(*h_network) + h_network->_nb_layers * sizeof(h_network->_layers[0]);
	for (unsigned int i=0; i<h_network->_nb_layers; i++) {
		struct layer& layer = h_network->_layers[i];
		size += (layer._X_size + layer._Y_size) * 2 * sizeof(layer._X[0]);
		if (i!=0) {
			size += (layer._W_height * layer._W_width) * 2 * sizeof(layer._W[0]);
		}
	}
	cudaMalloc(&d_network, size);

	struct layer* p_layers =  h_network->_layers;
	h_network->_layers = (struct layer*)(d_network+1);
	cudaMemcpy(d_network, h_network, sizeof(*h_network), cudaMemcpyHostToDevice);
	h_network->_layers = p_layers;

	p_layers = (struct layer*)(d_network+1);
	pop::F32* start = (pop::F32*)((char*)d_network + sizeof(*d_network) + h_network->_nb_layers * sizeof(*p_layers));
	for (unsigned int i=0; i<h_network->_nb_layers; i++) {
		struct layer& layer = h_network->_layers[i];

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
}

void GPUNeuralNetwork::copyNetworkFromGPU() {
	cudaMemcpy(h_network, d_network, sizeof(*h_network), cudaMemcpyDeviceToHost);

	struct layer* p_layers =  h_network->_layers;
	h_network->_layers = new struct layer[h_network->_nb_layers];

	pop::F32* start = (pop::F32*)((char*)d_network + sizeof(*d_network) + h_network->_nb_layers * sizeof(*p_layers));
	for (unsigned int i=0; i<h_network->_nb_layers; i++) {
		struct layer& layer = h_network->_layers[i];

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

		// We do not need to copy the errors vectors (i.e., d_E_*), as they will be initialized during the propagateBack algorithm
		layer._d_E_X = NULL;
		layer._d_E_Y = NULL;
		layer._d_E_W = NULL;

		start += layer._X_size + layer._Y_size + layer._W_height*layer._W_width;
	}
}

void GPUNeuralNetwork::deleteNetworkOnGPU() {
	if (d_network != NULL) {
		cudaFree(d_network);
	}
}

// copy n elements from position min in h_data to the gpu
// if shuffle is not empty, then its value will be used to copy elements at random positions instead of contiguous ones
pop::F32* GPUNeuralNetwork::gpu_copyDataToGPU(pop::Vec<pop::VecF32> h_data, const unsigned int min, unsigned int n, std::vector<int> shuffle) {
	if (h_data.size() == 0 || h_data(0).size() == 0) {
		std::cerr << "GPUNeuralNetwork::gpu_copyDataToGPU(): h_data is empty" << std::endl;
		return NULL;
	}

	if (min + n > h_data.size()) {
		std::cerr << "GPUNeuralNetwork::gpu_copyDataToGPU(): n is too big" << std::endl;
		return NULL;
	}

	pop::F32* d_data;
	cudaMalloc(&d_data, n * h_data(0).size() * sizeof(h_data(0)(0)));
	pop::F32* start = d_data;
	for (int i=min; i<min+n; i++) {
		int pos = (shuffle.size()>0 ? shuffle[i] : i);
		cudaMemcpy(start, &h_data(pos)(0), h_data(pos).size() * sizeof(*d_data), cudaMemcpyHostToDevice);
		start += h_data(pos).size();
	}

	return d_data;
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
		printf("\n--Layer %d, _X_size = %d, _Y_size = %d, _W_height = %d, _W_width = %d\n", l, layer._X_size, layer._Y_size, layer._W_height, layer._W_width);

		printVectorOnGPU(layer._X, layer._X_size, (char*)"_X");
		printVectorOnGPU(layer._Y, layer._Y_size, (char*)"_Y");
		printMatrixOnGPU(layer._W, layer._W_height, layer._W_width, (char*)"_W");
		printVectorOnGPU(layer._d_E_X, layer._X_size, (char*)"_d_E_X");
		printVectorOnGPU(layer._d_E_Y, layer._Y_size, (char*)"_d_E_Y");
		printMatrixOnGPU(layer._d_E_W, layer._W_height, layer._W_width, (char*)"_d_E_W");
	}
}

void GPUNeuralNetwork::gpu_displayNetwork() {
	printNetworkOnGPU<<<1, 1>>>(d_network);
	cudaDeviceSynchronize();
}

__global__ void gpu_propagateFront_setInput(struct neural_network *network, pop::F32* in_set, unsigned int in_elt_size, unsigned int idx) {
	int tid = blockDim.x*blockIdx.x + threadIdx.x;
	if (tid < in_elt_size) {
		network->_layers[0]._X[tid] = in_set[idx*in_elt_size+tid];
	}
}

__global__ void gpu_propagateFront_computeSigmoid(struct neural_network *network, int l) {
	int tid = blockDim.x*blockIdx.x + threadIdx.x;
	if (tid < network->_layers[l]._Y_size) {
		network->_layers[l]._X[tid] = 1.7159f*tanhf(0.66666667f*network->_layers[l]._Y[tid]);
	}
}

__global__ void gpu_propagateFront_setOutput(struct neural_network *network, pop::F32* out_computed) {
	int tid = blockDim.x*blockIdx.x + threadIdx.x;
	if (tid < network->_layers[network->_nb_layers-1]._X_size) {
		out_computed[tid] = network->_layers[network->_nb_layers-1]._X[tid];
	}
}

/*
 * Propagate in_set[idx] in out_computed (they must reside in GPU memory)
 * in_set: set of all the inputs. Each element's size is in_elt_size
 * out_computed: the output element, of size equal to the number of neurons in the last layer
 */
void GPUNeuralNetwork::gpu_propagateFront(pop::F32* in_set, unsigned int in_elt_size, unsigned int idx, pop::F32* out_computed) {
	int block, grid;
	unsigned int max_nb_threads = popcuda::getMaxNumberThreadsPerBlock();

	block = (h_network->_layers[0]._X_size < max_nb_threads ? h_network->_layers[0]._X_size : max_nb_threads);
	grid = h_network->_layers[0]._X_size / max_nb_threads + (h_network->_layers[0]._X_size%max_nb_threads ? 1 : 0);
	gpu_propagateFront_setInput<<<grid, block>>>(d_network, in_set, in_elt_size, idx);

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
			std::cout << "Cublas error in _Y[l+1] = _W[l+1] * _X[l] for layer l = " << l << ", cublas status: " << popcuda::cublasGetErrorString(stat) << std::endl;
		}

		// _X[l+1] = sigmoid(_Y[l+1])
		block = (h_network->_layers[l+1]._X_size < max_nb_threads ? h_network->_layers[l+1]._X_size : max_nb_threads);
		grid = h_network->_layers[l+1]._X_size / max_nb_threads + (h_network->_layers[l+1]._X_size%max_nb_threads ? 1 : 0);
		gpu_propagateFront_computeSigmoid<<<grid, block>>>(d_network, l+1);
	}

	cublasDestroy(handle);

	block = (h_network->_layers[h_network->_nb_layers-1]._X_size < max_nb_threads ? h_network->_layers[h_network->_nb_layers-1]._X_size : max_nb_threads);
	grid = h_network->_layers[h_network->_nb_layers-1]._X_size / max_nb_threads + (h_network->_layers[h_network->_nb_layers-1]._X_size%max_nb_threads ? 1 : 0);
	gpu_propagateFront_setOutput<<<grid, block>>>(d_network, out_computed);
}

__global__ void gpu_propagateBackFirstDerivate_setXError(struct neural_network *network, pop::F32* desired_output, unsigned int in_elt_size, unsigned int idx, int l) {
	int tid = blockDim.x*blockIdx.x + threadIdx.x;
	if (tid < network->_layers[l]._X_size) {
		network->_layers[l]._d_E_X[tid] = network->_layers[l]._X[tid] - desired_output[idx*in_elt_size+tid];
	}
}

__global__ void gpu_propagateBackFirstDerivate_setYError(struct neural_network *network, int l) {
	int tid = blockDim.x*blockIdx.x + threadIdx.x;
	if (tid < network->_layers[l]._X_size) {
		float S = network->_layers[l]._X[tid];
		network->_layers[l]._d_E_Y[tid] = network->_layers[l]._d_E_X[tid] * (0.666667f/1.7159f*(1.7159f*1.7159f-S*S));
	}
}

__global__ void gpu_propagateBackFirstDerivate_setWeight(struct neural_network *network, int l) {
	int tid = blockDim.x*blockIdx.x + threadIdx.x;
	struct layer& layer = network->_layers[l];
	if (tid < layer._W_height*layer._W_width) {
		int i = tid / layer._W_width;
		int j = tid % layer._W_width;

		layer._d_E_W[tid] = layer._d_E_Y[i] * network->_layers[l-1]._X[j];
#ifndef SET_W_ERROR_CUBLAS
		layer._W[tid] = layer._W[tid] - network->_eta * layer._d_E_Y[i] * network->_layers[l-1]._X[j];
#endif
	}
}

__global__ void gpu_propagateBackFirstDerivate_setPreviousXError(struct neural_network *network, int l) {
	int tid = blockDim.x*blockIdx.x + threadIdx.x;
	if (tid < network->_layers[l-1]._X_size) {
		struct layer& layer = network->_layers[l];
		struct layer& prev_layer = network->_layers[l-1];
		pop::F32 s = 0.0f;

		for (unsigned int i=0; i<layer._W_height; i++) {
			s += layer._W[i*layer._W_width+tid] * layer._d_E_Y[i];
		}
		prev_layer._d_E_X[tid] = s;
	}
}

/*
 * Propagate back diff(out_set[idx], out_computed) using the network on the GPU
 * out_set: set of all the inputs. Each element's size is out_elt_size
 * out_computed: the output element computed previously (using propagateFrontGPU), of size out_elt_size
 */
void GPUNeuralNetwork::gpu_propagateBackFirstDerivate(pop::F32* out_set, unsigned int out_elt_size, unsigned int idx) {
	int block, grid;
	unsigned int max_nb_threads = popcuda::getMaxNumberThreadsPerBlock();

#ifdef SET_W_ERROR_CUBLAS
	cublasStatus_t	stat;
	cublasHandle_t handle;
	cublasCreate(&handle);
	float alpha = 1.0f;
	float beta = -h_network->_eta;
#endif

	// start points to the start of the layer nb_layers-1
	pop::F32* start = (pop::F32*)((char*)d_network + sizeof(*d_network) + h_network->_nb_layers * sizeof(h_network->_layers[0]));
	for (unsigned int l=0; l<h_network->_nb_layers-1; l++) {
		struct layer& layer = h_network->_layers[l];
		start += (layer._X_size + layer._Y_size + layer._W_height*layer._W_width)*2;
	}

	for (unsigned int l=h_network->_nb_layers-1; l>0; l--) {
		struct layer& layer = h_network->_layers[l];
		struct layer& prev_layer = h_network->_layers[l-1];

		// _d_E_X[l] = _X[l] - desired_output
		if (l == h_network->_nb_layers-1){
			block = (layer._X_size < max_nb_threads ? layer._X_size : max_nb_threads);
			grid = layer._X_size / max_nb_threads + (layer._X_size%max_nb_threads ? 1 : 0);
			gpu_propagateBackFirstDerivate_setXError<<<grid, block>>>(d_network, out_set, out_elt_size, idx, l);
		}

		// _d_E_Y[l] = _d_E_X[l] * derived_sigmoid(_X[l])
		block = (layer._Y_size < max_nb_threads ? layer._Y_size : max_nb_threads);
		grid = layer._Y_size / max_nb_threads + (layer._Y_size%max_nb_threads ? 1 : 0);
		gpu_propagateBackFirstDerivate_setYError<<<grid, block>>>(d_network, l);

		// _d_E_W[l-1] = _d_E_Y[l] * _X[l-1]
		unsigned int nb_weights = layer._W_height * layer._W_width;
		block = (nb_weights < max_nb_threads ? nb_weights : max_nb_threads);
		grid = nb_weights / max_nb_threads + (nb_weights%max_nb_threads ? 1 : 0);
		gpu_propagateBackFirstDerivate_setWeight<<<grid, block>>>(d_network, l);

		// _W[l-1] = _W[l-1] - _eta * _d_E_W[l-1]
#ifdef SET_W_ERROR_CUBLAS
		pop::F32* d_W = start + layer._X_size + layer._Y_size;
		pop::F32* d_dW = d_W + layer._W_height*layer._W_width + layer._X_size + layer._Y_size;
		stat = cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, layer._W_width, layer._W_height, &alpha, d_W, layer._W_width, &beta, d_dW, layer._W_width, d_W, layer._W_width);
		if (stat != CUBLAS_STATUS_SUCCESS) {
			std::cout << "Cublas error in _W[l-1] = _W[l-1] - _eta * _d_E_W[l-1] for layer l = " << l << ", cublas status: " << popcuda::cublasGetErrorString(stat) << std::endl;
		}

		if (l>1) {
			start -= (prev_layer._X_size + prev_layer._Y_size + prev_layer._W_height*prev_layer._W_width)*2;
		}
#else
		// this is done in the previous kernel (gpu_propagateBackFirstDerivate_setWeight)
#endif

		// _d_E_X[l-1][j] = sum_{i=0}^{_W[l-1].sizeI()}{_W[l](i, j) * _d_E_Y[l](i)}, j=0 to _X[l].size()
		block = (prev_layer._X_size < max_nb_threads ? prev_layer._X_size : max_nb_threads);
		grid = prev_layer._X_size / max_nb_threads + (prev_layer._X_size%max_nb_threads ? 1 : 0);
		gpu_propagateBackFirstDerivate_setPreviousXError<<<grid, block>>>(d_network, l);
	}

#ifdef SET_W_ERROR_CUBLAS
	cublasDestroy(handle);
#endif
}

//This version is very simple. We can leverage the parallelism of the GPU and do something better
__global__ void gpu_computeError_v1(pop::F32* desired_output, pop::F32* computed_output, unsigned int out_elt_size, unsigned int idx, int* error) {
	int max1 = 0;
	int max2 = 0;
	for (int i=1; i<out_elt_size; i++) {
		if (desired_output[idx*out_elt_size+i] > desired_output[idx*out_elt_size+max1]) {
			max1 = i;
		}
		if (computed_output[i] > computed_output[max2]) {
			max2 = i;
		}
	}

	if (max1 != max2) {
		(*error)++;
	}
}

void GPUNeuralNetwork::gpu_computeError(pop::F32* out_set, pop::F32* out_computed, unsigned int out_elt_size, unsigned int idx, int* error) {
	gpu_computeError_v1<<<1, 1>>>(out_set, out_computed, out_elt_size, idx, error);
}

void GPUNeuralNetwork::gpu_learn(pop::Vec<pop::VecF32>& vtraining_in, pop::Vec<pop::VecF32>& vtraining_out, pop::Vec<pop::VecF32>& vtest_in, pop::Vec<pop::VecF32>& vtest_out, bool final_cpu_test, const int nb_epoch) {
	size_t total_size_training = (vtraining_in.size()*vtraining_in(0).size() + vtraining_out.size()*vtraining_out(0).size()) * sizeof(vtraining_in(0)(0));
	size_t total_size_test = (vtest_in.size()*vtest_in(0).size() + vtest_out.size()*vtest_out(0).size()) * sizeof(vtest_in(0)(0));
	std::cout << "total training size: " << total_size_training << ", total size test: " << total_size_test << std::endl;

	size_t free, total;
	cudaMemGetInfo(&free, &total);
	const size_t available = GPU_MEMORY_PRESSURE*free;

	pop::F32* d_out;
	cudaMalloc(&d_out, vtest_out(0).size() * sizeof(vtest_out(0)(0)));

	int error_training, error_test;
	int *d_error_training, *d_error_test;
	cudaMalloc(&d_error_training, sizeof(error_training));
	cudaMalloc(&d_error_test, sizeof(error_test));

	std::vector<int> v_global_rand(vtraining_in.size());
	for(unsigned int i=0;i<v_global_rand.size();i++)
		v_global_rand[i]=i;

	std::cout<<"iter_epoch\t error_train\t error_test\t learning rate"<<std::endl;

	for (unsigned int i=0;i<nb_epoch;i++) {
		int start, stop, step;
		error_training = error_test = 0;
		cudaMemcpy(d_error_training, &error_training, sizeof(error_training), cudaMemcpyHostToDevice);
		cudaMemcpy(d_error_test, &error_test, sizeof(error_test), cudaMemcpyHostToDevice);
		std::random_shuffle(v_global_rand.begin(), v_global_rand.end(), pop::Distribution::irand());

		//*********************** TRAINING ***********************
		start = 0;
		stop = vtraining_in.size();
		step = available / (vtraining_in(0).size()*sizeof(vtraining_in(0)(0)) + vtraining_out(0).size()*sizeof(vtraining_out(0)(0)));
		while (start < stop) {
			const int nb_elts = min(step, stop-start);
			//std::cout << "training: start=" << start << ", stop=" << stop << ", step=" << step << ", nb_elts=" << nb_elts << std::endl;
			pop::F32* d_vtraining_in = GPUNeuralNetwork::gpu_copyDataToGPU(vtraining_in, start, nb_elts, v_global_rand);
			pop::F32* d_vtraining_out = GPUNeuralNetwork::gpu_copyDataToGPU(vtraining_out, start, nb_elts, v_global_rand);

			for(unsigned int j=0;j<nb_elts;j++) {
				gpu_propagateFront(d_vtraining_in, vtraining_in(0).size(), j, d_out);
				gpu_propagateBackFirstDerivate(d_vtraining_out, vtraining_out(0).size(), j);
				gpu_computeError(d_vtraining_out, d_out, vtraining_out(0).size(), j, d_error_training);
			}

			cudaFree(d_vtraining_in);
			cudaFree(d_vtraining_out);
			start += step;
		}

		//*********************** TEST ***********************
		start = 0;
		stop = vtest_in.size();
		step = available / (vtest_in(0).size()*sizeof(vtest_in(0)(0)) + vtest_out(0).size()*sizeof(vtest_out(0)(0)));
		while (start < stop) {
			//std::cout << "test: start=" << start << ", stop=" << stop << ", step=" << step << std::endl;
			const int nb_elts = min(start+step, stop);
			pop::F32* d_vtest_in = GPUNeuralNetwork::gpu_copyDataToGPU(vtest_in, start, nb_elts);
			pop::F32* d_vtest_out = GPUNeuralNetwork::gpu_copyDataToGPU(vtest_out, start, nb_elts);

			for(unsigned int j=0;j<nb_elts;j++){
				gpu_propagateFront(d_vtest_in, vtest_in(0).size(), j, d_out);
				gpu_computeError(d_vtest_out, d_out, vtest_out(0).size(), j, d_error_test);
			}

			cudaFree(d_vtest_in);
			cudaFree(d_vtest_out);
			start += step;
		}

		setEta(getEta()*0.9);

		cudaMemcpy(&error_training, d_error_training, sizeof(error_training), cudaMemcpyDeviceToHost);
		cudaMemcpy(&error_test, d_error_test, sizeof(error_test), cudaMemcpyDeviceToHost);

		std::cout<<i<<"\t"<<error_training*1./vtraining_in.size()<<"\t"<<error_test*1./vtest_in.size() <<"\t"<<getEta()<<"\t"<<getCurrentTime()<<std::endl;

		//If you want to save the network:
		//copyNetworkFromGPU();
		//save("nnet_" + getCurrentTime() + ".bin");
	}

	cudaFree(d_error_training);
	cudaFree(d_error_test);
	cudaFree(d_out);

	copyNetworkFromGPU();

	if (final_cpu_test) {
		//FINAL TEST WITH THE CPU
		error_test = 0;
		for(unsigned int j=0;j<vtest_in.size();j++){
			pop::VecF32 vout;
			propagateFront(vtest_in(j),vout);
			int label1 = std::distance(vout.begin(),std::max_element(vout.begin(),vout.end()));
			int label2 = std::distance(vtest_out(j).begin(),std::max_element(vtest_out(j).begin(),vtest_out(j).end()));
			if(label1!=label2){
				error_test++;
			}
		}

		std::cout<<"FINAL-CPU\t"<< " - " << "\t"<<error_test*1./vtest_in.size() <<"\t"<<getEta()<<"\t"<<getCurrentTime()<<std::endl;
	}
}

void GPUNeuralNetwork::gpu_propagate(pop::Vec<pop::VecF32>& vtraining_in, pop::Vec<pop::VecF32>& vtest_in, const int nb_epoch) {
	size_t total_size_training = vtraining_in.size() * vtraining_in(0).size() * sizeof(vtraining_in(0)(0));
	size_t total_size_test = vtest_in.size() * vtest_in(0).size() * sizeof(vtest_in(0)(0));
	std::cout << "total training size: " << total_size_training << ", total size test: " << total_size_test << std::endl;

	size_t free, total;
	cudaMemGetInfo(&free, &total);
	const size_t available = GPU_MEMORY_PRESSURE*free;

	pop::F32* d_out;
	cudaMalloc(&d_out, h_network->_layers[h_network->_nb_layers-1]._X_size);

	int error_training, error_test;
	int *d_error_training, *d_error_test;
	cudaMalloc(&d_error_training, sizeof(error_training));
	cudaMalloc(&d_error_test, sizeof(error_test));

	std::vector<int> v_global_rand(vtraining_in.size());
	for(unsigned int i=0;i<v_global_rand.size();i++)
		v_global_rand[i]=i;

	std::cout << "epoch\ttotal_samples\ttotal_duration(ns)\tduration_per_sample(ns)" << std::endl;

	for (unsigned int i=0;i<nb_epoch;i++) {
		struct timespec time_start, time_end;
		int start, stop, step;

		clock_gettime(CLOCK_REALTIME, &time_start);

		//*********************** TRAINING ***********************
		start = 0;
		stop = vtraining_in.size();
		step = available / (vtraining_in(0).size()*sizeof(vtraining_in(0)(0)));
		while (start < stop) {
			const int nb_elts = min(step, stop-start);
			pop::F32* d_vtraining_in = GPUNeuralNetwork::gpu_copyDataToGPU(vtraining_in, start, nb_elts);

			for(unsigned int j=0;j<nb_elts;j++) {
				gpu_propagateFront(d_vtraining_in, vtraining_in(0).size(), j, d_out);
			}

			cudaFree(d_vtraining_in);
			start += step;
		}

		//*********************** TEST ***********************
		start = 0;
		stop = vtest_in.size();
		step = available / (vtest_in(0).size()*sizeof(vtest_in(0)(0)));
		while (start < stop) {
			const int nb_elts = min(start+step, stop);
			pop::F32* d_vtest_in = GPUNeuralNetwork::gpu_copyDataToGPU(vtest_in, start, nb_elts);

			for(unsigned int j=0;j<nb_elts;j++){
				gpu_propagateFront(d_vtest_in, vtest_in(0).size(), j, d_out);
			}

			cudaFree(d_vtest_in);
			start += step;
		}

		clock_gettime(CLOCK_REALTIME, &time_end);
		unsigned long duration = (time_end.tv_sec-time_start.tv_sec)*1000000000 + (time_end.tv_nsec-time_start.tv_nsec);
		std::cout << i << "\t" << vtraining_in.size()+vtest_in.size() << duration << "\t" << duration/(vtraining_in.size()+vtest_in.size()) << std::endl;
	}

	cudaFree(d_out);
}
#endif

void test_neural_net_cpu(const int nb_epoch) {
	std::vector<unsigned int> v_layer;
	v_layer.push_back(2);
	v_layer.push_back(3);
	v_layer.push_back(1);
	GPUNeuralNetwork network(v_layer, 0.01);

	std::cout << "\n********** CPU **********\n" << std::endl;

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

	for(unsigned int i=0;i<nb_epoch;i++){
		std::random_shuffle ( v_global_rand.begin(), v_global_rand.end() , pop::Distribution::irand());
		int error=0;
		for(unsigned int j=0;j<v_global_rand.size();j++){
			pop::VecF32 vout;
			network.propagateFront(v_in(v_global_rand[j]),vout);
			network.propagateBackFirstDerivate(v_out(v_global_rand[j]));

			int label1 = std::distance(vout.begin(),std::max_element(vout.begin(),vout.end()));
			int label2 = std::distance(v_out(v_global_rand[j]).begin(),std::max_element(v_out(v_global_rand[j]).begin(),v_out(v_global_rand[j]).end()));
			if(label1!=label2)
				error++;
		}

		//std::cout<<i<<"\t"<<error*1.0/v_global_rand.size()<<std::endl;
	}

	//test the training
	for(int j=0;j<4;j++){
		pop::VecF32 vout;
		network.propagateFront(v_in(j), vout);
		std::cout<<vout<<std::endl;// we obtain the expected value -1 , 1 , 1 , -1
	}
	std::cout<<std::endl;
}

void test_neural_net_cpu_mnist(const int nb_epoch) {
	pop::Vec<pop::Vec<pop::Mat2UI8> > number_training =  pop::TrainingNeuralNetwork::loadMNIST("/media/pl/shared/PL/neural_nets_samples/MNIST/train-images-idx3-ubyte","/media/pl/shared/PL/neural_nets_samples/MNIST/train-labels-idx1-ubyte");
	pop::Vec<pop::Vec<pop::Mat2UI8> > number_test =  pop::TrainingNeuralNetwork::loadMNIST("/media/pl/shared/PL/neural_nets_samples/MNIST/t10k-images-idx3-ubyte","/media/pl/shared/PL/neural_nets_samples/MNIST/t10k-labels-idx1-ubyte");

	double size_in= number_training(0)(0).getDomain()(0) * number_training(0)(0).getDomain()(1);
	std::cout << "size trainings: " << number_training(0).size() << std::endl;

	std::vector<unsigned int> v_layer;
	v_layer.push_back(size_in);
	v_layer.push_back(1000);
	v_layer.push_back(1000);
	v_layer.push_back(number_training.size());
	GPUNeuralNetwork network(v_layer, 0.001);

	pop::Vec<pop::VecF32> vtraining_in;
	pop::Vec<pop::VecF32> vtraining_out;

	pop::TrainingNeuralNetwork::convertMatrixToInputValueNeuron(vtraining_in,vtraining_out,number_training,number_training(0)(0).getDomain(),pop::NNLayerMatrix::Mass,pop::NNLayerMatrix::MinusOneToOne);

	pop::Vec<pop::VecF32> vtest_in;
	pop::Vec<pop::VecF32> vtest_out;
	pop::TrainingNeuralNetwork::convertMatrixToInputValueNeuron(vtest_in,vtest_out,number_test,number_test(0)(0).getDomain(),pop::NNLayerMatrix::Mass,pop::NNLayerMatrix::MinusOneToOne);

	number_training.clear();
	number_test.clear();


	size_t total_size_training = (vtraining_in.size()*vtraining_in(0).size() + vtraining_out.size()*vtraining_out(0).size()) * sizeof(vtraining_in(0)(0));
	size_t total_size_test = (vtest_in.size()*vtest_in(0).size() + vtest_out.size()*vtest_out(0).size()) * sizeof(vtest_in(0)(0));
	std::cout << "total training size: " << total_size_training << ", total size test: " << total_size_test << std::endl;

	std::vector<int> v_global_rand(vtraining_in.size());
	for(unsigned int i=0;i<v_global_rand.size();i++)
		v_global_rand[i]=i;

	std::cout<<"iter_epoch\t error_train\t error_test\t learning rate"<<std::endl;

	for (unsigned int i=0;i<nb_epoch;i++) {
		std::random_shuffle ( v_global_rand.begin(), v_global_rand.end() ,pop::Distribution::irand());
		int error_training=0,error_test=0;

		for(unsigned int j=0;j<v_global_rand.size();j++){
			pop::VecF32 vout;
			network.propagateFront(vtraining_in(v_global_rand[j]),vout);
			int label1 = std::distance(vout.begin(),std::max_element(vout.begin(),vout.end()));
			network.propagateBackFirstDerivate(vtraining_out(v_global_rand[j]));
			int label2 = std::distance(vtraining_out(v_global_rand[j]).begin(),std::max_element(vtraining_out(v_global_rand[j]).begin(),vtraining_out(v_global_rand[j]).end()));
			if(label1!=label2){
				error_training++;
			}
		}
		for(unsigned int j=0;j<vtest_in.size();j++){
			pop::VecF32 vout;
			network.propagateFront(vtest_in(j),vout);
			int label1 = std::distance(vout.begin(),std::max_element(vout.begin(),vout.end()));
			int label2 = std::distance(vtest_out(j).begin(),std::max_element(vtest_out(j).begin(),vtest_out(j).end()));
			if(label1!=label2){
				error_test++;
			}
		}

		network.setEta(network.getEta()*0.9);
		std::cout<<i<<"\t"<<error_training*1./v_global_rand.size()<<"\t"<<error_test*1./vtest_in.size() <<"\t"<<network.getEta()<<"\t"<<getCurrentTime()<<std::endl;
	}
}

#if defined(HAVE_CUDA)
void test_neural_net_gpu(const int nb_epoch) {
	std::vector<unsigned int> v_layer;
	v_layer.push_back(2);
	v_layer.push_back(3);
	v_layer.push_back(1);
	GPUNeuralNetwork network(v_layer, 0.01);

	std::cout << "\n********** GPU **********\n" << std::endl;

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
		return;
	}

	//use the backpropagation algorithm with first order method
	std::vector<int> v_global_rand(v_in.size());
	for(unsigned int i=0;i<v_global_rand.size();i++)
		v_global_rand[i]=i;

	pop::F32* d_in_set = GPUNeuralNetwork::gpu_copyDataToGPU(v_in, 0, v_in.size());
	pop::F32* d_out_set = GPUNeuralNetwork::gpu_copyDataToGPU(v_out, 0, v_out.size());

	pop::F32* d_out;
	cudaMalloc(&d_out, v_out(0).size() * sizeof(v_in(0)(0)));

	int error;
	int* d_error;
	cudaMalloc(&d_error, sizeof(error));

	std::cout<<"iter_epoch\t error_train"<<std::endl;
	for(unsigned int i=0;i<nb_epoch;i++){
		std::random_shuffle ( v_global_rand.begin(), v_global_rand.end() , pop::Distribution::irand());

		error = 0;
		cudaMemcpy(d_error, &error, sizeof(error), cudaMemcpyHostToDevice);

		for(unsigned int j=0;j<v_global_rand.size();j++){
			network.gpu_propagateFront(d_in_set, v_in(0).size(), v_global_rand[j], d_out);
			network.gpu_propagateBackFirstDerivate(d_out_set, v_out(0).size(), v_global_rand[j]);
			network.gpu_computeError(d_out_set, d_out, v_out(0).size(), v_global_rand[j], d_error);
		}

		//cudaMemcpy(&error, d_error, sizeof(error), cudaMemcpyDeviceToHost);
		//std::cout<<i<<"\t"<<error*1.0/v_global_rand.size()<<std::endl;
	}

	cudaFree(d_error);
	cudaFree(d_out);
	cudaFree(d_in_set);
	cudaFree(d_out_set);

	network.copyNetworkFromGPU();

	//test the training
	for(int j=0;j<4;j++){
		pop::VecF32 vout;
		network.propagateFront(v_in(j), vout);
		std::cout<<vout<<std::endl;// we obtain the expected value -1 , 1 , 1 , -1
	}
	std::cout<<std::endl;
}

void test_neural_net_gpu_mnist(const int nb_epoch) {
	pop::Vec<pop::Vec<pop::Mat2UI8> > number_training =  pop::TrainingNeuralNetwork::loadMNIST("/media/pl/shared/PL/neural_nets_samples/MNIST/train-images-idx3-ubyte","/media/pl/shared/PL/neural_nets_samples/MNIST/train-labels-idx1-ubyte");
	pop::Vec<pop::Vec<pop::Mat2UI8> > number_test =  pop::TrainingNeuralNetwork::loadMNIST("/media/pl/shared/PL/neural_nets_samples/MNIST/t10k-images-idx3-ubyte","/media/pl/shared/PL/neural_nets_samples/MNIST/t10k-labels-idx1-ubyte");

	double size_in= number_training(0)(0).getDomain()(0) * number_training(0)(0).getDomain()(1);
	std::cout << "size trainings: " << number_training(0).size() << std::endl;

	std::vector<unsigned int> v_layer;
	v_layer.push_back(size_in);
	v_layer.push_back(1000);
	v_layer.push_back(1000);
	v_layer.push_back(number_training.size());
	GPUNeuralNetwork network(v_layer, 0.001);

	pop::Vec<pop::VecF32> vtraining_in;
	pop::Vec<pop::VecF32> vtraining_out;
	pop::TrainingNeuralNetwork::convertMatrixToInputValueNeuron(vtraining_in,vtraining_out,number_training,number_training(0)(0).getDomain(),pop::NNLayerMatrix::Mass,pop::NNLayerMatrix::MinusOneToOne);

	pop::Vec<pop::VecF32> vtest_in;
	pop::Vec<pop::VecF32> vtest_out;
	pop::TrainingNeuralNetwork::convertMatrixToInputValueNeuron(vtest_in,vtest_out,number_test,number_test(0)(0).getDomain(),pop::NNLayerMatrix::Mass,pop::NNLayerMatrix::MinusOneToOne);

	number_training.clear();
	number_test.clear();

	network.gpu_learn(vtraining_in, vtraining_out, vtest_in, vtest_out, false, nb_epoch);
}

void test_neural_net_gpu_augmented_database(const int max_files_per_folder, const int network_for_training, std::string database_training, std::string database_test, const int nb_epoch) {
	std::cout << "Starting to load training database at " << getCurrentTime() << std::endl;
	pop::Vec<pop::VecF32> vtraining_in;
	pop::Vec<pop::VecF32> vtraining_out;
	loadDatabase(database_training, max_files_per_folder, vtraining_in, vtraining_out);
	std::cout << "Training database loaded at " << getCurrentTime() << std::endl;

	std::cout << "Starting to load test database at " << getCurrentTime() << std::endl;
	pop::Vec<pop::VecF32> vtest_in;
	pop::Vec<pop::VecF32> vtest_out;
	loadDatabase(database_test, max_files_per_folder, vtest_in, vtest_out);
	std::cout << "Test database loaded at " << getCurrentTime() << std::endl;

	int size_in = vtraining_in(0).size();
	int size_out = vtraining_out(0).size();

	std::vector<unsigned int> v_layer;
	v_layer.push_back(size_in);
	switch (network_for_training) {
	case 5:
		for (int i=0; i<9; i++) {
			v_layer.push_back(1000);
		}
		break;
	case 4:
		v_layer.push_back(2500);
	case 3:
		v_layer.push_back(2000);
	case 2:
		v_layer.push_back(1500);
	case 1:
		v_layer.push_back(1000);
		v_layer.push_back(500);
		break;
	default:
		std::cerr << "Database training: unknown network " << network_for_training << std::endl;
		return;
	}
	v_layer.push_back(size_out);
	GPUNeuralNetwork network(v_layer, 0.001);
	std::cout << "Network created at " << getCurrentTime() << std::endl;

	network.gpu_learn(vtraining_in, vtraining_out, vtest_in, vtest_out, true, nb_epoch);
}

void bench_propagate_front_gpu_augmented_database(const int max_files_per_folder, std::string network_path, std::string database_training, std::string database_test, const int nb_epoch) {
	std::cout << "Starting to load training database at " << getCurrentTime() << std::endl;
	pop::Vec<pop::VecF32> vtraining_in;
	pop::Vec<pop::VecF32> vtraining_out;
	loadDatabase(database_training, max_files_per_folder, vtraining_in, vtraining_out);
	vtraining_out.clear();
	std::cout << "Training database loaded at " << getCurrentTime() << std::endl;

	std::cout << "Starting to load test database at " << getCurrentTime() << std::endl;
	pop::Vec<pop::VecF32> vtest_in;
	pop::Vec<pop::VecF32> vtest_out;
	loadDatabase(database_test, max_files_per_folder, vtest_in, vtest_out);
	vtest_out.clear();
	std::cout << "Test database loaded at " << getCurrentTime() << std::endl;

	GPUNeuralNetwork network;
	network.load(network_path);
	network.copyNetworkToGPU();
	std::cout << "Network created at " << getCurrentTime() << std::endl;

	network.gpu_propagate(vtraining_in, vtest_in, nb_epoch);
}
#endif
