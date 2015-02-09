#include "popconfig.h"

#if defined(HAVE_CUDA)

#include <iostream>

#include "c_neural_net.h"
#include "Population.h"
#include "microtime.h"

struct neural_network* createNetwork(std::vector<unsigned int> v_layer, double eta) {
	struct neural_network* network = new struct neural_network;

	network->nb_layers = v_layer.size();
	network->layers = new struct layer[network->nb_layers];
	network->_eta = eta;

	for(unsigned int i=0;i<v_layer.size();i++){
		int size_layer = v_layer[i];
		struct layer& l = network->layers[i];

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
			unsigned int size_layer_previous = network->layers[i-1]._X_size;
			pop::DistributionNormal n(0,1./std::sqrt(size_layer_previous));

			l._W_height = size_layer;
			l._W_width = size_layer_previous;
			l._W = new pop::F32*[l._W_height];
			for (unsigned int j=0; j<l._W_height; j++) {
				l._W[j] = new pop::F32[l._W_width];
				for (unsigned int k=0; k<l._W_width; k++) {
					l._W[j][k] = n.randomVariable();
				}
			}
			l._d_E_W = NULL;
		} else {
			l._W = NULL;
			l._d_E_W = NULL;
		}
	}

	return network;
}

void printNeuronsVector(pop::F32* V, unsigned int size, std::string label) {
	if (V == NULL) {
		std::cout << label << " = NULL" << std::endl;
	} else {
		std::cout << label << "(" << size << ") = [";
		for (int i=0; i<size; i++) {
			std::cout << "\t" << V[i];
		}
		std::cout << "\t]" << std::endl;
	}
}

void printWeightMatrix(pop::F32** M, unsigned int height, unsigned int width, std::string label) {
	if (M == NULL) {
		std::cout << label << " = NULL" << std::endl;
	} else {
		std::cout << label << "(" << height << ", " << width << ") = [" << std::endl;
		for (int i=0; i<height; i++) {
			for (int j=0; j<width; j++) {
				std::cout << "\t" << M[i][j];
			}
			std::cout << std::endl;
		}
		std::cout << "]" << std::endl;
	}
}

void printNetwork(struct neural_network* network) {
	std::cout << "####################" << std::endl;
	std::cout << "Number of layers: " << network->nb_layers << std::endl;
	std::cout << "Eta: " << network->_eta << std::endl;

	for (unsigned int l=0; l<network->nb_layers; l++) {
		struct layer& layer = network->layers[l];

		std::cout << "\n-- Layer " << l << ":" << std::endl;
		printNeuronsVector(layer._X, layer._X_size, "_X");
		printNeuronsVector(layer._Y, layer._Y_size, "_Y");
		printNeuronsVector(layer._d_E_X, layer._X_size, "_d_E_X");
		printNeuronsVector(layer._d_E_Y, layer._Y_size, "_d_E_Y");
		printWeightMatrix(layer._W, layer._W_height, layer._W_width, "_W");
		printWeightMatrix(layer._d_E_W, layer._W_height, layer._W_width, "_d_E_W");
	}

	std::cout << "####################" << std::endl;
}

void propagateFront(struct neural_network* network, const pop::VecF32& in , pop::VecF32 &out) {
	std::copy(in.begin(),in.end(), network->layers[0]._X);

	//TODO: send network to gpu

	//TODO: do these computations on GPU

	for (unsigned int l=0; l<network->nb_layers-1; l++) {
		struct layer& prev_layer = network->layers[l];
		struct layer& layer = network->layers[l+1];

		// _Y[l+1] = _W[l] * _X[l]
		for (unsigned int i=0; i<layer._Y_size; i++) {
			layer._Y[i] = 0;
			for (unsigned int j=0; j<prev_layer._X_size; j++) {
				layer._Y[i] += layer._W[i][j] * prev_layer._X[j];
			}
		}

		// _X[l+1] = sigmoid(_Y[l+1])
		for (unsigned int i=0; i<layer._Y_size; i++) {
			layer._X[i] = sigmoid(layer._Y[i]);
		}
	}

	//TODO: retrieve network from gpu

	struct layer& last_layer = network->layers[network->nb_layers-1];
	if (out.size() != last_layer._X_size) {
		out.resize(last_layer._X_size);
	}
	std::copy(last_layer._X, last_layer._X+last_layer._X_size,out.begin());
}

void propagateBackFirstDerivate(struct neural_network* network, const pop::VecF32& desired_output) {
	for (unsigned int l=0; l<network->nb_layers; l++) {
		struct layer& layer = network->layers[l];
		if (layer._X != NULL && layer._d_E_X == NULL) {
			layer._d_E_X = new pop::F32[layer._X_size];
			memcpy(layer._d_E_X, layer._X, sizeof(layer._X[0]) * layer._X_size);
		}
		if (layer._Y != NULL && layer._d_E_Y == NULL) {
			layer._d_E_Y = new pop::F32[layer._Y_size];
			memcpy(layer._d_E_Y, layer._Y, sizeof(layer._X[0]) * layer._Y_size);
		}
		if (layer._W != NULL && layer._d_E_W == NULL) {
			layer._d_E_W = new pop::F32*[layer._W_height];
			for (unsigned int j=0; j<layer._W_height; j++) {
				layer._d_E_W[j] = new pop::F32[layer._W_width];
				memcpy(layer._d_E_W[j], layer._W[j], sizeof(layer._W[0][0]) * layer._W_width);
			}
		}
	}

	//TODO: send network to GPU

	//TODO: perform computations on GPU

	for (unsigned int l=network->nb_layers-1; l>0; l--) {
		struct layer& layer = network->layers[l];
		struct layer& prev_layer = network->layers[l-1];

		// _d_E_X[l] = _X[l] - desired_output
		if (l == network->nb_layers-1){
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
				layer._d_E_W[i][j] = layer._d_E_Y[i] * prev_layer._X[j];
				layer._W[i][j] = layer._W[i][j] - network->_eta*layer._d_E_W[i][j];
			}
		}

		// _d_E_X[l-1][j] = sum_{i=0}^{_W[l-1].sizeI()}{_W[l](i, j) * _d_E_Y[l](i)}, j=0 to _X[l].size()
		for(unsigned int j=0; j<prev_layer._X_size; j++){
			prev_layer._d_E_X[j] = 0;
			for (unsigned int i=0; i<layer._W_height; i++) {
				prev_layer._d_E_X[j] += layer._W[i][j] * layer._d_E_Y[i];
			}
		}
	}

	//TODO: retrieve network from gpu
}

void deleteNetwork(struct neural_network* network) {
	for (int i=0; i<network->nb_layers; i++) {
		struct layer& l = network->layers[i];

		delete[] l._X;
		if (l._d_E_X != NULL) {
			delete[] l._d_E_X;
		}

		delete[] l._Y;
		if (l._d_E_Y != NULL) {
			delete[] l._d_E_Y;
		}

		if (l._W != NULL) {
			for (unsigned int j=0; j<l._W_height; j++) {
				delete[] l._W[j];
			}
			delete[] l._W;
		}

		if (l._d_E_W != NULL) {
			for (unsigned int j=0; j<l._W_height; j++) {
				delete[] l._d_E_W[j];
			}
			delete[] l._d_E_W;
		}
	}
	delete[] network->layers;
}

void test_neural_net(void) {
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

#endif
