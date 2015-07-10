#include "popconfig.h"

#include <iostream>
#include <time.h>

#include "c_neural_net.h"
#include "popcuda.h"
#include "Population.h"
#include "microtime.h"
#include "data/utility/BasicUtility.h"

// if defined, then loadDatabase() will process a batch of images instead of 1 image at a time. May improve disk access
#define BATCH_LOADING

static std::string getCurrentTime() {
	time_t rawtime;
	time(&rawtime);
	struct tm * timeinfo = localtime(&rawtime);
	std::string s = asctime(timeinfo);
	s.erase(s.length()-1);
	return s;
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

void test_neural_net_cpu(const int nb_epoch) {
	std::vector<struct layer_representation> v_layer;
	struct layer_representation lr;
	lr.type = LAYER_INPUT;
	lr.nb_neurons = 2;
	v_layer.push_back(lr);
	lr.type = LAYER_FULLY_CONNECTED;
	lr.nb_neurons = 3;
	v_layer.push_back(lr);
	lr.nb_neurons = 1;
	v_layer.push_back(lr);
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
    pop::Vec<pop::Vec<pop::Mat2UI8> > number_training =  pop::TrainingNeuralNetwork::loadMNIST("/home/olivia/workspace/MNIST/train-images-idx3-ubyte","/home/olivia/workspace/MNIST/train-labels-idx1-ubyte");
    pop::Vec<pop::Vec<pop::Mat2UI8> > number_test =  pop::TrainingNeuralNetwork::loadMNIST("/home/olivia/workspace/MNIST/t10k-images-idx3-ubyte","/home/olivia/workspace/MNIST/t10k-labels-idx1-ubyte");

	double size_in = number_training(0)(0).getDomain()(0) * number_training(0)(0).getDomain()(1);
	std::cout << "size trainings: " << number_training(0).size() << std::endl;

	std::vector<struct layer_representation> v_layer;
	struct layer_representation lr;
	lr.type = LAYER_INPUT;
	lr.nb_neurons = size_in;
	v_layer.push_back(lr);
	lr.type = LAYER_FULLY_CONNECTED;
	lr.nb_neurons = 1000;
	v_layer.push_back(lr);
	lr.nb_neurons = 1000;
	v_layer.push_back(lr);
	lr.nb_neurons = number_training.size();
	v_layer.push_back(lr);
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

void test_neural_net_conv_cpu(const int nb_epoch) {
	int size_input_matrix = 7;
	int nbr_map=1;

	std::vector<struct layer_representation> v_layer;
	struct layer_representation lr;
	lr.type = LAYER_INPUT_MATRIX;
	lr.sizei_map = size_input_matrix;
	lr.sizej_map = size_input_matrix;
	lr.nbr_map = nbr_map;
	v_layer.push_back(lr);
	lr.type = LAYER_CONVOLUTIONAL;
	lr.nbr_map = nbr_map;
	lr.radius_kernel = 1;
	lr.sub_resolution_factor = 2;
	v_layer.push_back(lr);
	lr.type = LAYER_FULLY_CONNECTED;
	lr.nb_neurons = 1;
	v_layer.push_back(lr);
	GPUNeuralNetwork network(v_layer, 0.01);

	std::cout << "\n********** CPU **********\n" << std::endl;

	pop::Mat2F32 m(size_input_matrix,size_input_matrix);
	pop::DistributionNormal d(0,1);
	for(unsigned int i=0; i<m.size(); i++) {
		m(i) = d.randomVariable();
	}

	pop::Vec<pop::Mat2F32> v_m(1);
	v_m(0) = m;
	pop::Vec<pop::F32> v_in(size_input_matrix*size_input_matrix);
	for(unsigned int index_map=0; index_map<v_m.size(); index_map++) {
		int shift_map = index_map*v_m(0).size();
		std::copy(v_m(index_map).begin(), v_m(index_map).end(), v_in.begin()+shift_map);
	}

	pop::VecF32 v_out(1);
	pop::VecF32 v_out_desired(1);
	v_out_desired(0) = -1;
	for (int epoch=0; epoch<nb_epoch; epoch++) {
		network.propagateFront(v_in, v_out);
		network.propagateBackFirstDerivate(v_out_desired);
		std::cout << epoch << "\t" << v_out(0) << std::endl;
	}
}

void test_neural_net_conv_cpu_mnist(const int nb_epoch) {
    pop::Vec<pop::Vec<pop::Mat2UI8> > number_training =  pop::TrainingNeuralNetwork::loadMNIST("/home/olivia/workspace/MNIST/train-images-idx3-ubyte","/home/olivia/workspace/MNIST/train-labels-idx1-ubyte");
    pop::Vec<pop::Vec<pop::Mat2UI8> > number_test =  pop::TrainingNeuralNetwork::loadMNIST("/home/olivia/workspace/MNIST/t10k-images-idx3-ubyte","/home/olivia/workspace/MNIST/t10k-labels-idx1-ubyte");

	double size_in= number_training(0)(0).getDomain()(0) * number_training(0)(0).getDomain()(1);
	std::cout << "size trainings: " << number_training(0).size() << std::endl;

	std::vector<struct layer_representation> v_layer;
	struct layer_representation lr;
	lr.type = LAYER_INPUT_MATRIX;
	//Simard: input is 29x29
	pop::Vec2I32 domain(29, 29);
	lr.sizei_map = domain(0);
	lr.sizej_map = domain(1);
	lr.nbr_map = 1;
	v_layer.push_back(lr);
	lr.type = LAYER_CONVOLUTIONAL;
	lr.nbr_map = 6;
	lr.radius_kernel = 2;
	lr.sub_resolution_factor = 2;
	v_layer.push_back(lr);
	lr.type = LAYER_CONVOLUTIONAL;
	lr.nbr_map = 50;
	lr.radius_kernel = 2;
	lr.sub_resolution_factor = 2;
	v_layer.push_back(lr);
	lr.type = LAYER_FULLY_CONNECTED;
	lr.nb_neurons = 100;
	v_layer.push_back(lr);
	lr.type = LAYER_FULLY_CONNECTED;
	lr.nb_neurons = number_training.size();
	v_layer.push_back(lr);
	GPUNeuralNetwork network(v_layer, 0.001);

	pop::Vec<pop::VecF32> vtraining_in;
	pop::Vec<pop::VecF32> vtraining_out;
    //convertit mat en vecteur
	pop::TrainingNeuralNetwork::convertMatrixToInputValueNeuron(vtraining_in,vtraining_out,number_training,domain,pop::NNLayerMatrix::Mass,pop::NNLayerMatrix::MinusOneToOne);

	pop::Vec<pop::VecF32> vtest_in;
	pop::Vec<pop::VecF32> vtest_out;
	pop::TrainingNeuralNetwork::convertMatrixToInputValueNeuron(vtest_in,vtest_out,number_test,domain,pop::NNLayerMatrix::Mass,pop::NNLayerMatrix::MinusOneToOne);

	number_training.clear();
	number_test.clear();

	size_t total_size_training = (vtraining_in.size()*vtraining_in(0).size() + vtraining_out.size()*vtraining_out(0).size()) * sizeof(vtraining_in(0)(0));
	size_t total_size_test = (vtest_in.size()*vtest_in(0).size() + vtest_out.size()*vtest_out(0).size()) * sizeof(vtest_in(0)(0));
	std::cout << "total training size: " << total_size_training << ", total size test: " << total_size_test << std::endl;


    //choisir ordre des images
	std::vector<int> v_global_rand(vtraining_in.size());
	for(unsigned int i=0;i<v_global_rand.size();i++)
        v_global_rand[i]=i; // vecteur de pointeurs qui pointent sur img

	std::cout<<"iter_epoch\t error_train\t error_test\t learning rate"<<std::endl;

    for (unsigned int i=0;i<nb_epoch;i++) {//une itération avec un set d'images ordonné différemment à chaque fois
		std::random_shuffle ( v_global_rand.begin(), v_global_rand.end() ,pop::Distribution::irand());
		int error_training=0,error_test=0;

            //phase training
		for(unsigned int j=0;j<v_global_rand.size();j++){
			pop::VecF32 vout;
            //algo propagate front
            network.propagateFront(vtraining_in(v_global_rand[j]),vout); // vtraining_in = images
            int label1 = std::distance(vout.begin(),std::max_element(vout.begin(),vout.end()));//distance entre begin et la case max pour atteindre la case ou ya le max
            //propagate back
			network.propagateBackFirstDerivate(vtraining_out(v_global_rand[j]));
            //v_training_out = valeur désiré
            //v_out = valeur reelle
			int label2 = std::distance(vtraining_out(v_global_rand[j]).begin(),std::max_element(vtraining_out(v_global_rand[j]).begin(),vtraining_out(v_global_rand[j]).end()));
			if(label1!=label2){
                error_training++; //calcul de erreur en plus
			}
		}

        //phase test
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
void test_neural_net_gpu(const int nb_epoch) {//
	std::vector<struct layer_representation> v_layer;
	struct layer_representation lr;
	lr.type = LAYER_INPUT;
	lr.nb_neurons = 2;
	v_layer.push_back(lr);
	lr.type = LAYER_FULLY_CONNECTED;
	lr.nb_neurons = 3;
	v_layer.push_back(lr);
	lr.nb_neurons = 1;
	v_layer.push_back(lr);
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


    //on verifie que on a assez de place sur la carte pour nos donnees
	size_t total_size_sets = (v_in.size()*v_in(0).size() + v_out.size()*v_out(0).size()) * sizeof(v_in(0)(0));
	size_t free, total;
	cudaMemGetInfo(&free, &total);
	if (total_size_sets > GPU_MEMORY_PRESSURE*free) {
		std::cerr << "Not enough memory on the GPU to process the whole sets at once. You need to copy the sets pieces by pieces" << std::endl;
		return;
	}

	//use the backpropagation algorithm with first order method
    std::vector<int> v_global_rand(v_in.size()); //tableau des indices qui accedent aux img
	for(unsigned int i=0;i<v_global_rand.size();i++)
		v_global_rand[i]=i;

    //copie des img sur le gpu d_in_set pointeur vers copie
	pop::F32* d_in_set = GPUNeuralNetwork::gpu_copyDataToGPU(v_in, 0, v_in.size());
    //valeurs desirees
	pop::F32* d_out_set = GPUNeuralNetwork::gpu_copyDataToGPU(v_out, 0, v_out.size());

    //valeurs de sortie reelles
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

        //algos
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

    //copie du reseau (x,y,w) sur le cpu
	network.copyNetworkFromGPU();

    //test the training on cpu
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

	std::vector<struct layer_representation> v_layer;
	struct layer_representation lr;
	lr.type = LAYER_INPUT;
	lr.nb_neurons = size_in;
	v_layer.push_back(lr);
	lr.type = LAYER_FULLY_CONNECTED;
	lr.nb_neurons = 1000;
	v_layer.push_back(lr);
	lr.nb_neurons = 1000;
	v_layer.push_back(lr);
	lr.nb_neurons = number_training.size();
	v_layer.push_back(lr);
	GPUNeuralNetwork network(v_layer, 0.001);

	pop::Vec<pop::VecF32> vtraining_in;
	pop::Vec<pop::VecF32> vtraining_out;
	pop::TrainingNeuralNetwork::convertMatrixToInputValueNeuron(vtraining_in,vtraining_out,number_training,number_training(0)(0).getDomain(),pop::NNLayerMatrix::Mass,pop::NNLayerMatrix::MinusOneToOne);

	pop::Vec<pop::VecF32> vtest_in;
	pop::Vec<pop::VecF32> vtest_out;
	pop::TrainingNeuralNetwork::convertMatrixToInputValueNeuron(vtest_in,vtest_out,number_test,number_test(0)(0).getDomain(),pop::NNLayerMatrix::Mass,pop::NNLayerMatrix::MinusOneToOne);

	number_training.clear();
	number_test.clear();

	network.gpu_learn(vtraining_in, vtraining_out, vtest_in, vtest_out, true, nb_epoch);
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

	std::vector<struct layer_representation> v_layer;
	struct layer_representation lr;

#if 0
	lr.type = LAYER_INPUT;
	lr.nb_neurons = size_in;
	v_layer.push_back(lr);

	lr.type = LAYER_FULLY_CONNECTED;
	switch (network_for_training) {
	case 5:
		for (int i=0; i<9; i++) {
			lr.nb_neurons = 1000;
			v_layer.push_back(lr);
		}
		break;
	case 4:
		lr.nb_neurons = 2500;
		v_layer.push_back(lr);
		/* no break */
	case 3:
		lr.nb_neurons = 2000;
		v_layer.push_back(lr);
		/* no break */
	case 2:
		lr.nb_neurons = 1500;
		v_layer.push_back(lr);
		/* no break */
	case 1:
		lr.nb_neurons = 1000;
		v_layer.push_back(lr);
		lr.nb_neurons = 500;
		v_layer.push_back(lr);
		break;
	default:
		std::cerr << "Database training: unknown network " << network_for_training << std::endl;
		return;
	}

	lr.type = LAYER_FULLY_CONNECTED;
	lr.nb_neurons = size_out;
	v_layer.push_back(lr);

#else

	lr.type = LAYER_INPUT_MATRIX;
	//Simard: input is 29x29
	pop::Vec2I32 domain(29, 29);
	lr.sizei_map = domain(0);
	lr.sizej_map = domain(1);
	lr.nbr_map = 1;
	v_layer.push_back(lr);
	lr.type = LAYER_CONVOLUTIONAL;
	lr.nbr_map = 6;
	lr.radius_kernel = 2;
	lr.sub_resolution_factor = 2;
	v_layer.push_back(lr);
	lr.type = LAYER_CONVOLUTIONAL;
	lr.nbr_map = 50;
	lr.radius_kernel = 2;
	lr.sub_resolution_factor = 2;
	v_layer.push_back(lr);
	lr.type = LAYER_FULLY_CONNECTED;
	lr.nb_neurons = 100;
	v_layer.push_back(lr);
	lr.type = LAYER_FULLY_CONNECTED;
	lr.nb_neurons = size_out;
	v_layer.push_back(lr);
#endif

	GPUNeuralNetwork network(v_layer, 0.001);
	std::cout << "Network created at " << getCurrentTime() << std::endl;
	network.save("/tmp/network.bin");
	std::cout << "saved" << std::endl;

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

void test_neural_net_conv_gpu(const int nb_epoch) {
	int size_input_matrix = 7;
	int nbr_map=1;

	std::vector<struct layer_representation> v_layer;
	struct layer_representation lr;
	lr.type = LAYER_INPUT_MATRIX;
	lr.sizei_map = size_input_matrix;
	lr.sizej_map = size_input_matrix;
	lr.nbr_map = nbr_map;
	v_layer.push_back(lr);
	lr.type = LAYER_CONVOLUTIONAL;
	lr.nbr_map = nbr_map;
	lr.radius_kernel = 1;
	lr.sub_resolution_factor = 2;
	v_layer.push_back(lr);
	lr.type = LAYER_FULLY_CONNECTED;
	lr.nb_neurons = 1;
	v_layer.push_back(lr);
	GPUNeuralNetwork network(v_layer, 0.01);

	std::cout << "\n********** GPU **********\n" << std::endl;

	network.gpu_displayNetwork();

	pop::Mat2F32 m(size_input_matrix,size_input_matrix);
	pop::DistributionNormal d(0,1);
	for(unsigned int i=0; i<m.size(); i++) {
		m(i) = d.randomVariable();
	}

	pop::Vec<pop::Mat2F32> v_m(1);
	v_m(0) = m;
	pop::Vec<pop::VecF32> v_in(1,pop::VecF32(size_input_matrix*size_input_matrix));
	for(unsigned int index_map=0; index_map<v_m.size(); index_map++) {
		int shift_map = index_map*v_m(0).size();
		std::copy(v_m(index_map).begin(), v_m(index_map).end(), v_in(0).begin()+shift_map);
	}

	pop::Vec<pop::VecF32> v_out(1,pop::VecF32(1));//1 vector of one scalar value
	v_out(0)(0)=-1;// -1

	size_t total_size_sets = (v_in.size()*v_in(0).size() + v_out.size()*v_out(0).size()) * sizeof(v_in(0)(0));
	size_t free, total;
	cudaMemGetInfo(&free, &total);
	if (total_size_sets > GPU_MEMORY_PRESSURE*free) {
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

	std::cout<<"iter_epoch\t error_train"<<std::endl;
	for(unsigned int i=0;i<nb_epoch;i++){
		std::random_shuffle ( v_global_rand.begin(), v_global_rand.end() , pop::Distribution::irand());

		for(unsigned int j=0;j<v_in.size();j++){
			network.gpu_propagateFront(d_in_set, v_in(0).size(), j, d_out);
			network.gpu_propagateBackFirstDerivate(d_out_set, v_out(0).size(), j);
		}

		pop::F32 out;
		cudaMemcpy(&out, d_out, sizeof(out), cudaMemcpyDeviceToHost);
		if (i==0 || i==nb_epoch-1) {
			std::cout<<i<<"\t"<<out<<std::endl;
		}
	}

	cudaFree(d_out);
	cudaFree(d_in_set);
	cudaFree(d_out_set);
}

void test_neural_net_conv_gpu_mnist(const int nb_epoch) {
    pop::Vec<pop::Vec<pop::Mat2UI8> > number_training =  pop::TrainingNeuralNetwork::loadMNIST("/home/olivia/workspace/MNIST/train-images-idx3-ubyte","/home/olivia/workspace/MNIST/train-labels-idx1-ubyte");
    pop::Vec<pop::Vec<pop::Mat2UI8> > number_test =  pop::TrainingNeuralNetwork::loadMNIST("/home/olivia/workspace/MNIST/t10k-images-idx3-ubyte","/home/olivia/workspace/MNIST/t10k-labels-idx1-ubyte");

	double size_in= number_training(0)(0).getDomain()(0) * number_training(0)(0).getDomain()(1);
	std::cout << "size trainings: " << number_training(0).size() << std::endl;

	std::vector<struct layer_representation> v_layer;
	struct layer_representation lr;
	lr.type = LAYER_INPUT_MATRIX;
	//Simard: input is 29x29
	pop::Vec2I32 domain(29, 29);
	lr.sizei_map = domain(0);
	lr.sizej_map = domain(1);
	lr.nbr_map = 1;
	v_layer.push_back(lr);
	lr.type = LAYER_CONVOLUTIONAL;
	lr.nbr_map = 6;
	lr.radius_kernel = 2;
	lr.sub_resolution_factor = 2;
	v_layer.push_back(lr);
	lr.type = LAYER_CONVOLUTIONAL;
	lr.nbr_map = 50;
	lr.radius_kernel = 2;
	lr.sub_resolution_factor = 2;
	v_layer.push_back(lr);
	lr.type = LAYER_FULLY_CONNECTED;
	lr.nb_neurons = 100;
	v_layer.push_back(lr);
	lr.type = LAYER_FULLY_CONNECTED;
	lr.nb_neurons = number_training.size();
	v_layer.push_back(lr);
	GPUNeuralNetwork network(v_layer, 0.001);

	pop::Vec<pop::VecF32> vtraining_in;
	pop::Vec<pop::VecF32> vtraining_out;
	pop::TrainingNeuralNetwork::convertMatrixToInputValueNeuron(vtraining_in,vtraining_out,number_training,domain,pop::NNLayerMatrix::Mass,pop::NNLayerMatrix::MinusOneToOne);

	pop::Vec<pop::VecF32> vtest_in;
	pop::Vec<pop::VecF32> vtest_out;
	pop::TrainingNeuralNetwork::convertMatrixToInputValueNeuron(vtest_in,vtest_out,number_test,domain,pop::NNLayerMatrix::Mass,pop::NNLayerMatrix::MinusOneToOne);

	number_training.clear();
	number_test.clear();

	network.gpu_learn(vtraining_in, vtraining_out, vtest_in, vtest_out, true, nb_epoch);
}

void test_neural_net() {
	pop::Vec2I32 domain(640, 480);
	std::vector<struct layer_representation> v_layer;
	struct layer_representation lr;
	lr.type = LAYER_INPUT_MATRIX;
	lr.sizei_map = domain(0);
	lr.sizej_map = domain(1);
	lr.nbr_map = 1;
	v_layer.push_back(lr);
	lr.type = LAYER_CONVOLUTIONAL;
	lr.nbr_map = 2;
	lr.radius_kernel = 1;
	lr.sub_resolution_factor = 1;
	v_layer.push_back(lr);
	lr.type = LAYER_CONVOLUTIONAL;
	lr.nbr_map = 1;
	lr.radius_kernel = 1;
	lr.sub_resolution_factor = 1;
	v_layer.push_back(lr);
	GPUNeuralNetwork network(v_layer, 0.01);

	/*
	 * For this example, you need to specify the filters when creating the network. In createNetwork():
		{
			//layer 1 : kernels are ID
			struct layer& l = h_network->_layers[1];
			for (unsigned int j=0; j<l._W_height * l._W_width; j++) {
			l._W[j] = 0;
			}
			l._W[4] = 1;
			l._W[13] = 1;
		}
		{
			//layer 2: kernels are Sobel
			struct layer& l = h_network->_layers[2];
			l._W[0] = -1; l._W[1] = -2; l._W[2] = -1;
			l._W[3] =  0; l._W[4] =  0; l._W[5] =  0;
			l._W[6] =  1; l._W[7] =  2; l._W[8] =  1; l._W[9] = 0; //bias neuron
			l._W[10] = -1; l._W[11] = 0; l._W[12] = 1;
			l._W[13] = -2; l._W[14] = 0; l._W[15] = 2;
			l._W[16] = -1; l._W[17] = 0; l._W[18] = 1; l._W[19] = 0; //bias neuron
		}
	 */

	network.displayNetwork();

	pop::Mat2UI8 image_in(domain);
    image_in.load("/home/olivia/workspace/Population/image/Bikesgray.jpg");
	pop::VecF32 vin = pop::NNLayerMatrix::inputMatrixToInputNeuron(image_in,domain,pop::NNLayerMatrix::Mass,pop::NNLayerMatrix::MinusOneToOne);

	pop::Vec2I32 domain_out(domain(0)-4, domain(1)-4);
	pop::Mat2UI8 image_out(domain_out);

	std::cout << "Test on CPU" << std::endl;

	pop::VecF32 vout;
	network.propagateFront(vin, vout);

	for (int i=0; i<domain_out(0); i++) {
		for (int j=0; j<domain_out(1); j++) {
			image_out(i, j) = vout(i*domain_out(1)+j);
		}
	}
    image_out.save("/home/olivia/workspace/Population/image/Bikesgray-sobel-cpu.jpg");

	std::cout << "Test on GPU" << std::endl;

	pop::F32* d_in;
	cudaMalloc(&d_in, vin.size()*sizeof(*d_in));
	cudaMemcpy(d_in, vin.data(), vin.size()*sizeof(vin(0)), cudaMemcpyHostToDevice);
	pop::F32* d_out;
	cudaMalloc(&d_out, domain_out(0)*domain_out(1)*sizeof(*d_out));

	network.gpu_propagateFront(d_in, vin.size(), 0, d_out);

	cudaMemcpy(vout.data(), d_out, vout.size()*sizeof(vout(0)), cudaMemcpyDeviceToHost);
	for (int i=0; i<domain_out(0); i++) {
		for (int j=0; j<domain_out(1); j++) {
			image_out(i, j) = vout(i*domain_out(1)+j);
		}
	}
    image_out.save("/home/olivia/workspace/Population/image/Bikesgray-sobel-gpu.jpg");

	cudaFree(d_out);
	cudaFree(d_in);

	std::cout << "Tests ok!" << std::endl;


#if 0
	const int nb_epoch = 200;
	int size_input_matrix = 7;
	int nbr_map=1;

	std::vector<struct layer_representation> v_layer;
	struct layer_representation lr;
	lr.type = LAYER_INPUT_MATRIX;
	lr.sizei_map = size_input_matrix;
	lr.sizej_map = size_input_matrix;
	lr.nbr_map = nbr_map;
	v_layer.push_back(lr);
	lr.type = LAYER_CONVOLUTIONAL;
	lr.nbr_map = nbr_map;
	lr.radius_kernel = 1;
	lr.sub_resolution_factor = 2;
	v_layer.push_back(lr);
	lr.type = LAYER_FULLY_CONNECTED;
	lr.nb_neurons = 1;
	v_layer.push_back(lr);
	GPUNeuralNetwork network(v_layer, 0.01);

	std::cout << "\n********** GPU **********\n" << std::endl;

	network.gpu_displayNetwork();

	pop::Mat2F32 m(size_input_matrix,size_input_matrix);
	pop::DistributionNormal d(0,1);
	for(unsigned int i=0; i<m.size(); i++) {
		m(i) = d.randomVariable();
	}

	pop::Vec<pop::Mat2F32> v_m(1);
	v_m(0) = m;
	pop::Vec<pop::VecF32> v_in(1,pop::VecF32(size_input_matrix*size_input_matrix));
	for(unsigned int index_map=0; index_map<v_m.size(); index_map++) {
		int shift_map = index_map*v_m(0).size();
		std::copy(v_m(index_map).begin(), v_m(index_map).end(), v_in(0).begin()+shift_map);
	}

	pop::Vec<pop::VecF32> v_out(1,pop::VecF32(1));//1 vector of one scalar value
	v_out(0)(0)=-1;// -1

	size_t total_size_sets = (v_in.size()*v_in(0).size() + v_out.size()*v_out(0).size()) * sizeof(v_in(0)(0));
	size_t free, total;
	cudaMemGetInfo(&free, &total);
	if (total_size_sets > GPU_MEMORY_PRESSURE*free) {
		std::cerr << "Not enough memory on the GPU to process the whole sets at once. You need to copy the sets pieces by pieces" << std::endl;
		return;
	}

	pop::F32* d_in_set = GPUNeuralNetwork::gpu_copyDataToGPU(v_in, 0, v_in.size());
	pop::F32* d_out_set = GPUNeuralNetwork::gpu_copyDataToGPU(v_out, 0, v_out.size());

	pop::F32* d_out;
	cudaMalloc(&d_out, v_out(0).size() * sizeof(v_in(0)(0)));

	std::cout<<"iter_epoch\t error_train"<<std::endl;
	for(unsigned int i=0;i<nb_epoch;i++){
		for(unsigned int j=0;j<v_in.size();j++){
			network.gpu_propagateFront(d_in_set, v_in(0).size(), j, d_out);
			network.gpu_propagateBackFirstDerivate(d_out_set, v_out(0).size(), j);
		}
	}

	std::cout<<"gpu test"<<std::endl;
	for(unsigned int j=0;j<v_in.size();j++){
		network.gpu_propagateFront(d_in_set, v_in(0).size(), j, d_out);

		pop::F32 out;
		cudaMemcpy(&out, d_out, sizeof(out), cudaMemcpyDeviceToHost);
		std::cout << "gpu: " << out << std::endl;
	}

	cudaFree(d_out);
	cudaFree(d_in_set);
	cudaFree(d_out_set);

	network.copyNetworkFromGPU();

	//test the training
	for(unsigned int j=0;j<v_in.size();j++){
		pop::VecF32 vout;
		network.propagateFront(v_in(0), vout);
		std::cout << "cpu: " << vout << std::endl;
	}
#endif
}

#endif
