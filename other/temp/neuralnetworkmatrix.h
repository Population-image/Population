#ifndef NEURALNETWORKMATRIX_H
#define NEURALNETWORKMATRIX_H

#include"Population.h"
using namespace pop;//Population namespace


class NeuralNetworkFullyConnected
{
public:
	double sigmoid(double x){ return 1.7159*tanh(0.66666667*x);}
	//    double derived_sigmoid(double S){ return 1.7159*(1+(0.66666667*S))*(1.7159-(0.66666667*S));}  // derivative of the sigmoid as a function of the sigmoid's output

	double derived_sigmoid(double S){ return 0.666667f/1.7159f*(1.7159f+(S))*(1.7159f-(S));}  // derivative of the sigmoid as a function of the sigmoid's output


	void createNetwork(std::vector<unsigned int> v_layer){
		for(unsigned int i=0;i<v_layer.size();i++){
			int size_layer =v_layer[i];
			if(i!=v_layer.size()-1)
				_X.push_back(VecF32(size_layer+1,1));//add the neuron with constant value 1
			else
				_X.push_back(VecF32(size_layer,1));//except for the last one
			_Y.push_back(VecF32(size_layer));

			if(i!=0){
				int size_layer_previous = _X[i-1].size();
				Mat2F32  R(size_layer  ,size_layer_previous);
				DistributionNormal n(0,1./std::sqrt(size_layer_previous));
				for(unsigned int i = 0;i<R.size();i++){
					R[i]= n.randomVariable();
				}
				_W.push_back(R);
			}
		}
	}

	void propagateFront(const pop::VecF32& in , pop::VecF32 &out){
		std::copy(in.begin(),in.end(),_X[0].begin());

		for(unsigned int layer_index=0;layer_index<_W.size();layer_index++){

			_Y[layer_index+1] = _W[layer_index] * _X[layer_index];
			for(unsigned int j=0;j<_Y[layer_index+1].size();j++){
				_X[layer_index+1][j] = sigmoid(_Y[layer_index+1][j]);
			}
		}
		if(out.size()!=_X.rbegin()->size())
			out.resize(_X.rbegin()->size());
		std::copy(_X.rbegin()->begin(),_X.rbegin()->begin()+out.size(),out.begin());
	}
	void propagateBackFirstDerivate(const pop::VecF32& desired_output){
		if(_d_E_X.size()==0){
			_d_E_X = _X;
			_d_E_Y = _Y;
			_d_E_W = _W;
		}
		for( int index_layer=_X.size()-1;index_layer>0;index_layer--){
			//X error
			if(index_layer==_X.size()-1){
				for(unsigned int j=0;j<_X[index_layer].size();j++){
					_d_E_X[index_layer][j] = (_X[index_layer][j]-desired_output[j]);
				}
			}

			for(unsigned int i=0;i<_d_E_Y[index_layer].size();i++){
				_d_E_Y[index_layer][i] = _d_E_X[index_layer][i] * derived_sigmoid(_X[index_layer][i]);
			}

			for(unsigned int j=0;j<_d_E_W[index_layer-1].sizeJ();j++){
				for(unsigned int i=0;i<_d_E_W[index_layer-1].sizeI();i++){
					_d_E_W[index_layer-1](i,j)= _d_E_Y[index_layer][i]*_X[index_layer-1][j];
					_W[index_layer-1](i,j)= _W[index_layer-1](i,j) - _eta* _d_E_W[index_layer-1](i,j);
				}
			}

			for(unsigned int j=0;j<_X[index_layer-1].size();j++){
				_d_E_X[index_layer-1][j]=0;
				for(unsigned int i=0;i<_W[index_layer-1].sizeI();i++){
					_d_E_X[index_layer-1][j]+=_W[index_layer-1](i,j)*_d_E_Y[index_layer][i];
				}
			}
		}

	}

	void printNeuronVector(pop::VecF32 V, std::string label) {
		std::cout << label << "(" << V.size() << ") = [";
		std::cout << V << "]" << std::endl;
	}

	void printWeightMatrix(pop::Mat2F32 M, std::string label) {
		std::cout << label << "(" << M.sizeI() << ", " << M.sizeJ() << ") = [" << std::endl;
		std::cout << M << "]" << std::endl;
	}

	void printNetwork(void) {
		std::cout << "####################" << std::endl;
		std::cout << "Number of layers: " << _X.size() << std::endl;
		std::cout << "Eta: " << _eta << std::endl;

		for (unsigned int l=0; l<_X.size(); l++) {
			std::cout << "\n-- Layer " << l << ":" << std::endl;

			printNeuronVector(_X[l], "_X");
			printNeuronVector(_Y[l], "_Y");
			if(_d_E_X.size()==0){
				std::cout << "_d_E_X = NULL" << std::endl;
				std::cout << "_d_E_Y = NULL" << std::endl;
			} else {
				printNeuronVector(_d_E_X[l], "_d_E_X");
				printNeuronVector(_d_E_Y[l], "_d_E_Y");
			}
			if (l != 0) {
				printWeightMatrix(_W[l-1], "_W");
				if(_d_E_X.size()==0){
					std::cout << "_d_E_W = NULL" << std::endl;
				} else {
					printWeightMatrix(_d_E_W[l-1], "_d_E_W");
				}
			} else {
				std::cout << "_W = NULL" << std::endl;
				std::cout << "_d_E_W = NULL" << std::endl;
			}
		}

		std::cout << "####################" << std::endl;

	}

	double _eta;
	Vec<pop::VecF32>  _X;
	Vec<pop::VecF32>  _Y;
	Vec<pop::Mat2F32> _W;
	Vec<pop::VecF32>  _d_E_X;
	Vec<pop::VecF32>  _d_E_Y;
	Vec<pop::Mat2F32> _d_E_W;
};




void neuralnetwortest(){
	NeuralNetworkFullyConnected network;

	//XOR
	std::vector<unsigned int> v_layer;
	v_layer.push_back(2);
	v_layer.push_back(3);
	v_layer.push_back(1);
	network.createNetwork(v_layer);
	network._eta = 0.01;

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
			network.propagateFront(v_in(v_global_rand[j]),vout);
			network.propagateBackFirstDerivate(v_out(v_global_rand[j]));

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
		network.propagateFront(v_in(j), vout);
		std::cout<<vout<<std::endl;// we obtain the expected value -1 , 1 , 1 , -1
	}

#if 0
	{
	    NeuralNetworkFeedForward n;
	    n.addInputLayer(2);//2 scalar input
	    n.addLayerFullyConnected(3);// 1 fully connected layer with 3 neurons
	    n.addLayerFullyConnected(1);// 1 scalar output
	    //create the training set
	    // (-1,-1)->-1
	    // ( 1,-1)-> 1
	    // (-1, 1)-> 1
	    // ( 1, 1)->-1
	    Vec<VecF32> v_in(4,VecF32(2));//4 vector of two scalar values
	    v_in(0)(0)=-1;v_in(0)(1)=-1; // (-1,-1)
	    v_in(1)(0)= 1;v_in(1)(1)=-1; // ( 1,-1)
	    v_in(2)(0)=-1;v_in(2)(1)= 1; // (-1, 1)
	    v_in(3)(0)= 1;v_in(3)(1)= 1; // ( 1, 1)

	    Vec<VecF32> v_out(4,VecF32(1));//4 vector of one scalar value
	    v_out(0)(0)=-1;// -1
	    v_out(1)(0)= 1;//  1
	    v_out(2)(0)= 1;//  1
	    v_out(3)(0)=-1;// -1
	    //use the backprogation algorithm with first order method
	    TrainingNeuralNetwork::trainingFirstDerivative(n,v_in,v_out,0.01,1000);
	    //test the training
	    for(int j=0;j<4;j++){
	        VecF32 vout;
	        n.propagateFront(v_in(j),vout);
	        std::cout<<vout<<std::endl;// we obtain the expected value -1 , 1 , 1 , -1
	    }
	}
#endif


	//MNIST
#if 0
	Vec<Vec<Mat2UI8> > number_training =  TrainingNeuralNetwork::loadMNIST("/media/pl/shared/PL/neural_nets_samples/MNIST/train-images-idx3-ubyte","/media/pl/shared/PL/neural_nets_samples/MNIST/train-labels-idx1-ubyte");
	Vec<Vec<Mat2UI8> > number_test =  TrainingNeuralNetwork::loadMNIST("/media/pl/shared/PL/neural_nets_samples/MNIST/t10k-images-idx3-ubyte","/media/pl/shared/PL/neural_nets_samples/MNIST/t10k-labels-idx1-ubyte");

	double size_in=number_training(0)(0).getDomain()(0)*number_training(0)(0).getDomain()(1);
	std::cout<<"size trainings: "<<number_training(0).size()<<std::endl;
	std::vector<unsigned int> v_layer;
	v_layer.push_back(size_in);
	v_layer.push_back(1000);
	v_layer.push_back(1000);
	v_layer.push_back(number_training.size());
	network.createNetwork(v_layer);
	network._eta = 0.001;

	Vec<VecF32> vtraining_in;
	Vec<VecF32> vtraining_out;

	double ratio = 1;
	TrainingNeuralNetwork::convertMatrixToInputValueNeuron(vtraining_in,vtraining_out,number_training,number_training(0)(0).getDomain(),NNLayerMatrix::Mass,NNLayerMatrix::MinusOneToOne,ratio);

	Vec<VecF32> vtest_in;
	Vec<VecF32> vtest_out;
	TrainingNeuralNetwork::convertMatrixToInputValueNeuron(vtest_in,vtest_out,number_test,number_training(0)(0).getDomain(),NNLayerMatrix::Mass,NNLayerMatrix::MinusOneToOne,1);

	number_training.clear();
	number_test.clear();

	std::vector<int> v_global_rand(vtraining_in.size());
	for(unsigned int i=0;i<v_global_rand.size();i++)
		v_global_rand[i]=i;

	std::cout<<"iter_epoch\t error_train\t error_test\t learning rate"<<std::endl;

	for(unsigned int i=0;i<100;i++){
		std::random_shuffle ( v_global_rand.begin(), v_global_rand.end() ,Distribution::irand());
		int error_training=0,error_test=0;

		for(unsigned int j=0;j<v_global_rand.size();j++){
			VecF32 vout;
			network.propagateFront(vtraining_in(v_global_rand[j]),vout);
			int label1 = std::distance(vout.begin(),std::max_element(vout.begin(),vout.end()));
			network.propagateBackFirstDerivate(vtraining_out(v_global_rand[j]));
			int label2 = std::distance(vtraining_out(v_global_rand[j]).begin(),std::max_element(vtraining_out(v_global_rand[j]).begin(),vtraining_out(v_global_rand[j]).end()));
			if(label1!=label2){
				error_training++;
			}
		}
		for(unsigned int j=0;j<vtest_in.size();j++){
			VecF32 vout;
			network.propagateFront(vtest_in(j),vout);
			int label1 = std::distance(vout.begin(),std::max_element(vout.begin(),vout.end()));
			int label2 = std::distance(vtest_out(j).begin(),std::max_element(vtest_out(j).begin(),vtest_out(j).end()));
			if(label1!=label2){
				error_test++;
			}
		}
		network._eta *=0.9;
		std::cout<<i<<"\t"<<error_training*1./v_global_rand.size()<<"\t"<<error_test*1./vtest_in.size() <<"\t"<<network._eta <<std::endl;
	}
#endif
}
#endif
