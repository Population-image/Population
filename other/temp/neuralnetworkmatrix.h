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
}
#endif
