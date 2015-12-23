
#include"data/notstable/neuralnetworkblas.h"
#include"data/notstable/blas.h"

namespace popblas {

NeuralLayer::~NeuralLayer(){

}

NeuralLayerLinear::NeuralLayerLinear(unsigned int nbr_neurons)
    :__Y(nbr_neurons),__X(nbr_neurons)
{

}

NeuralLayerLinear::NeuralLayerLinear(const NeuralLayerLinear & net){
    __Y=net.__Y;
    __X=net.__X;

    _d_E_Y=net._d_E_Y;
    _d_E_X=net._d_E_X;

}

NeuralLayerLinear&  NeuralLayerLinear::operator=(const NeuralLayerLinear & net){
    __Y=net.__Y;
    __X=net.__X;

    _d_E_Y=net._d_E_Y;
    _d_E_X=net._d_E_X;
    return *this;
}

BMat& NeuralLayerLinear::X(){return __X;}
const BMat& NeuralLayerLinear::X()const{return __X;}
BMat& NeuralLayerLinear::d_E_X(){return _d_E_X;}
void NeuralLayerLinear::setTrainable(bool istrainable){
    if(istrainable==true){
        this->_d_E_Y = this->__X;
        this->_d_E_X = this->__X;
    }else{
        this->_d_E_Y = 0;
        this->_d_E_X = 0;
    }
}

void NeuralLayerLinear::print(){
    std::cout<<"Number neuron="<<this->__X.getDomain().multCoordinate()<<std::endl;
}

void NeuralLayerLinearFullyConnected::backwardCPU(NeuralLayer& layer_previous){

    BMat& d_E_X_previous= layer_previous.d_E_X();
    for(unsigned int i=0;i<this->__Y.size();i++){
        this->_d_E_Y(i) = this->_d_E_X(i)*NeuronSigmoid::derivedActivation(this->__X(i));
    }

    this->_d_E_W = 0;
    popblas::blas::ger(1, this->_d_E_Y, this->_X_biais, this->_d_E_W);
    for(unsigned int j=0;j<d_E_X_previous.size();j++){
        d_E_X_previous(j)=0;
        for(unsigned int i=0;i<this->_W.sizeI();i++){
            d_E_X_previous(j)+=this->_d_E_Y(i)*this->_W(i,j);
        }
    }
    // TODO : rewrite the last five lines using BLAS
}
