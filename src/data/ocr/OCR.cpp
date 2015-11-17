#include"PopulationConfig.h"
#include "data/ocr/OCR.h"

#include "data/vec/VecN.h"
#include "algorithm/GeometricalTransformation.h"
#include "algorithm/Processing.h"
#include"data/notstable/CharacteristicCluster.h"
#include"data/mat/MatNDisplay.h"

namespace pop
{
OCR::~OCR()
{

}
OCRNeuralNetwork::~OCRNeuralNetwork(){

}

//NeuralNetworkFeedForward &   OCRNeuralNetwork::neuralNetworkFeedForward(){
//    return _n;
//}

//const NeuralNetworkFeedForward &   OCRNeuralNetwork::neuralNetworkFeedForward()const{
//    return _n;
//}
NeuralNet &   OCRNeuralNetwork::neuralNetworkFeedForward(){
    return _n;
}

const NeuralNet &   OCRNeuralNetwork::neuralNetworkFeedForward()const{
    return _n;
}

char OCRNeuralNetwork::parseMatrix(const Mat2UI8 & m){
    if(_n.layers().size()==0)
    {
        std::cerr<<"Neural network is empty. Used setDictionnary to construct it. I give one for digit number.  the folder $${PopulationPath}/file/handwrittendigitneuralnetwork.xml, you can find the handwritten dictionnary. So the code is ocr.setDictionnary($${PopulationPath}/file/neuralnetwork.xml) ";
    }
    else{

        VecF32 vin= _n.inputMatrixToInputNeuron(m);
        VecF32 vout;
        _n.forwardCPU(vin,vout);
        //std::cout << "vout of neural network : " << vout << std::endl;
        //                _n.propagateFront(vin,vout);
        VecF32::iterator itt = std::max_element(vout.begin(),vout.end());
        std::cout << __FILE__ << "::" << __LINE__ << "itt : " << *itt << std::endl;
        int label_max = std::distance(vout.begin(),itt);
        F32 value_max = *itt;
//        std::cout << "value_max : " << value_max << std::endl;
        if(value_max<0)
            _isrecognized=false;
        else
            _isrecognized=true;
        _confidence = (std::min)(100,static_cast<int>(value_max*100));
        std::string c= _n.label2String()[label_max];
//        std::cout << "label2String of NN : " << _n.label2String() << std::endl;
        return c[0];

    }
    return '?';
}

int OCRNeuralNetwork::characterConfidence(){
    return _confidence;
}

bool OCRNeuralNetwork::setDictionnary(std::string xmlfile){
   std::cout << "xmlfile : " << xmlfile << std::endl;
    if(BasicUtility::isFile(xmlfile)){
        _n.load(xmlfile.c_str());
        return true;
    }else{
        return false;
    }

}

bool OCRNeuralNetwork::setDictionnaryByteArray(const char * byte_array){
    _n.loadByteArray(byte_array);
    return true;
}

bool OCRNeuralNetwork::isRecognitionCharacter(){
    return _isrecognized;
}

}


