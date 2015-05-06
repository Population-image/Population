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
        //                _n.propagateFront(vin,vout);
        VecF32::iterator itt = std::max_element(vout.begin(),vout.end());
        int label_max = std::distance(vout.begin(),itt);
        F32 value_max = *itt;
        if(value_max<0)
            _isrecognized=false;
        else
            _isrecognized=true;
        _confidence = static_cast<int>(value_max*100);
        std::string c= _n.label2String()[label_max];
        return c[0];

    }
    return '?';
}
int OCRNeuralNetwork::characterConfidence(){
    return _confidence;
}
bool OCRNeuralNetwork::setDictionnary(std::string xmlfile){
    if(BasicUtility::isFile(xmlfile)){
        _n.load(xmlfile.c_str());
        return true;
    }else{
        return false;
    }

}
bool OCRNeuralNetwork::setDictionnaryByteArray(const char * byte_array){
    _n.loadByteArray(byte_array);
}


bool OCRNeuralNetwork::isRecognitionCharacter(){
    return _isrecognized;

}

}


