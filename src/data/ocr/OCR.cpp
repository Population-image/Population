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
NeuralNetworkFeedForward &   OCRNeuralNetwork::neuralNetworkFeedForward(){
    return _n;
}

const NeuralNetworkFeedForward &   OCRNeuralNetwork::neuralNetworkFeedForward()const{
    return _n;
}



std::string OCR::parseText(const Mat2UI8 & m,int nbr_pixels_width_caracter){

    std::string str1 = _parseTextByContrast(m,nbr_pixels_width_caracter);
    Mat2UI8 m_contrast = m.opposite();
    std::string str2 =  _parseTextByContrast(m_contrast,nbr_pixels_width_caracter);
    return str1+"\n"+str2;
}
std::string OCR::_parseTextByContrast(const Mat2UI8 & m,int nbr_pixels_width_caracter){
    Mat2UI8 threhold =        Processing::thresholdNiblackMethod(m,0.2f,3*nbr_pixels_width_caracter,-20);

    Mat2UI32 label = Processing::clusterToLabel(threhold,0);


    CharacteristicClusterFilterMass filter_mass;
    filter_mass._min = nbr_pixels_width_caracter*10;
    filter_mass._max = nbr_pixels_width_caracter*1000;

    CharacteristicClusterFilterAsymmetryHeightPerWidth filter_asymmetry;
    filter_asymmetry._min =0.5;
    filter_asymmetry._max = 20;
    Vec<CharacteristicClusterFilter*> v_filter;
    v_filter.push_back(&filter_mass);
    v_filter.push_back(&filter_asymmetry);

    label =  applyClusterFilter(label,v_filter );

    pop::Vec<CharacteristicClusterDistance*> v_dist;
    pop::Vec<F32> v_weight;


    CharacteristicClusterDistanceHeight dist_height;
    CharacteristicClusterDistanceWidthInterval dist_interval_width;
    CharacteristicClusterDistanceHeightInterval dist_interval_height;
    v_dist.push_back(&dist_height);
    v_dist.push_back(&dist_interval_width);
    v_dist.push_back(&dist_interval_height);
    v_weight.push_back(10);
    v_weight.push_back(0.1f);
    v_weight.push_back(4);
    Vec<Vec<Mat2UI8> > v_v_img = applyGraphCluster(label,v_dist,v_weight,0.5f);

    std::string str;
    for(unsigned int i=0;i<v_v_img.size();i++){
        std::string str2;
        bool parse=false;
        for(unsigned int j=0;j<v_v_img(i).size();j++){

            char c = this->parseMatrix(v_v_img(i)(j));
//            std::cout<<i<<" "<<c<<" "<<this->characterConfidence() <<std::endl;
//            v_v_img(i)(j).display();
            if(this->isRecognitionCharacter()){
                str2.push_back(c);
                parse=true;
            }
        }
//        std::cout<<str2<<std::endl;
        if(str.size()!=0&&parse==true){

            str=str+" ";
          }
        str=str+str2;
    }
    return str;
}

char OCRNeuralNetwork::parseMatrix(const Mat2UI8 & m){
    if(_n.layers().size()==0)
    {
        std::cerr<<"Neural network is empty. Used setDictionnary to construct it. I give one for digit number.  the folder $${PopulationPath}/file/handwrittendigitneuralnetwork.xml, you can find the handwritten dictionnary. So the code is ocr.setDictionnary($${PopulationPath}/file/neuralnetwork.xml) ";
    }
    else{

        VecF32 vin= _n.inputMatrixToInputNeuron(m);
        VecF32 vout;
        _n.propagateFront(vin,vout);
        VecF32::iterator itt = std::max_element(vout.begin(),vout.end());
        int label_max = std::distance(vout.begin(),itt);
        F32 value_max = *itt;
        if(value_max<0)
            _isrecognized=false;
        else
            _isrecognized=true;
        _confidence = value_max*100;
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
    return true;

}

bool OCRNeuralNetwork::isRecognitionCharacter(){
    return _isrecognized;

}

}


