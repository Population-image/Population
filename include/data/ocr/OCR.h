#ifndef OCR_H
#define OCR_H
#include<string>
#include"data/mat/MatN.h"
#include"data/neuralnetwork/NeuralNetwork.h"
namespace pop
{
/*! \ingroup Other
* \defgroup OCR OCR
* \brief
* @{
*/
class POP_EXPORTS OCR
{
public:
    virtual ~OCR();

    std::string parseText(const Mat2UI8 & binary,int nbr_pixels_width_caracter);
    /*!
    \brief apply the OCR on a binary matrix containing a single caracter
    \param binary binary input matrix
    \return OCR single character
    !*/
    virtual char parseMatrix(const Mat2UI8 & binary)=0;
    virtual bool isRecognitionCharacter()=0;

    /*!
    \brief return the the character confidence after OCR parsing (between 0 and 100) in an array
    \return  confidence value of the character
    !*/
    virtual int characterConfidence()=0;

    virtual bool setDictionnary(std::string path_dic)=0;
    virtual bool setDictionnaryByteArray(const char *  byte_array)=0;

private:
    std::string _parseTextByContrast(const Mat2UI8 & binary,int nbr_pixels_width_caracter);
};

class POP_EXPORTS OCRNeuralNetwork : public OCR
{
private:
    NeuralNetworkFeedForward _n;
    int _confidence;
    bool _isrecognized;
public:
    NeuralNetworkFeedForward &   neuralNetworkFeedForward();
    const NeuralNetworkFeedForward &   neuralNetworkFeedForward()const;
    char parseMatrix(const Mat2UI8 & binary);
    bool isRecognitionCharacter();
    int characterConfidence();
    bool setDictionnary(std::string path_dic);
    bool setDictionnaryByteArray(const char * byte_array);

};
/*!
@}
*/



}
#endif // OCR_H
