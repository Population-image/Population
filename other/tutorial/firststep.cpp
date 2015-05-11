#include"Population.h"//Single header
using namespace pop;//Population namespace
int main()
{
	 
    Mat2UI8 img;//2d grey-level image object
    img.load(POP_PROJECT_SOURCE_DIR+std::string("/image/iex.png"));//replace this path by those on your computer
    img = PDE::nonLinearAnisotropicDiffusion(img);//filtering
    int value;
    Mat2UI8 threshold = Processing::thresholdOtsuMethod(img,value);//threshold segmentation
    threshold.save("iexthreshold.pgm");
    Mat2RGBUI8 color = Visualization::labelForeground(threshold,img);//Visual validation
    color.display();

    DistributionUniformInt d(0,10);
    std::cout<<d.randomVariable()<<std::endl;
    OCRNeuralNetwork ocr;
    ocr.setDictionnary("/home/vincent/DEV2/DEV/LAPI-ACCES/usr/bin/dictionaries/neuralnetwork.xml");
    return 0;
}

