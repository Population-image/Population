#include"Population.h"//Single header
using namespace pop;//Population namespace
int main()
{
    CollectorExecutionInformationSingleton::getInstance()->setActivate(true);//Activate the information manager
    try{//Enclose this portion of code in a try block
        Mat2UI8 img;//2d grey-level image object
        img.load(POP_PROJECT_SOURCE_DIR+std::string("/image/iex.png"));//replace this path by those on your computer
        img = PDE::nonLinearAnisotropicDiffusion(img);//filtering
        double value;
        Mat2UI8 threshold = Processing::thresholdOtsuMethod(img,value);//threshold segmentation
        threshold.save("iexthreshold.pgm");
        Mat2RGBUI8 color = Visualization::labelForeground(threshold,img);//Visual validation
        color.display();
    }
    catch(const pexception &e){
        e.display();//Display the error in a window
    }
    return 0;
}

