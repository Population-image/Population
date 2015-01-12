#include"Population.h"//Single header
using namespace pop;//Population namespace
int main(){

    Mat2UI8 img;
    img.load(POP_PROJECT_SOURCE_DIR+std::string("/image/iex.png"));//replace this path by those on your computer
    img.display("Initial image",false);
    img = PDE::nonLinearAnisotropicDiffusion(img);//filtering
    img.display();
    double value;
    Mat2UI8 threshold = Processing::thresholdOtsuMethod(img,value);//threshold segmentation
    threshold.save("iexthreshold.png");
    Mat2RGBUI8 color = Visualization::labelForeground(threshold,img);//Visual validation
    color.display("Segmented image",true);
    return 0;
}
