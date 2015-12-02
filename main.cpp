#include"Population.h"
using namespace pop;//Population namespace

int main(){
    Mat2UI8 m(5,7);
    Mat2UI8 m_ptr(m.getDomain(),m.data());
    Mat2UI8 m_copy = m_ptr.copyData();
    m_copy.resize(7,2);
    std::cout<<m_copy<<std::endl;
    return 1;

    Mat2UI8 img;
    img.load(POP_PROJECT_SOURCE_DIR+std::string("/image/iex.png"));//replace this path by those on your computer
    img = PDE::nonLinearAnisotropicDiffusion(img);//filtering
    int value;
    Mat2UI8 threshold = Processing::thresholdOtsuMethod(img,value);//threshold segmentation
    threshold.save("iexthreshold.pgm");
    Mat2RGBUI8 color = Visualization::labelForeground(threshold,img);//Visual validation
    color.display();
    return 0;
}

