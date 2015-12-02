#include"Population.h"
using namespace pop;//Population namespace

int main(){
    Mat2UI8 m(5,4);
    m.fill(0);
    m(1,1)=20;
    Mat2UI8 m_ptr(m.getDomain(),m.data());
    Mat2UI8 m_col = m.selectColumn(1);
    Mat2UI8 m_row = m.selectRow(1);
    std::cout<<m_col<<std::endl<<std::endl;;
    std::cout<<m_row<<std::endl;
    std::cout<<m_ptr<<std::endl;
    return 1;

    OCRNeuralNetwork net;
    net.setDictionnary("neuralnetwork.xml");
    Mat2UI8 img(5,5);
    img.load("_3.png");//replace this path by those on your computer
    char c = net.parseMatrix(img);
    std::cout<<c<<std::endl;
    return 1;

    img.load(POP_PROJECT_SOURCE_DIR+std::string("/image/iex.png"));//replace this path by those on your computer

    img = PDE::nonLinearAnisotropicDiffusion(img);//filtering
    int value;
    Mat2UI8 threshold = Processing::thresholdOtsuMethod(img,value);//threshold segmentation
    threshold.save("iexthreshold.pgm");
    Mat2RGBUI8 color = Visualization::labelForeground(threshold,img);//Visual validation
    color.display();
    return 0;
}

