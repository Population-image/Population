#include"Population.h"
using namespace pop;//Population namespace
int main(){
//    Mat2UI8 img;//2d grey-level image object
//    img.load(POP_PROJECT_SOURCE_DIR+std::string("/image/iex.png"));//replace this path by those on your computer
//    img = PDE::nonLinearAnisotropicDiffusion(img);//filtering
//    int value;
//    Mat2UI8 threshold = Processing::thresholdOtsuMethod(img,value);//threshold segmentation
//    threshold.save("iexthreshold.pgm");
//    Mat2RGBUI8 color = Visualization::labelForeground(threshold,img);//Visual validation
//    color.display();
//    return 0;

    pop::Mat2RGBUI8 img1;
    img1.load("/home/dokhanh/workspace/DEV/LAPI-API/build/export/examples/samples/DG356EB.jpg");
    double ratio = 0.15;
    //double thickness_letter_in_pixel = ratio*img1.sizeJ()*0.022;
    //thickness_letter_in_pixel =(std::min)(40.,(std::max)(3.,thickness_letter_in_pixel));
    int adpatative_filter_shift = 0;
    //int alpha = 1.f/thickness_letter_in_pixel;
    double alpha = 0.25;
    img1.display();
    char* img_char = reinterpret_cast<char*>(img1.data());
    pop::Mat2UI8 m;
    pop::Mat2RGBUI8 color(pop::Vec2I32(img1.sizeI(), img1.sizeJ()), reinterpret_cast<pop::RGBUI8*>(img_char));
    //color.display();
    m.resize(color.getDomain());
    for(int i =0;i<color.size();i++){
        m(i)=(0.299f*color(i).r()+0.587f*color(i).g())/(0.299f+0.587f);
    }
    //m.display();
    m = m.opposite();
    Mat2UI8 threshold = Processing::thresholdAdaptativeDeriche(m,alpha,-10+ adpatative_filter_shift);
    threshold.display();
    return 0;
}


