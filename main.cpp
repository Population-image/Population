#include"Population.h"
#include"algorithm/processingtensor.h"
using namespace pop;//Population namespace
int main(){
//    Mat2UI8 img;//2d grey-level image object
//    img.load(POP_PROJECT_SOURCE_DIR+std::string("/image/iex.png"));//replace this path by those on your computer

//    img = Processing::thresholdAdaptativeSmoothFast(img,0.01);
//    img.display();
//    img = PDE::nonLinearAnisotropicDiffusion(img);//filtering
//    int value;
//    Mat2UI8 threshold = Processing::thresholdOtsuMethod(img,value);//threshold segmentation
//    threshold.save("iexthreshold.pgm");
//    Mat2RGBUI8 color = Visualization::labelForeground(threshold,img);//Visual validation
//    color.display();

    // test de Khanh
    // 0 : Population    1 : Tensor
#define METHOD 0

    pop::Mat2RGBUI8 img1;
    img1.load("/home/dokhanh/workspace/DEV/LAPI-API/build/export/examples/samples/DG356EB.jpg");
    int adpatative_filter_shift = 0;
    double alpha = 0.25;
    img1.display();
    char* img_char = reinterpret_cast<char*>(img1.data());
    pop::Mat2UI8 m;
    pop::Mat2RGBUI8 color(pop::Vec2I32(img1.sizeI(), img1.sizeJ()), reinterpret_cast<pop::RGBUI8*>(img_char));
    m.resize(color.getDomain());
    for(int i =0;i<color.size();i++){
        m(i)=(0.299f*color(i).r()+0.587f*color(i).g())/(0.299f+0.587f);
    }
    m = m.opposite();

    std::chrono::time_point<std::chrono::system_clock> start, end, start1, end1;
    start = std::chrono::high_resolution_clock::now();

#if (METHOD == 0)
    // population
    //Mat2UI8 threshold = Processing::thresholdAdaptativeSmoothDeriche(m,alpha,-10+ adpatative_filter_shift);
    Mat2UI8 threshold = Processing::thresholdAdaptativeSmoothFast(m,alpha,-10+ adpatative_filter_shift);
#endif

#if (METHOD == 1)
    // tensor
    Mat2UI8 threshold = ProcessingTensor::thresoldAdaptiveSmoothDeriche(m, alpha, -10 + adpatative_filter_shift);
#endif

    end = std::chrono::high_resolution_clock::now();
    std::cout << __FILE__ << "::" << __LINE__ << "finish after " << std::chrono::duration<double, std::milli>(end-start).count() << std::endl;

    threshold.display();
    return 0;
}


