#include<iostream>
#include"../../include/Population.h"
#include<chrono>

int main(int argc, char* argv[]) {
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
    pop::Mat2UI8 threshold1 = pop::Processing::thresholdAdaptativeSmoothDeriche(m,alpha,-10+ adpatative_filter_shift);
#endif

#if (METHOD == 1)
    // tensor
    pop::Mat2UI8 threshold = ProcessingTensor::thresoldAdaptiveSmoothDeriche(m, alpha, -10 + adpatative_filter_shift);
#endif

    end = std::chrono::high_resolution_clock::now();
    std::cout << __FILE__ << "::" << __LINE__ << "finish after " << std::chrono::duration<double, std::milli>(end-start).count() << std::endl;

    start = std::chrono::high_resolution_clock::now();
#if (METHOD == 0)
    // smoothDeriche fast
    pop::Mat2UI8 threshold2 = pop::Processing::thresholdAdaptativeSmoothDericheFast(m,alpha,-10+ adpatative_filter_shift);
#endif

#if (METHOD == 1)
    // tensor
    pop::Mat2UI8 threshold = ProcessingTensor::thresoldAdaptiveSmoothDeriche(m, alpha, -10 + adpatative_filter_shift);
#endif

    end = std::chrono::high_resolution_clock::now();
    std::cout << __FILE__ << "::" << __LINE__ << "finish after " << std::chrono::duration<double, std::milli>(end-start).count() << std::endl;

    start = std::chrono::high_resolution_clock::now();
#if (METHOD == 0)
    // smoothFast
    pop::Mat2UI8 threshold3 = pop::Processing::thresholdAdaptativeSmoothFast(m,alpha,-10+ adpatative_filter_shift);
#endif

#if (METHOD == 1)
    // tensor
    pop::Mat2UI8 threshold = ProcessingTensor::thresoldAdaptiveSmoothDeriche(m, alpha, -10 + adpatative_filter_shift);
#endif

    end = std::chrono::high_resolution_clock::now();
    std::cout << __FILE__ << "::" << __LINE__ << "finish after " << std::chrono::duration<double, std::milli>(end-start).count() << std::endl;

    threshold1.display();
    threshold2.display();
    threshold3.display();
    return 0;
}
