#include"Population.h"//Single header
using namespace pop;//Population namespace
#include"chrono"
int main()
{
    NeuralNet net;
    int size_i=200;
    int size_j=200;
    int nbr_map=3;
    net.addLayerMatrixInput(size_i,size_j,nbr_map);
    net.addLayerMatrixConvolutionSubScaling(20,2,2);
    net.addLayerMatrixConvolutionSubScaling(30,2,2);
    net.addLayerMatrixConvolutionSubScaling(40,2,2);


    VecF32 v_in(size_i*size_j*nbr_map);
    DistributionNormal d(0,1);
    for(unsigned int i=0;i<size_i*size_j*nbr_map;i++){
        v_in(i)=d.randomVariable();
    }

    VecF32 v_out;
    auto start_global= std::chrono::high_resolution_clock::now();
    for(unsigned int i=0;i<100;i++){
        net.forwardCPU(v_in,v_out);
    }
    auto end_global= std::chrono::high_resolution_clock::now();
    std::cout<<"processing : "<<std::chrono::duration<double, std::milli>(end_global-start_global).count()<<std::endl;
    return 1;


    //    omp_set_num_threads(6);
    //    pop::Mat2UI8 m(1200,1600);
    //    //auto start_global = std::chrono::high_resolution_clock::now();

    //    m=thresholdNiblackMethod(m);
    //    //auto end_global = std::chrono::high_resolution_clock::now();
    //    std::cout<<"processing nimblack1 : "<<std::chrono::duration<double, std::milli>(end_global-start_global).count()<<std::endl;

    //    int time1 = time(NULL);
    //     //start_global = std::chrono::high_resolution_clock::now();

    //    m=Processing::thresholdNiblackMethod(m);
    //     //end_global = std::chrono::high_resolution_clock::now();
    //    std::cout<<"processing nimblack2 : "<<std::chrono::duration<double, std::milli>(end_global-start_global).count()<<std::endl;

    //    return 1;

    //    //	m = m*m;
    //    int time2 = time(NULL);
    //    std::cout<<time2-time1<<std::endl;
    //    Mat2UI8 img;//2d grey-level image object
    //    img.load(POP_PROJECT_SOURCE_DIR+std::string("/image/iex.png"));//replace this path by those on your computer
    //    img = PDE::nonLinearAnisotropicDiffusion(img);//filtering
    //    int value;
    //    Mat2UI8 threshold = Processing::thresholdOtsuMethod(img,value);//threshold segmentation
    //    threshold.save("iexthreshold.pgm");
    //    Mat2RGBUI8 color = Visualization::labelForeground(threshold,img);//Visual validation
    //    color.display();
    return 0;
}
