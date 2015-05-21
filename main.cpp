#include"Population.h"//Single header
using namespace pop;//Population namespace
#if __cplusplus > 199711L // c++11
#include <chrono>
#endif
#include<map>

//UI32 minValue(const std::vector<UI32> & v_values, int index_in){
//    if(v_values[index_in]==0){
//        return index_in;
//    }else{
//        index_buttom = v_values[index_in];
//        return minValue(v_values,index_buttom);
//    }

//}

template<typename PixelType>
VecF32 normalizedImageToNeuralNet( const MatN<2,PixelType>& f,Vec2I32 domain ,MatNInterpolation interpolation=MATN_INTERPOLATION_BILINEAR) {
    //    F32 mean     = Analysis::meanValue(f);
    //    F32 standart = Analysis::standardDeviationValue(f);

    //    VecF32 v_in(domain.multCoordinate());
    //    int k=0;

    //    for(unsigned int i=0;i<domain(0);i++){
    //        for(unsigned int j=0;j<domain(1);j++,k++){

    //            VecN<DIM,F32> x( (i-0.5)*alpha(0),(j-0.5)*alpha(1));
    //            if(interpolation.isValid(f.getDomain(),x)){
    //                v_in[k] = (interpolation.apply(f,x)-mean)/standart;
    //            }else{
    //                std::cerr<<"errror normalized"<<std::endl;
    //            }
    //        }
    //    }
    //    return v_in;
}
int main()
{
    {
        Mat2UI8 m(5,5);//
        //        Mat2UI8::IteratorEOrder it_order=m.getIteratorEOrder(1,1);
        //        while(it_order.next()){
        //            std::cout<<it_order.x()<<std::endl;
        //        }
        //        return 1;
        //        Mat2UI8::IteratorEDomain it = m.getIteratorEDomain();
        //        while(it.next()){
        //            std::cout<<it.x()<<std::endl;
        //        }
        //        return 0;

        //    m(0,0)=1;m(0,1)=1;m(0,2)=1;
        //    m(1,0)=1;m(1,1)=1;m(1,2)=1;
        //    m(2,0)=1;m(2,1)=1;m(2,2)=1;


        m(4,0)=1;m(4,1)=1;m(4,2)=1;m(4,3)=1;m(4,4)=1;

        m.load((std::string(POP_PROJECT_SOURCE_DIR)+"/image/Lena.bmp").c_str());
        m = Processing::threshold(m,120);
#if __cplusplus > 199711L // c++11
        auto start_global = std::chrono::high_resolution_clock::now();
#else
        unsigned int start_global = time(NULL);
#endif
        ProcessingAdvanced::clusterToLabel(m,m.getIteratorENeighborhood(1,1),m.getIteratorEOrder(0,1));
        //        clusterToLabel(m,m.getIteratorENeighborhood(1,1),m.getIteratorEDomain());
#if __cplusplus > 199711L // c++11
        auto end_global= std::chrono::high_resolution_clock::now();
        std::cout<<"processing : "<<std::chrono::duration<double, std::milli>(end_global-start_global).count()<<std::endl;
#else
        unsigned int end_global = time(NULL);
        std::cout << "processing: " << (start_global-end_global) << "s" << std::endl;
#endif

#if __cplusplus > 199711L // c++11
        start_global= std::chrono::high_resolution_clock::now();
#else
        start_global = time(NULL);
#endif
        Processing::clusterToLabel(m);
#if __cplusplus > 199711L // c++11
        auto end_global= std::chrono::high_resolution_clock::now();
        std::cout<<"processing : "<<std::chrono::duration<double, std::milli>(end_global-start_global).count()<<std::endl;
#else
        start_global = time(NULL);
        std::cout << "processing: " << (start_global-end_global) << "s" << std::endl;
#endif


        //        Visualization::labelToRandomRGB().display();
        //    m = Processing::threshold(m,120);
        //    m.display();
        //    std::cout<<clusterToLabel(m)<<std::endl;
        //    Visualization::labelToRandomRGB(clusterToLabel(m)).display();
        //    Processing::clusterToLabel(m);
        return 1;
    }
    std::string plaque = "/home/vincent/Desktop/plate.jpeg";
    std::string plaque_mask = "/home/vincent/Desktop/plate_mask.jpeg";
    Mat2RGBUI8 plate;
    plate.load(plaque);
    Mat2UI8 plate_mask;
    plate_mask.load(plaque);


    Vec2I32 domain(80,200);




    NeuralNet net;
    int size_i=domain(0);
    int size_j=domain(1);
    int nbr_map=3;
    net.addLayerMatrixInput(size_i,size_j,nbr_map);
    net.addLayerMatrixConvolutionSubScaling(20,2,2);
    net.addLayerMatrixConvolutionSubScaling(30,2,2);
    net.addLayerMatrixConvolutionSubScaling(40,2,2);


    VecF32 v_in(size_i*size_j*nbr_map);

    Mat2UI8 plate_r,plate_g,plate_b;
    Convertor::toRGB(plate,plate_r,plate_g,plate_b);
    VecF32 v_r,v_g,v_b;

    v_r = normalizedImageToNeuralNet(plate_r,domain);




    //    DistributionNormal d(0,1);
    //    for(unsigned int i=0;i<size_i*size_j*nbr_map;i++){
    //        v_in(i)=d.randomVariable();
    //    }

    VecF32 v_out;
#if __cplusplus > 199711L // c++11
    auto start_global = std::chrono::high_resolution_clock::now();
#else
    unsigned int start_global = time(NULL);
#endif
    for(unsigned int i=0;i<100;i++){
        net.forwardCPU(v_in,v_out);
    }
#if __cplusplus > 199711L // c++11
    auto end_global= std::chrono::high_resolution_clock::now();
    std::cout<<"processing : "<<std::chrono::duration<double, std::milli>(end_global-start_global).count()<<std::endl;
#else
    unsigned int end_global = time(NULL);
    std::cout << "processing: " << (start_global-end_global) << "s" << std::endl;
#endif
    return 1;

    //        clusterToLabel(m,m.getIteratorENeighborhood(1,1),m.getIteratorEDomain());


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
