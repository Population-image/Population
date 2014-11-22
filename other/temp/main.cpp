#include"Population.h"
#include"treillisnumeric.h"
#include"data/GP/TypelistMacros.h"
#include<iostream>
#include"neuralnetworkmatrix.h"
#include"Cluster.h"
#include"data/notstable/graph/Graph.h"
#include"dependency/VideoFFMPEG.h"
#include"popconfig.h"
using namespace pop;













int main(){

    {

        Mat2UI8 m;
        m.load(POP_PROJECT_SOURCE_DIR+std::string());

        m(0,0)=1; m(0,1)=1;m(1,1)=1;

        m(1,3)=1;m(2,3)=1;

        
        
        //v_display.rbegin()->set_title("title");

        //m.display();
//        std::cout<<pop::Analysis::meanValue(m)<<std::endl;
//        std::cout<<m<<std::endl;
//        std::cout<<Processing::clusterMax(m)<<std::endl;
        return 1;
//        disp.display(video.retrieveMatrixGrey());



        //        pop::GraphAdjencyList<int> g;
        //        g.addVertex();g.vertex(0)=1;
        //        g.addVertex();g.vertex(1)=1;
        //        g.addVertex();g.vertex(2)=1;
        //        g.addVertex();g.vertex(3)=1;
        //        g.addEdge();
        ////        g.addEdge();
        //        g.connection(0,1,3);
        ////        g.connection(1,3,2);
        //        pop::GraphAdjencyList<int>::IteratorEDomain it_g = g.getIteratorEDomain();
        //        pop::GraphAdjencyList<int>::IteratorENeighborhood it_n = g.getIteratorENeighborhood();

        //    //    typename FunctionTypeTraitsSubstituteF<pop::GraphAdjencyList<unsigned>,UI32 >::Result g2(g.getDomain());

        //    //    Processing::clusterMax(g,1);
        //        pop::GraphAdjencyList<UI32> g2 = Processing::clusterToLabel(g);
        //        for(unsigned int i=0;i<g2.sizeVertex();i++){
        //            std::cout<<g2.vertex(i)<<std::endl;

        //        }
        //        return 1;
        //        VecF64 v1(2),v2(2);
        //        v1(0)=0.5;v1(1)=2;
        //        v2(0)=  2;v2(1)=0.4;
        //        std::cout<<pop::productInner(v1,v2)<<std::endl;
        //        return 0;
        //        Mat2F64 f;
        ////        f.multTermByTerm()
        //        TreillisNumeric numeric;
        //        numeric.ocr.setDictionnary(POP_PROJECT_SOURCE_DIR+std::string("/file/neuralnetwork.xml"));
        //        pop::Mat2UI8 m;
        //        m.load("/home/vincent/Downloads/IMG_20141007_094605.jpg");
        //        m = m.opposite();
        //        numeric.treillisNumerique(m,10);
        //        Vec<Vec<Mat2UI8> > number_training =  TrainingNeuralNetwork::loadMNIST( "/home/vincent/train-images.idx3-ubyte","/home/vincent/train-labels.idx1-ubyte");


        //        neuralnetwortest();
        //        return 1;
    }
    {
        std::string str = "/home/vincent/CVSA5_ANV/VID_20141013_142441.mp4";
        VideoVLC video;
        OCRNeuralNetwork ocr;
         ocr.setDictionnary(POP_PROJECT_SOURCE_DIR+std::string("/file/neuralnetwork.xml"));
        video.open(str);
        MatNDisplay disp;
        OCRPlateNumber ocr_recongnition;
        ocr_recongnition.setOCR(&ocr);

        while(video.grabMatrixGrey()){
            //#if Pop_OS==1
            //            std::string dir = "/home/vincent/Dropbox/Vidatis/coutin/Voiture/";
            //#else
            //            std::string dir = "C:/Users/tariel/Dropbox/Vidatis/coutin/voiture/";
            //#endif

            //            std::vector<std::string> v_dir = BasicUtility::getFilesInDirectory(dir);

            //            for(unsigned int i= 0;i<v_dir.size();i++){
            //                std::string file = dir+"/"+ v_dir[i];
            //                std::cout<<file<<std::endl;
            //                if(pop::BasicUtility::isFile(file)){
            Mat2UI8 m;
            m = video.retrieveMatrixGrey();
            int scale = 8;
            Mat2UI32 label = ocr_recongnition.segmentation(m,scale);



            //                Visualization::labelToRandomRGB(label).display();
            ocr_recongnition.recognition(m,label,scale);

//            std::cout<<"treillis : "<<(double) (end_global - start_global) / CLOCKS_PER_SEC<<std::endl;
        }



        //            }
        //        }
        //        DistanceCharacteristicHeigh dist_height;
        //        DistanceCharacteristicWidth dist_width;
        //        DistanceCharacteristicWidthInterval dist_interval_width;
        //        DistanceCharacteristicHeightInterval dist_interval_height;
        //        DistanceCharacteristicGreyLevel dist_grey_level;



        //        DistributionExpression d_heigh("(8*x)^2");
        //        DistributionExpression d_width("(x)^2");
        //        DistributionExpression d_interval_width("(0.5*x)^2");
        //        DistributionExpression d_interval_height("(4*x)^2");
        //        DistributionExpression d_grey_level("(4*x)^2");
        //CharacteristicClusterMix







        //        std::cout<<"process : "<<(double) (end_global - start_global) / CLOCKS_PER_SEC<<std::endl;
        //                   Visualization::labelToRandomRGB(label).display();
        //numeric.treillisNumeriqueVersionText(label,10);
        return 1;
    }
    //    neuralnetwortest();
    {
        //                Mat2UI8 m;
        ////               m.load("/home/vincent/Population/doc/html/iex.png");
        ////                m.load("/home/vincent/Desktop/plaque/Tchad/P1000297-small.JPG");
        //              m.load("/home/vincent/Desktop/_.jpg");
        //            m = Processing::median(m,2);
        //            m.display();
        //                Mat2F64 grad= Processing::gradientMagnitudeGaussian(Mat2F64(m),1);
        //                Mat2UI32 label = Processing::minimaLocalMap(grad,2);
        //                label = Processing::dilation(label,2);
        //                 label =regionGrowingAdamsBischofMeanOverStandardDeviation33(label,m,40);
        //                Visualization::labelToRandomRGB(label).display();
        //                return 1;
    }
    //    {
    //        Mat2UI8 m;
    //        m.load("/home/vincent/Desktop/_.png");
    //        CharacteristicClusterMix mix;
    //        mix.setMatrix(m);
    //        Vec2I32 x(0,0);
    //        mix.addPoint(x);
    //        x=Vec2I32(19,1);
    //        mix.addPoint(x);
    ////        x=Vec2I32(20,3);
    ////        mix.addPoint(x);

    //        CharacteristicClusterMix mix2;
    //        mix2.setMatrix(m);
    //        x=Vec2I32(0,4);
    //        mix2.addPoint(x);
    //        x=Vec2I32(30,7);
    //        mix2.addPoint(x);



    //        CharacteristicClusterDistanceHeight dist_height;
    //        CharacteristicClusterDistanceWidth dist_width;
    //        CharacteristicClusterDistanceWidthInterval dist_interval_width;
    //        CharacteristicClusterDistanceHeightInterval dist_interval_height;


    //        Distribution dist1("(8*x)^2");
    //        Distribution("(x)^2"));
    //        (Distribution("(0.5*x)^2"));
    //        (Distribution("(2*x)^2"));

    ////        x=Vec2I32(2,10);
    ////        mix2.addPoint(x);

    //        DistanceCharacteristicGreyLevel<CharacteristicClusterMix> dist1;
    //        DistanceCharacteristicHeigh<CharacteristicClusterMix> dist2;
    //        Distribution d1("0");
    //        Distribution d2("(3*x)^2");


    //        DistanceSumCharacteristic<CharacteristicClusterMix> dist_sum;
    //        dist_sum.addDistance(d1,&dist1);
    //        dist_sum.addDistance(d2,&dist2);
    //        std::cout<<dist_sum.operator ()(mix,mix2)<<std::endl;
    //        std::cout<<dist2.operator ()(mix,mix2)<<std::endl;


    //        return 1;

    //        //        TreillisNumeric numeric;
    //        //        numeric.ocr.setDictionnary(POP_PROJECT_SOURCE_DIR+std::string("/file/neuralnetwork.xml"));
    //        //        Mat2UI8 m;
    //        //        m.load("/home/vincent/Desktop/_.png");
    //        ////        m = m.opposite();
    //        //        clock_t start_global, end_global;
    //        //        start_global = clock();
    //        //        numeric.treillisNumeriqueVersionText(m,2);
    //        //        end_global = clock();
    //        //        std::cout<<"process : "<<(double) (end_global - start_global) / CLOCKS_PER_SEC<<std::endl;
    //        //        getchar();
    //    }
    MatNDisplay disp;
    try{
        TreillisNumeric numeric;
        numeric.ocr.setDictionnary(POP_PROJECT_SOURCE_DIR+std::string("/file/neuralnetwork.xml"));
        //std::string dir = "/home/vincent/Dropbox/Vidatis/SingleEvent/";
        std::string dir = "/home/vincent/Dropbox/Vidatis/coutin";
        //        std::string dir = "C:/Users/tariel/Dropbox/Vidatis/coutin/";

        //std::string dir = "/home/vincent/Dropbox/Vidatis/coutin/voiture/";
        std::vector<std::string> v_dir = BasicUtility::getFilesInDirectory(dir);

        for(unsigned int i= 0;i<v_dir.size();i++){
            std::string dir2 = dir+"/"+ v_dir[i];
            std::vector<std::string> v_dir2 = BasicUtility::getFilesInDirectory(dir2);

            for(unsigned int i= 0;i<v_dir2.size();i++){
                std::string file = dir2+"/"+ v_dir2[i];


                //                std::cout<<file<<std::endl;

                if(BasicUtility::isFile(file)){
                    clock_t start_global, end_global;
                    //        for(unsigned int i= 0;i<v_dir.size();i++){
                    //            std::string dir2 = dir + v_dir[i];
                    //            std::vector<std::string> v_file = BasicUtility::getFilesInDirectory(dir2);
                    //            for(unsigned int j= 0;j<v_file.size();j++){
                    //                std::string file = dir2+"/"+ v_file[j];
                    Mat2UI8 m;
                    m.load(file.c_str());
                    disp.display(m);
                    double scale=0.5;
                    double factor = 15;
                    m= GeometricalTransformation::scale(m,Vec2F64(scale,scale));

                    start_global = clock();
                    Mat2F64 grad= Processing::gradientMagnitudeGaussian(Mat2F64(m),1);
                    Mat2UI32 label = Processing::minimaLocalMap(grad,2);
                    label = Processing::dilation(label,1,0);
                    end_global = clock();
                    std::cout<<"process : "<<(double) (end_global - start_global) / CLOCKS_PER_SEC<<std::endl;
                    std::cout<<"region"<<std::endl;
                    start_global = clock();
                    label =Processing::regionGrowingMergingLevel(label,m,20);
                    end_global = clock();
                    std::cout<<"process : "<<(double) (end_global - start_global) / CLOCKS_PER_SEC<<std::endl;
                    //                    Visualization::labelToRandomRGB(label).display();
                    numeric.treillisNumeriqueVersionText(label,4);
                    //                    Visualization::labelAverageRGB(label,m).display();
                    //                    return 1;


                    //                    clock_t start_global, end_global;
                    //                    start_global = clock();
                    //                    Mat2UI8 grad = Processing::gradientMagnitudeSobel(m);

                    //                    grad = Processing::dynamic(grad,20);
                    //                    end_global = clock();
                    //                    std::cout<<"grad : "<<(double) (end_global - start_global) / CLOCKS_PER_SEC<<std::endl;
                    //                    //                    grad.display();
                    //                    Mat2UI32 label = Processing::minimaRegional(grad);

                    //                    //                    Visualization::labelForeground(label,m,0).display();


                    //                    // m = GeometricalTransformation::rotate(m,0.4);
                    //                    //                    double scale=1;
                    //                    //                    double factor = 15;
                    //                    //                    m= GeometricalTransformation::scale(m,Vec2F64(scale,scale));
                    //                    std::cout<<m.getDomain()<<std::endl;
                    //                    m = m.opposite();

                    //                    start_global = clock();
                    //                    numeric.treillisNumerique(m,factor*scale);
                    //                    end_global = clock();
                    //                    std::cout<<"process : "<<(double) (end_global - start_global) / CLOCKS_PER_SEC<<std::endl;

                }
            }
        }
    }
    catch(const pexception &e){
        e.display();//Display the error in a window
    }
    return 1;
}
