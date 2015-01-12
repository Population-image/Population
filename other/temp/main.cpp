#include"Population.h"
#include"treillisnumeric.h"
#include<iostream>
#include"neuralnetworkmatrix.h"
#include"Cluster.h"
#include"data/notstable/graph/Graph.h"
#include"popconfig.h"
using namespace pop;

int main(){
    MatNDisplay disp;
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
    return 1;
}
