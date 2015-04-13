#include"Population.h"
#include<iostream>
#include"neuralnetworkmatrix.h"
#include"data/notstable/graph/Graph.h"
#include"popconfig.h"
using namespace pop;

struct Line{

    static Vec<Vec2F32> pointInLine(Vec2F32 xmin,Vec2F32 xmax, F32 step){
        Vec2F32 direction = xmax-xmin;
        F32 distance = direction.norm(2);
        direction/=distance;
        Vec2F32 x=xmin;
        Vec<Vec2F32> v_points;
        for(unsigned int i = 0;i<distance/step;i++){
            v_points.push_back(x);
            x +=direction;
        }
        return v_points;
    }
    template<typename PixelType>
    static F32 mean(const Vec<Vec2F32>& line,const MatN<2,PixelType> & m  ){
        F32 sum = 0;
        for(unsigned int i=0;i<line.size();i++){
            if(m.isValid(line(i))){
                sum+=m.interpolationBilinear(line(i));
            }
        }
        return sum/line.size();
    }
    template<typename PixelType>
    static Vec2F32  barycentre(const Vec<Vec2F32>& line,const MatN<2,PixelType> & m  ){
        Vec2F32 barycentre(0,0);
        F32 sum = 0;
        for(unsigned int i=0;i<line.size();i++){
            if(m.isValid(line(i))){
                sum+=m.interpolationBilinear(line(i));
                barycentre += (sum*line(i));
            }
        }
        return barycentre/sum;
    }



};



int main(){
    {
        Vec<LinearLeastSquareRANSACModel::Data> data;
        VecF32 x(2);F32 y;
        x(0)=1;x(1)=1;y=5.2;data.push_back(LinearLeastSquareRANSACModel::Data(x,y));
        x(0)=1;x(1)=2;y=5.8;data.push_back(LinearLeastSquareRANSACModel::Data(x,y));
        x(0)=1;x(1)=3;y=6.8;data.push_back(LinearLeastSquareRANSACModel::Data(x,y));
        x(0)=1;x(1)=4;y=8.2;data.push_back(LinearLeastSquareRANSACModel::Data(x,y));
        x(0)=1;x(1)=5;y=13; data.push_back(LinearLeastSquareRANSACModel::Data(x,y));

        LinearLeastSquareRANSACModel m;
        Vec<LinearLeastSquareRANSACModel::Data> dataconsencus;
        ransac(data,10,1,2,m,dataconsencus);
        std::cout<<m.getBeta()<<std::endl;
        std::cout<<m.getError()<<std::endl;
        std::cout<<dataconsencus<<std::endl;
        return 1;
    }
    //    {
    //        Mat2UI8 m;
    //        m.load(POP_PROJECT_SOURCE_DIR+std::string("/image/barriere.png"));
    //        Mat2UI8 edge = Processing::edgeDetectorCanny(m,2,0.5,5);
    //        edge.display("edge",false);
    //        Mat2F32 hough = Feature::transformHough(edge);
    //        hough.display("hough",false);
    //        std::vector< std::pair<Vec2F32, Vec2F32 > > v_lines = Feature::HoughToLines(hough,edge ,0.5);
    //        Mat2RGBUI8 m_hough(m);
    //        for(unsigned int i=0;i<v_lines.size();i++){
    //            Draw::line(m_hough,v_lines[i].first,v_lines[i].second,  RGBUI8(255,0,0),2);
    //        }
    //        m_hough.display();
    //    }
    Mat2UI8 m_init;
     m_init.load("/home/vincent/Desktop/leslie.jpg");
//    m_init.load("/home/vincent/Desktop/coutin.jpg");

    F32 scalefactor =400./m_init.sizeJ();
    pop::Vec2F32 border_extra=pop::Vec2F32(0.01f,0.1f);
    Mat2UI8 m = GeometricalTransformation::scale(m_init,Vec2F32(scalefactor,scalefactor));
    std::cout<<m.getDomain()<<std::endl;

    Mat2UI8 tophat_init =   Processing::closing(m,2)-m;
    tophat_init.display("top hat",false);
    Mat2UI8 elt(3,3);
    elt(1,0)=1;
    elt(1,1)=1;
    elt(1,2)=1;

    Mat2UI8 tophat = Processing::closingStructuralElement(tophat_init,elt  ,8);
    tophat = Processing::openingStructuralElement(tophat,elt  ,2  );
    elt = elt.transpose();
    tophat = Processing::openingStructuralElement(tophat,elt  ,2 );
    tophat.display("filter",false);

    int value;
    Mat2UI8 binary = Processing::thresholdOtsuMethod(tophat,value);
    Mat2UI8 binary2 =  Processing::threshold(tophat,20);
    binary = pop::minimum(binary,binary2);
    binary.display("binary",false);



    Mat2UI32 imglabel =Processing::clusterToLabel(binary);
    Visualization::labelToRandomRGB(imglabel).display("label",false);
    pop::Vec<pop::Vec2I32> v_xmin;
    pop::Vec<pop::Vec2I32> v_xmax;
    pop::Vec<Mat2UI8> v_img = Analysis::labelToMatrices(imglabel,v_xmin,v_xmax);



    for(unsigned int i =0;i<v_img.size();i++){
        Vec2I32 domain = v_img[i].getDomain();
        if(domain(1)>0.5*binary.sizeJ()){
            Vec2F32 x_min_i = Vec2F32(v_xmin[i])/scalefactor;
            Vec2F32 x_max_i = Vec2F32(v_xmax[i])/scalefactor;

            int height = x_max_i(0)- x_min_i(0);
            x_min_i(0)-= height/2;
            x_max_i(0)+= height/2;
            F32 angle_rot_radian;
            Processing::rotateAtHorizontal(m_init(x_min_i,x_max_i),angle_rot_radian).display("label",true,false);

        }
    }



    return 0;


    //    if(display)temp =Draw::mergeTwoMatrixVertical(temp,tophat);
    //    Mat2UI8 elt(3,3);
    //    elt(1,0)=1;
    //    elt(1,1)=1;
    //    elt(1,2)=1;

    //    tophat = Processing::closingStructuralElement(tophat,elt  ,std::ceil(pixel_width_letter));
    //    if(display)temp =Draw::mergeTwoMatrixVertical(temp,tophat);
    //    tophat = Processing::openingStructuralElement(tophat,elt  ,std::ceil(pixel_width_letter)/4  );
    //    elt = elt.transpose();
    //    tophat = Processing::openingStructuralElement(tophat,elt  ,std::ceil(pixel_width_letter)/factor_opening_vertical );
    //    if(display)temp =Draw::mergeTwoMatrixVertical(temp,tophat);
    //    int value;
    //    Mat2UI8 binary = Processing::thresholdOtsuMethod(tophat,value);
    //    Mat2UI8 binary2 =  Processing::threshold(tophat,10);



    neuralnetwortest2();
    return 1;
}
