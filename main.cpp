#include <stdio.h>
#include <iostream>
#include"time.h"

#include"Population.h"//Single header
#include"data/notstable/Classifer.h"
using namespace pop;//Population namespace
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"


struct Coordinate
{
    Vec2F32 _x;
    typedef Vec2F32  Feature;
    F32 operator ()(const Vec2F32 & x)const{
        return (_x-x).norm(2);
    }
};


int main(){


    ClassiferThreshold<Coordinate> c;
    VecF32 v_affec(4);
    VecF32 v_coeff(4);
    VecF32 v_weigh(4);
    v_affec(0)=0;v_coeff(0)=0.3;
    v_affec(1)=1;v_coeff(1)=2;
    v_affec(2)=0;v_coeff(2)=-0.5;
    v_affec(3)=1;v_coeff(3)=-4;
    c.setTraining(v_coeff,v_affec);
    v_weigh(0)=0.2;v_weigh(1)=0.2;v_weigh(2)=0.2;v_weigh(3)=0.2;
    c.setWeight(v_weigh);
    c.training();
    std::cout<<c._error<<std::endl;
    std::cout<<c._threshold<<std::endl;
    std::cout<<c._sign<<std::endl;




    return 1;

    Mat2UI8 img1,img2;

    img2.load("/home/vincent/Desktop/Photo_CNI/cn2.png");



    img1.load("/home/vincent/Desktop/Photo_CNI/cn3.png");
    int number_match_point =20;
    int min_overlap = 5;

    typedef KeyPointPyramid<2> KeyPointAlgo;


    Pyramid<2,F32> pyramid_gaussian = Feature::pyramidGaussian(img1);
    Vec<KeyPointPyramid<2> > keypoint1   = Feature::keyPointSIFT(pyramid_gaussian);
//    Feature::drawKeyPointsCircle(img1,keypoint1).display("keypoint");

    Vec<Descriptor<KeyPointAlgo > >descriptor1 = Feature::descriptorPieChart(img1,keypoint1);
//    Feature::drawDescriptorArrow(img1,descriptor1).display("descriptor");

    Pyramid<2,F32> pyramid2 = Feature::pyramidGaussian(img2);
    Vec<KeyPointAlgo > keypoint2 = Feature::keyPointSIFT(pyramid2);

    Feature::drawKeyPointsCircle(img2,keypoint2).display("keypoint2",false);
    Vec<Descriptor<KeyPointAlgo > >descriptor2 = Feature::descriptorPieChart(img2,keypoint2);
    Feature::drawDescriptorArrow(img2,descriptor2).display("descriptor",false);
    Vec<DescriptorMatch<Descriptor<KeyPointAlgo > > > match = Feature::descriptorMatchBruteForce(descriptor1,descriptor2);
    if(number_match_point<match.size())
        match.erase(match.begin()+number_match_point,match.end());
    std::cout<<match.size()<<std::endl;
    match = Feature::descriptorFilterNoOverlap(match,min_overlap);
    Feature::drawDescriptorMatch(img1,img2,match).display();
    return 1;
    //        Mat2UI8 img;
    //        img.load(POP_PROJECT_SOURCE_DIR+std::string("/image/iex.png"));//replace this path by those on your computer
    //        img.display("Initial image",false);
    //        img = PDE::nonLinearAnisotropicDiffusion(img);//filtering
    //        img.display();
    //        F32 value;
    //        Mat2UI8 threshold = Processing::thresholdOtsuMethod(img,value);//threshold segmentation
    //        threshold.save("iexthreshold.png");
    //        Mat2RGBUI8 color = Visualization::labelForeground(threshold,img);//Visual validation
    //        color.display("Segmented image",true);
    return 0;
}
