#include <stdio.h>
#include <iostream>
#include"time.h"

#include"Population.h"//Single header
using namespace pop;//Population namespace
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
template<typename Descriptor>
Vec<DescriptorMatch<Descriptor >   > descriptorMatch2(const Vec<Descriptor > & descriptor1, const Vec<Descriptor > & descriptor2){


    Vec<DescriptorMatch<Descriptor > > v_match;
    for(unsigned int j=0;j<descriptor1.size();j++){
        double distance =1000;
        int index=-1;
        for(unsigned int i=0;i<descriptor2.size();i++){
            const Descriptor& d1=descriptor1[j];
            const Descriptor& d2=descriptor2[i];
            double dist_temp = pop::distance(d1.data(),d2.data(),2);
            if(dist_temp<distance){
                index = i;
                distance =dist_temp;
            }
        }
        std::cout<<distance<<std::endl;
        DescriptorMatch<Descriptor > match;
        match._d1 = descriptor1[j];
        match._d2 = descriptor2[index];
        match._error= distance;
        v_match.push_back(match);
    }
    std::sort(v_match.begin(),v_match.end());
    return v_match;
}

int main(){
    {



        Mat2F64 m(15000,15000);
        VecF64 kernel(30);
        int time1= time(NULL);
        Mat2F64::IteratorEDomain it2 =  m.getIteratorEDomain();
       Processing::smoothGaussian(m,3);
        int time2= time(NULL);
        std::cout<<time2-time1<<std::endl;

        cv::Mat m_cv(15000,15000,CV_64FC1);
        cv::Mat dest(15000,15000,CV_64FC1);
        time1= time(NULL);
        cv::GaussianBlur(m_cv,dest,cv::Size(),3,3);
        time2= time(NULL);
                std::cout<<time2-time1<<std::endl;
        return 1;
    }

    Mat2UI8 img1(7,4),img2;


//    img1.load("/home/vincent/Desktop/Photo_CNI/cn3.png");
    img1.load("/home/vincent/Desktop/_.jpg");

    GeometricalTransformation::subResolution(img1,6).display();
//    GeometricalTransformation::scale(img1,Vec2F64(2,2),MATN_INTERPOLATION_BILINEAR);
    img2.load("/home/vincent/Desktop/Photo_CNI/cn2.png");
    int number_match_point =20;
    int min_overlap = 5;

    typedef KeyPointPyramid<2> KeyPointAlgo;
    img1 = GeometricalTransformation::scale(img1,Vec2F64(2,2),MATN_INTERPOLATION_BILINEAR);

    Pyramid<2,F64> pyramid_gaussian = Feature::pyramidGaussian(img1,1.6,1);
    Vec<KeyPointPyramid<2> > keypoint1   = Feature::keyPointSIFT(pyramid_gaussian);
    Feature::drawKeyPointsCircle(img1,keypoint1).display("keypoint");

    Vec<Descriptor<KeyPointAlgo > >descriptor1 = Feature::descriptorPieChart(img1,keypoint1);
    Feature::drawDescriptorArrow(img1,descriptor1).display("descriptor");

    Pyramid<2,F64> pyramid2 = Feature::pyramidGaussian(img2);
    Vec<KeyPointAlgo > keypoint2 = Feature::keyPointSIFT(pyramid2);

    Feature::drawKeyPointsCircle(img2,keypoint2).display("keypoint2",false);
    Vec<Descriptor<KeyPointAlgo > >descriptor2 = Feature::descriptorPieChart(img2,keypoint2);
    Feature::drawDescriptorArrow(img2,descriptor2).display("descriptor",false);
    Vec<DescriptorMatch<Descriptor<KeyPointAlgo > > > match = descriptorMatch2(descriptor1,descriptor2);
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
    //        double value;
    //        Mat2UI8 threshold = Processing::thresholdOtsuMethod(img,value);//threshold segmentation
    //        threshold.save("iexthreshold.png");
    //        Mat2RGBUI8 color = Visualization::labelForeground(threshold,img);//Visual validation
    //        color.display("Segmented image",true);
    return 0;
}
