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
    {

        F32 porosity=0.95;
        F32 radius=5;
        DistributionDirac ddirac_radius(radius);

        F32 heightmix=30;
        F32 heightmax=31;
        DistributionUniformReal duniform_height(heightmix,heightmax);

        F32 moment_order2 = pop::Statistics::moment(ddirac_radius,2,0,40);
        F32 moment_order1 = pop::Statistics::moment(duniform_height,1,0,100);
        //8*E^2(R)/E^3(std::cos(theta))
        F32 volume_expectation = 3.14159265*moment_order2*moment_order1;
        Vec3F32 domain(128);//2d field domain
        F32 lambda=-std::log(porosity)/std::log(2.718)/volume_expectation;
        ModelGermGrain3 grain = RandomGeometry::poissonPointProcess(domain,lambda);//generate the 2d Poisson point process
        grain.setBoundaryCondition(MATN_BOUNDARY_CONDITION_PERIODIC);
        RandomGeometry::cylinder(grain,ddirac_radius,duniform_height);
        Mat3RGBUI8 img_VecN = RandomGeometry::continuousToDiscrete(grain);
//        img_VecN.display();
        Mat3UI8 img_VecN_grey;
        img_VecN_grey = img_VecN;

        Mat2F32 m=  Analysis::histogram(img_VecN_grey);
        std::cout<<"Realization porosity"<<m(0,1)<<std::endl;

        img_VecN_grey = pop::Processing::greylevelRemoveEmptyValue(img_VecN_grey);
        Mat3F32 phasefield = PDE::allenCahn(img_VecN_grey,5);
        phasefield = PDE::getField(img_VecN_grey,phasefield,1,3);
        Scene3d scene;
        pop::Visualization::marchingCubeLevelSet(scene,phasefield);
        pop::Visualization::lineCube(scene,img_VecN);
        scene.display();
        return 1;
    }
    F32 porosity=0.8;
    F32 radius=5;
    DistributionDirac ddirac_radius(radius);

    F32 heightmix=40;
    F32 heightmax=70;
    DistributionUniformReal duniform_height(heightmix,heightmax);

    F32 moment_order2 = pop::Statistics::moment(ddirac_radius,2,0,40);
    F32 moment_order1 = pop::Statistics::moment(duniform_height,1,0,100);
    //8*E^2(R)/E^3(std::cos(theta))
    F32 volume_expectation = 3.14159265*moment_order2*moment_order1;
    Vec3F32 domain(256);//2d field domain
    F32 lambda=-std::log(porosity)/std::log(2.718)/volume_expectation;
    ModelGermGrain3 grain = RandomGeometry::poissonPointProcess(domain,lambda);//generate the 2d Poisson point process


    //    DistributionMultiVariateFromDistribution
    //    Vec<Distribution&> v_orientation;
    //    v_orientation.push_back(Distribution(0,3.14159265,"UNIFORMREAL"));
    //    v_orientation.push_back(Distribution(0,3.14159265,"UNIFORMREAL"));
    //    v_orientation.push_back(Distribution(0,3.14159265,"UNIFORMREAL"));
    //    RandomGeometry::cylinder(grain,ddirac_radius,duniform_height,v_orientation);
    //    Mat3RGBUI8 img_VecN = RandomGeometry::continuousToDiscrete(grain);
    //    Mat3UI8 img_VecN_grey;
    //    img_VecN_grey = img_VecN;

    //    Mat2F32 m=  Analysis::histogram(img_VecN_grey);
    //    std::cout<<"Realization porosity"<<m(0,1)<<std::endl;

    //    img_VecN_grey = pop::Processing::greylevelRemoveEmptyValue(img_VecN_grey);
    //    Mat3F32 phasefield = PDE::allenCahn(img_VecN_grey,15);
    //    phasefield = PDE::getField(img_VecN_grey,phasefield,1,3);
    //    Scene3d scene;
    //    pop::Visualization::marchingCubeLevelSet(scene,phasefield);
    //    pop::Visualization::lineCube(scene,img_VecN);
    //    scene.display();

    DistributionUniformReal d(0,1);
    std::cout<<time(NULL)<<std::endl;
    std::cout<<d.randomVariable()<<std::endl;
    return 0;

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
