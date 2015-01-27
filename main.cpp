#include <stdio.h>
#include <iostream>
#include"time.h"

#include"Population.h"//Single header
#include"data/notstable/Classifer.h"
using namespace pop;//Population namespace


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
        Mat2RGBUI8 img;
        img.load(POP_PROJECT_SOURCE_DIR+std::string("/image/iex.png"));
        Vec2F32 domain;
        domain= img.getDomain();
        ModelGermGrain2 grain = RandomGeometry::poissonPointProcess(domain,0.1);//generate the 2d Poisson point process


        DistributionExpression d("1/x^(3.1)");
        DistributionRegularStep dproba = pop::Statistics::toProbabilityDistribution(d,5,128);
        DistributionMultiVariateProduct d_radius(dproba,dproba);
        DistributionMultiVariateProduct d_angle(DistributionUniformInt(0,PI*2));


        RandomGeometry::box(grain,d_radius,d_angle);
        RandomGeometry::RGBFromMatrix(grain,img);
        grain.setModel( DeadLeave);
        grain.setBoundaryCondition(MATN_BOUNDARY_CONDITION_BOUNDED);
        Mat2RGBUI8 aborigenart = RandomGeometry::continuousToDiscrete(grain);

        aborigenart.display();
    }
    {
        //        Mat2UI8 iex;
        //        iex.load("../image/iex.pgm");
        //        Mat2F32 mverpore = Analysis::REVHistogram(iex,VecN<2,F32>(iex.getDomain())*0.5,250);

        //        VecF32 vindex = mverpore.getCol(0);//get the first column containing the grey-level range
        //        VecF32 v100 = mverpore.getCol(100);//get the col containing the histogram for r=100
        //        VecF32 v150 = mverpore.getCol(150);
        //        VecF32 v200 = mverpore.getCol(200);
        //        VecF32 v250 = mverpore.getCol(250);

        //        Mat2F32 mhistoradius100(v100.size(),2);
        //        mhistoradius100.setCol(0,vindex);
        //        mhistoradius100.setCol(1,v100);

        //        Mat2F32 mhistoradius150(v150.size(),2);
        //        mhistoradius150.setCol(0,vindex);
        //        mhistoradius150.setCol(1,v150);

        //        Mat2F32 mhistoradius200(v200.size(),2);
        //        mhistoradius200.setCol(0,vindex);
        //        mhistoradius200.setCol(1,v200);

        //        Mat2F32 mhistoradius250(v250.size(),2);
        //        mhistoradius250.setCol(0,vindex);
        //        mhistoradius250.setCol(1,v250);

        //        Distribution d100(mhistoradius100);
        //        Distribution d150(mhistoradius150);
        //        Distribution d200(mhistoradius200);
        //        Distribution d250(mhistoradius250);
        //        std::vector<Distribution> v;
        //        v.push_back(d100);v.push_back(d150);v.push_back(d200);v.push_back(d250);
        //        Distribution::multiDisplay(v);
    }
    {

        DistributionMultiVariateProduct drgb(DistributionUniformInt(0,255),DistributionUniformInt(0,255),DistributionUniformInt(0,255));
        std::cout<<drgb.randomVariable()<<std::endl;






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
