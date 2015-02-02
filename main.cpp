#include <stdio.h>
#include <iostream>
#include"time.h"

#include"Population.h"//Single header
#include"data/notstable/Classifer.h"
using namespace pop;//Population namespace


struct SeedPlane
{
    SeedPlane(){}
    SeedPlane(Vec2F32 x)
        :_x(x){}
    Vec2F32 _x;
    typedef Vec2F32  Feature;
    F32 operator ()(const Vec2F32 & x)const{
        return (_x-x).norm(2);
    }
};
class DistributionRing
{
public:
    double _radius;
    DistributionUniformReal _random_angle;
    DistributionRegularStep _regular_step;
    DistributionRing()
        :_random_angle(0,2*PI){
        std::string exp = "1./(x*sqrt(8*pi^3))*exp(-0.5*(x-4)^2)";
        DistributionExpression d(exp);
        for(unsigned int i=0;i<5;i++){
            std::cout<<d(0.5*i)<<std::endl;
        }
        _regular_step = Statistics::toProbabilityDistribution(d,0,10,0.01);


    }
    Vec2F64 randomVariable()const {
        double r_random =_regular_step.randomVariable();
        double r_angle  = _random_angle.randomVariable();
        VecF32 v(2);
        v(0)= r_random*std::cos(r_angle);
        v(1)= r_random*std::sin(r_angle);
        return v;
    }
};

int main(){
    Mat2UI8 m(4,4);
    m(1,1)=1;m(1,2)=1;
    m(2,1)=1;m(2,2)=1;
    std::vector<WaveletHaar<2,I32 > > v_wavelet = WaveletHaar<2,I32>::baseWaveletHaar();
    WaveletHaar<2,I32 > & wavelet = v_wavelet[0];
    wavelet.setMatrix(m);
    Vec2I32 x(2,2);
    std::cout<<wavelet.operator ()(x)<<std::endl;

//    std::cout<< wavelet<<std::endl;
}
int toto(){
    F32 mux=0;
    F32 muy=0;
    F32 sigmax=1;
    F32 sigmay=1;
    F32 rho = 0;
    DistributionMultiVariateNormal multi1(mux,muy,sigmax,sigmay,rho);
    DistributionRing multi2;
    //    Mat2RGBUI8 img(100,100);
    //    img.fill(RGBUI8(255,255,255));



    DistributionUniformReal d1(-5,5);

    Vec<ClassiferThresholdWeak<SeedPlane> > v_classifier_weak;
    int nbr_features=500;
    for(unsigned int j=0;j<nbr_features;j++){
        ClassiferThresholdWeak<SeedPlane> classifier;
        SeedPlane c_unit(Vec2F32(d1.randomVariable(),d1.randomVariable()));
        classifier.setFeatureToScalar(c_unit);
        v_classifier_weak.push_back(classifier);
    }

    int number_trainings=10000;
    VecF32 v_weight(number_trainings);
    Vec<int> v_affect(number_trainings);
    Vec<VecF32> v_coeff( nbr_features ,VecF32(number_trainings));

    for(unsigned int i=0;i<number_trainings;i++){
        v_weight(i)=1.f/number_trainings;
        if(i%2==0){
            v_affect(i)=false;
            for(unsigned int j=0;j<nbr_features;j++){
                v_coeff(j)(i)= v_classifier_weak(j).getFeatureToScalar()(multi1.randomVariable());
            }
        }else{
            v_affect(i)=true;
            for(unsigned int j=0;j<nbr_features;j++){
                v_coeff(j)(i)= v_classifier_weak(j).getFeatureToScalar()(multi2.randomVariable());
            }
        }
    }

    for(unsigned int j=0;j<nbr_features;j++){
        v_classifier_weak(j).setWeight(v_weight);
        v_classifier_weak(j).setTraining(v_coeff(j),v_affect);
    }

    ClassiferAdaBoost<ClassiferThresholdWeak<SeedPlane> > c_ada_boost;
    c_ada_boost.training(v_classifier_weak,200);
    Mat2RGBUI8 img(200,200);
    double factor = 10;
    for(int i=0;i<img.sizeI();i++){
        for(int j=0;j<img.sizeJ();j++){
            Vec2F32 v(  (i-(int)img.sizeI()/2)*factor/img.sizeI(), (j-(int)img.sizeJ()/2)*factor/img.sizeJ());
            if(c_ada_boost.operator ()(v)==true)
                img(i,j)=RGBUI8(255,255,255);
            else
                img(i,j)=0;
        }
    }
    MatNDisplay windows;
    while(0==0)
    {
        Vec2I32 v1 = multi1.randomVariable()*Vec2F32(img.getDomain())/factor+img.getDomain()/2;
        Vec2I32 v2 = multi2.randomVariable()*Vec2F32(img.getDomain())/factor+img.getDomain()/2;
        if(img.isValid(v1)){
            img(v1)=RGBUI8(255,0,0);
        }
        if(img.isValid(v2)){
            img(v2)=RGBUI8(0,0,255);
        }
        windows.display(img);

    }



    //    c.training();
    //    std::cout<<c._error<<std::endl;
    //    std::cout<<c._threshold<<std::endl;
    //    std::cout<<c._sign<<std::endl;

    //    c.setFeatureToScalar(feature);
    //    std::cout<<feature(Vec2F32(100,100))<<std::endl;
    //    std::cout<<c(Vec2F32(100,100))<<std::endl;
    return 1;




    //    VecF32 v_affec(4);
    //    VecF32 v_coeff(4);
    //    VecF32 v_weigh(4);
    //    v_affec(0)=0;v_coeff(0)=0.3;
    //    v_affec(1)=1;v_coeff(1)=2;
    //    v_affec(2)=0;v_coeff(2)=-0.5;
    //    v_affec(3)=1;v_coeff(3)=-4;
    //    c.setTraining(v_coeff,v_affec);
    //    v_weigh(0)=0.2;v_weigh(1)=0.2;v_weigh(2)=0.2;v_weigh(3)=0.2;
    //    c.setWeight(v_weigh);
    //    c.training();
    //    std::cout<<c._error<<std::endl;
    //    std::cout<<c._threshold<<std::endl;
    //    std::cout<<c._sign<<std::endl;




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
