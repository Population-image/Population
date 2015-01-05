#include"Population.h"//Single header
using namespace pop;//Population namespace
#include <functional>
int main()
{
//    auto f0    = [](double x){return x+1;};
//    std::cout<<f0(10)<<std::endl;

//    auto f1    = ;
    Mat2RGBUI8 imgg;
    imgg.load(POP_PROJECT_SOURCE_DIR+(std::string)"/image/outil.bmp");

    imgg = PDE::nonLinearAnisotropicDiffusion(imgg,100);
    imgg.display();
    Mat2F64 m1(3,3);m1(1,1)=0.25;
    Mat2F64 m2(3,3);m2(1,1)=1.25;
    auto it_d = m1.getIteratorEDomain();
    forEachFunctorBinaryFF(m1,m2,m1,[](double x1,double x2){return x1+x2;},it_d);
    std::cout<<m1<<std::endl;
    return 1;



    Mat2UI8 img;
    img = Processing::fill(img,(UI8)255);
    img.load(POP_PROJECT_SOURCE_DIR+(std::string)"/image/outil.bmp");
    img = img.opposite();
    Mat2Vec2F64 vel;
    int dir=0;
    VecF64 kx = PDE::permeability(img,vel,dir,0.01);
    vel= GeometricalTransformation::scale(vel,Vec2F64(8));
    Mat2RGBUI8 c = Visualization::vectorField2DToArrows(vel);
    c.display("velocity",true,false);
    dir=1;
    VecF64 ky = PDE::permeability(img,vel,dir,0.01);


//    FunctorF::
//    vel.

//    pop::Visualization::labelToRGBGradation(pop::Convertor::toVecN().display();
    Mat2F64 K(2,2);
    K.setCol(0,kx);
    K.setCol(1,ky);
    std::cout<<K<<std::endl;

    try{//Enclose this portion of code in a try block
        Mat2UI8 img;
        img.load(POP_PROJECT_SOURCE_DIR+std::string("/image/iex.png"));//replace this path by those on your computer
        img.display("Initial image",false);
        img = PDE::nonLinearAnisotropicDiffusion(img);//filtering
        img.display();
        double value;
        Mat2UI8 threshold = Processing::thresholdOtsuMethod(img,value);//threshold segmentation
        threshold.save("iexthreshold.png");
        Mat2RGBUI8 color = Visualization::labelForeground(threshold,img);//Visual validation
        color.display("Segmented image",true);
    }
    catch(const pexception &e){
        e.display();//Display the error in a window
    }
    return 0;
}
