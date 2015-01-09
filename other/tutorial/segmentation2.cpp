#include"Population.h"//Single header
using namespace pop;//Population namespace
int main(){
    Mat3UI8 img;
    img.load(POP_PROJECT_SOURCE_DIR+std::string("/image/rock3d.pgm"));
    //img.loadFromDirectory("/home/vincent/Desktop/tomo/","tomo2048","png");//if the 3d image is a stack of 2d images
    img = img(Vec3I32(0,0,0),Vec3I32(128,128,128));
    //        img.display();
    Mat3UI8 imgfilter= PDE::nonLinearAnisotropicDiffusion(img);
    Mat3UI8 grain= Processing::threshold(imgfilter,155);
    Mat3UI8 oil = Processing::threshold(imgfilter,70,110);
    oil = Processing::openingRegionGrowing(oil,2);//To remove the interface artefact
    Mat3UI8 air = Processing::threshold(imgfilter,0,40);
    Mat3UI8 seed = Processing::labelMerge(grain,oil);
    seed = Processing::labelMerge(seed,air);
    //        Visualization::labelForeground(seed,imgfilter).display();//check the good seed localization
    Mat3UI8 gradient = Processing::gradientMagnitudeDeriche(img,1.5);
    Mat3UI8 water = Processing::watershed(seed,gradient);
    grain = Processing::labelFromSingleSeed(water,grain);
    grain=grain/2;
    oil = Processing::labelFromSingleSeed(water,oil);
    oil = oil/4;
    Scene3d scene;
    Visualization::marchingCube(scene,grain);
    Visualization::marchingCube(scene,oil);
    Visualization::lineCube(scene,img);
    scene.display();

}
