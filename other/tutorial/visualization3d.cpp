#include"Population.h"//Single header
using namespace pop;//Population namespace


void visu2DSlice(){
    Mat3UI8 img;
    img.load(POP_PROJECT_SOURCE_DIR+std::string("/image/rock3d.pgm"));
    img.display();//use the arrows  to move in z-axis
}

void visu3DCube(){
    Scene3d scene;
    Mat3UI8 img;
    img.load(POP_PROJECT_SOURCE_DIR+std::string("/image/rock3d.pgm"));
    Visualization::cubeExtruded(scene,img);//add the cube surfaces to the scene
    Visualization::lineCube(scene,img);//add the border red lines to the scene to the scene
    scene.display(false);//display the scene
    waitKey();

}
void visu3DCubeExtrudedWithAxis(){
    Mat3UI8 img;
    img.load(POP_PROJECT_SOURCE_DIR+std::string("/image/rock3d.pgm"));
    Scene3d scene;
    Mat3UI8 extruded(img.getDomain());
    int radius=img.getDomain()(0)/2;
    Vec3I32 x1(0,0,0);
    Vec3I32 x2(img.getDomain());
    ForEachDomain3D(x,extruded){
        if((x-x1).norm(2)<radius||(x-x2).norm(2)<radius)
            extruded(x)=0;
        else
            extruded(x)=255;
    }
    Visualization::cubeExtruded(scene,img,extruded);//add the cube surfaces to the scene
    Visualization::lineCube(scene,img);//add the border red lines to the scene to the scene
    Visualization::axis(scene,40);//add axis
    scene.display(false);//display the scene
    waitKey();
}




void visu3DSlice(){
    Mat3UI8 img;
    img.load(POP_PROJECT_SOURCE_DIR+std::string("/image/rock3d.pgm"));
    Scene3d scene;
    Visualization::plane(scene,img,50,2);
    Visualization::plane(scene,img,50,1);
    Visualization::plane(scene,img,200,0);
    Visualization::lineCube(scene,img);
    scene.display(false);
    waitKey();
}


void visu3DMarchingCube(){

    Mat3UI8 img;
    img.load(POP_PROJECT_SOURCE_DIR+std::string("/image/rock3d.pgm"));
    img = img(Vec3I32(0,0,0),Vec3I32(64,64,64));

    //SEGMENTATION OF THE TREE PHASES
    Mat3UI8 imgfilter= Processing::median(img,2);
    Mat3UI8 grain= Processing::threshold(imgfilter,155);
    Mat3UI8 oil = Processing::threshold(imgfilter,70,110);
    oil = Processing::openingRegionGrowing(oil,2);//To remove the interface artefact
    Mat3UI8 air = Processing::threshold(imgfilter,0,40);
    Mat3UI8 seed = Processing::labelMerge(grain,oil);
    seed = Processing::labelMerge(seed,air);
    Mat3UI8 gradient = Processing::gradientMagnitudeDeriche(img,1.5);
    Mat3UI8 water = Processing::watershed(seed,gradient);

    //THE LABEL GRAIN WITH THE GREY-LEVEL 255*0.75
    grain = Processing::labelFromSingleSeed(water,grain);
    grain=Mat3F64(grain)*0.75;

    //THE LABEL OIL WITH THE GREY-LEVEL 255*0.4
    oil = Processing::labelFromSingleSeed(water,oil);
    oil = Mat3F64(oil)*0.4;

    Mat3UI8 grain_oil = grain+oil;
    Scene3d scene;
    Visualization::marchingCube(scene,grain_oil);//add the marching cube of the grain to the scene
    Visualization::lineCube(scene,grain_oil);//add the border red lines to the scene to the scene
    Visualization::axis(scene);
    scene.display(false);//display the scene
    waitKey();
}

int main(){
    visu2DSlice();
    visu3DCube();
    visu3DCubeExtrudedWithAxis();
    visu3DMarchingCube();
    visu3DSlice();
}
