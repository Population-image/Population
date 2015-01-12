#include"Population.h"//Single header
using namespace pop;//Population namespace

void plane(Mat3UI8 img){
    Mat2UI8 slice = GeometricalTransformation::plane(img,0,2);//extract the plane of the index 0 of the z-direction
    slice.display("plane");//close the windows
}
void crop(Mat3UI8 m){
    // SMALL CUBE
    Mat3UI8 small_cube = m(Vec3I32(0,0,0),Vec3I32(64,64,64));//the button corner of the cube is (0,0,0) and the top corner (64,64,64)
    small_cube.display("crop");
}
void subresolution(Mat3UI8 m){
    Mat3UI8 sub_resoltion =GeometricalTransformation::scale(m,Vec3F64(0.5,0.5,0.5));//scale the domain with the factor (0.5,0.5,0.5) so (256*0.5,256*0.5,256*0.5)=(128,128,128)
    Scene3d scene;
    Visualization::cubeExtruded(scene,sub_resoltion);//add the cube surfaces to the scene
    Visualization::lineCube(scene,sub_resoltion);//add the border red lines to the scene to the scene
    scene.display();//display the scene
}
int main()
{
    Mat3UI8 img;
    img.load(POP_PROJECT_SOURCE_DIR+std::string("/image/rock3d.pgm"));
    plane(img);
    crop(img);
    subresolution(img);
    return 0;
}
