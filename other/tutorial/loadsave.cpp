#include"Population.h"//Single header
using namespace pop;//Population namespace


void load2d(){
    Mat2UI8 m;//contruct an empty 2d matrix with 1 byte pixel type (grey-level)
    m.load(POP_PROJECT_SOURCE_DIR+std::string("/image/Lena.bmp"));//load the image
    m = m+100;//add the value 100 at each pixel value
    m.display();//display the image and to continue close the image
    Mat2RGBUI8 m_rgb;//contruct an empty 2d matrix with RGB pixel type
    m_rgb.load(POP_PROJECT_SOURCE_DIR+std::string("/image/Lena.bmp"));//load the image
    RGBUI8 red_value(100,0,0);// RGB value with Red=100, Green =0, Blue =0
    m_rgb = m_rgb+red_value;//add this value at each pixel value
    m_rgb.display();//display the image
}

void load3dDirectory(){
    Mat3UI8 m;//contruct an empty 3d matrix with 1 byte pixel type (grey-level)
    m.loadFromDirectory((POP_PROJECT_SOURCE_DIR+std::string("/image/meniscus/")).c_str(),"900-959_",".pgm");//Load all slices in the directory "${Pop_PATH}/image/meniscus/" with the given basefilename, "900-959_", and the extension ".png"
    //process ...
    m.display();//display the result use arrows to move in z-axis
    m.saveFromDirectory("process/","process",".bmp");//if the folder does not exist, we create it
}
void load3dpgm(){
    Mat3UI8 m;//contruct an empty 3d matrix with 1 byte pixel type (grey-level)
    m.load((POP_PROJECT_SOURCE_DIR+std::string("/image/rock3d.pgm")));
    m.display();//display the result use arrows to move in z-axis
    //process
    //...
    m.save("rock.pgm");//if the folder does not exist, we create it
}
void load3dRAW(){
    Mat3UI8 m;//contruct an empty 3d matrix with 1 byte pixel type
    m.loadRaw("D:/Users/vtariel/Documents/binary/data_256_256_100.raw",Vec3I32(256,256,100));
}

int main()
{
    load2d();
    load3dDirectory();
    load3dpgm();
    load3dRAW();
    return 0;
}
