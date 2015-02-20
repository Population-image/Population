#include"Population.h"//Single header
#include"data/notstable/CharacteristicCluster.h"
using namespace pop;//Population namespace

int main(int argc, char *argv[]){
    Mat3UI8 img;
    img.load(POP_PROJECT_SOURCE_DIR+std::string("/image/rock3d.pgm"));
    img = img(Vec3I32(0,0,0),Vec3I32(50,50,50));
    //        img.display();
    Mat3UI8 imgfilter=Processing::median(img,2,2);
    //        imgfilter.display();
    Mat2F32 m = Analysis::histogram(img);
    DistributionRegularStep d(m);
    //DistributionDisplay::display(d,d.getXmin(),d.getXmax());
    int threshold;
    Mat3UI8 grain= Processing::thresholdOtsuMethod(imgfilter,threshold);
    //        grain.display();
    Mat3RGBUI8 color= Visualization::labelForegroundBoundary(grain,imgfilter);
    color.display();
    grain.save("grain.pgm");
}

