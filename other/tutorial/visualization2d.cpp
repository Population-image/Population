#include"Population.h"//Single header
using namespace pop;//Population namespace


//#### class member ###
void classmember(){
    Mat2UI8 img;
    img.load(POP_PROJECT_SOURCE_DIR+std::string("/image/Lena.bmp"));
    img.display();//#display the image
    img.display("lena",false,false);//#display the image with lena title without stopping the execution without resizing the image
}

//#### display class ###
void classdisplay(){
    Mat2UI8 img;
    img.load(POP_PROJECT_SOURCE_DIR+std::string("/image/Lena.bmp"));
    MatNDisplay disp;//#class to display the image
    //#execute and display the result in the object window disp while this windows is not closed
    do{
        img = Processing::erosion(img,1);
        disp.display(img);
    }while(disp.is_closed()==false);
}

//#### Visualization algorithms ###
void visualizealgorithm(){
    //# label image in foreground of a grey(color) imageto check segmentation or seed localization
    Mat2UI8 img;
    img.load(POP_PROJECT_SOURCE_DIR+std::string("/image/Lena.bmp"));
    F32 value;
    Mat2UI8 thre = Processing::thresholdOtsuMethod(img,value);
    Mat2RGBUI8 foreground = Visualization::labelForeground (thre,img,0.7);
    foreground.display();

    //# display each label with a random colour
    DistributionPoisson d(0.001);
    Mat2UI32 field(512,512);//#realisation of a discrete poisson field
    Mat2UI32::IteratorEDomain it = field.getIteratorEDomain();
    int label = 1;
    while(it.next()){
        if(d.randomVariable() != 0){
            field(it.x())=label;
            label++;
        }
    }
    Visualization::labelToRandomRGB(field).display("seed",false);
    field = Processing::voronoiTesselationEuclidean(field);//#voronoi tesselation with  2-norm
    Mat2RGBUI8 voronoi = Visualization::labelToRandomRGB(field);
    voronoi.display();

}

int main(){
    classmember();
    classdisplay();
    visualizealgorithm();
}
