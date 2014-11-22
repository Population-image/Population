#include"Population.h"//Single header
using namespace pop;//Population namespace
int main(){
    CollectorExecutionInformationSingleton::getInstance()->setActivate(true);//execution information in the standart outstream
    try{//Enclose this portion of code in a try block with
        Mat2UI8 img;
        img.load("/home/vincent/population/iex.png");
        Mat2UI8 filter = Processing::smoothDeriche(img,1);
        filter = Processing::dynamic(filter,40);
        Mat2UI16 minima = Processing::minimaRegional(filter);
//        Visualization::labelForeground(minima,img).display();
        Mat2UI16 water =Processing::watershedBoundary(minima,filter,1);
        Mat2UI16 boundary = Processing::threshold(water,0,0);//the boundary label is 0
//        boundary.display("boundary",true,false);
        minima = Processing::labelMerge(boundary,minima);
        Mat2UI8 gradient = Processing::gradientMagnitudeDeriche(img,1);
        water = Processing::watershed(minima,gradient);
        Visualization::labelForeground(water,img).display();
    }
    catch(const pexception &e){
        std::cerr<<e.what()<<std::endl;
    }
}
