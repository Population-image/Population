#include"Population.h"
using namespace pop;//Population namespace



int main(){
    pop::Mat3UI8 m(5,5);
    m.load("../image/rock3d.pgm");
    m.display();
    //    Mat3UI8 fb;

    //    fb.load("../image/rock3d.pgm");
    //    fb= fb(Vec3I32(0,0,0),Vec3I32(100,100,100));
    //    double value;
    //    fb = Processing::thresholdOtsuMethod(fb,value);
    //    //fb.display();
    //    fb = Processing::greylevelRemoveEmptyValue(fb);
    //    Mat2F64 prev = Analysis::REVPorosity(fb,VecN<3,F64>(fb.getDomain())*0.5,200);
    //    std::cout<<prev<<std::endl;
    //    prev.saveAscii("profile.dat");
    //fb.display();
    //    try {
    ////        VideoFFMPEG c;
    ////        c.open("D:/Users/vtariel/Desktop/video_courte_Orly.mp4");
    ////        MatNDisplay disp;
    ////        while(c.grabMatrixGrey()){
    ////            disp.display(c.retrieveMatrixGrey());
    ////        }
    //        pop::Mat2UI8 m(1000,1000);

    //        int time1;
    //        time1 = time(NULL);
    //        m = m*m;
    //        std::cout<<time(NULL)-time1<<std::endl;
    //       // m.display();
    //    }
    //    catch(const pexception &e){
    //        e.display();//Display the error in a window
    //    }
}
