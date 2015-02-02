//#include <ctime>
#include"data/distribution/Distribution.h"
#include"time.h"
#include"data/mat/MatN.h"
#include"data/mat/MatNInOut.h"
#include"data/mat/MatNDisplay.h"
#include"algorithm/Draw.h"
#include"algorithm/Statistics.h"
namespace pop
{
unsigned long Distribution::_init[] = {static_cast<unsigned long>(time(NULL)), 0x234, 0x345, 0x456};
unsigned long Distribution::_length = 4;
MTRand_int32 Distribution::_irand(Distribution::_init, Distribution::_length);

MTRand_int32 &Distribution::irand(){
    return _irand;
}

void DistributionDisplay::display( const Distribution & d,F32 xmin,F32 xmax,F32 ymin,F32 ymax,int sizex,int sizey){
    Vec<const Distribution*> v_d;
    v_d.push_back(&d);
    display( v_d, xmin, xmax, ymin, ymax, sizex, sizey);
}
void DistributionDisplay::display( const Distribution & d1,const Distribution & d2,F32 xmin,F32 xmax,F32 ymin,F32 ymax,int sizex,int sizey){
    Vec<const Distribution*> v_d;
    v_d.push_back(&d1);
    v_d.push_back(&d2);
    display( v_d, xmin, xmax, ymin, ymax, sizex, sizey);
}
void DistributionDisplay::display( const Distribution & d1,const Distribution & d2,const Distribution & d3,F32 xmin,F32 xmax,F32 ymin,F32 ymax,int sizex,int sizey){
    Vec<const Distribution*> v_d;
    v_d.push_back(&d1);
    v_d.push_back(&d2);
    v_d.push_back(&d3);
    display( v_d, xmin, xmax, ymin, ymax, sizex, sizey);
}
void DistributionDisplay::display(Vec<const Distribution*> v_d,F32 xmin,F32 xmax,F32 ymin,F32 ymax,int sizewidth,int sizeheight){

    if(ymin == NumericLimits<F32>::minimumRange()){
        ymin = NumericLimits<F32>::maximumRange();
        for(unsigned int i=0;i<v_d.size();i++)
            ymin = std::min(ymin,pop::Statistics::minValue(*v_d(i),xmin,xmax));
    }
    if(ymax == NumericLimits<F32>::maximumRange()){
        ymax = NumericLimits<F32>::minimumRange();
        for(unsigned int i=0;i<v_d.size();i++)
            ymax = std::max(ymax,pop::Statistics::maxValue(*v_d(i),xmin,xmax));
    }


    MatN<2,unsigned char> img (sizeheight, sizewidth);
    for(unsigned int i=0;i<v_d.size();i++)
        Draw::distribution(*v_d(i), xmin, xmax, ymin, ymax,255-i*40,img);
    Draw::axis(img, xmin, xmax, ymin, ymax,255);

    MatNDisplay main_disp;
    main_disp.display(img);
    while (!main_disp.is_closed() ) {
        main_disp.set_title("Simple plot: left(right)-arrow to move x, up(down)-to move y and +(-) to (un)zoom");

        if(main_disp.is_keyARROWDOWN()){
            F32 diff =ymax-ymin;
            ymin -= diff*0.02f;
            ymax -= diff*0.02f;
        }
        else if(main_disp.is_keyARROWUP())
        {
            F32 diff =ymax-ymin;
            ymin += diff*0.02f;
            ymax += diff*0.02f;
        }
        else if(main_disp.is_keyARROWLEFT()){
            F32 diff =xmax-xmin;
            xmin -= diff*0.02f;
            xmax -= diff*0.02f;
        }
        else if(main_disp.is_keyARROWRIGHT())
        {
            F32 diff =xmax-xmin;
            xmin += diff*0.02f;
            xmax += diff*0.02f;
        }
        else if(main_disp.is_keyPADADD())
        {
            F32 diffx =xmax-xmin;
            F32 diffy =ymax-ymin;
            xmin += diffx*0.02f;
            xmax -= diffx*0.02f;
            ymin += diffy*0.02f;
            ymax -= diffy*0.02f;
        }
        else if(main_disp.is_keyPADSUB())
        {
            F32 diffx =xmax-xmin;
            F32 diffy =ymax-ymin;
            xmin -= diffx*0.02f;
            xmax += diffx*0.02f;
            ymin -= diffy*0.02f;
            ymax += diffy*0.02f;
        }else if(main_disp.is_keyS())
        {
            img.save("snaphot.png");
        }

        img=0;
        for(unsigned int i=0;i<v_d.size();i++)
            Draw::distribution(*v_d(i), xmin, xmax, ymin, ymax,255-i*40,img);
        Draw::axis(img, xmin, xmax, ymin, ymax,255);
        main_disp.display(img).waitTime();
    }

}
void Distribution::display(F32 xmin,F32 xmax)const{
    DistributionDisplay::display(*this,xmin,xmax);
}
}
