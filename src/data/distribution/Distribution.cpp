#include <ctime>
#include"data/distribution/Distribution.h"
#include"data/distribution/DistributionFromDataStructure.h"
#include"data/utility/BasicUtility.h"
#include"data/distribution/DistributionArithmetic.h"
#include"data/distribution/DistributionAnalytic.h"
#include"data/mat/MatN.h"
#include"data/mat/MatNInOut.h"
#include"data/mat/MatNDisplay.h"
#include"algorithm/Draw.h"
#include"algorithm/Statistics.h"
namespace pop
{
unsigned long Distribution::init[] = {static_cast<unsigned long>(time(NULL)), 0x234, 0x345, 0x456};
unsigned long Distribution::length = 4;
MTRand_int32 & Distribution::MTRand(){
    return irand;
}

Distribution::Distribution()
    :_deriveddistribution(NULL),irand(Distribution::init, Distribution::length)
{
}

Distribution::Distribution(const Distribution & d)
    :_deriveddistribution(d.clone()),irand(Distribution::init, Distribution::length)
{
}
Distribution & Distribution::operator =(const Distribution& d){
    this->___setPointererImplementation( d.clone());
    return *this;
}

Distribution::Distribution(F32 param , std::string type)
{
    if(type==DistributionPoisson::getKey() ){
        _deriveddistribution= new DistributionPoisson(param);
    }else if(type==DistributionExponential::getKey()){
        _deriveddistribution= new DistributionExponential(param);
    }else if(type==DistributionDirac::getKey())
        _deriveddistribution= new DistributionDirac(param);
    else
        std::cerr<<std::string("In Distribution::Distribution(F32 param , const char* type), type must be equal to one accepted distribution: ") +DistributionPoisson::getKey()+" "+DistributionExponential::getKey()+" "+DistributionDirac::getKey();
}
Distribution::Distribution(F32 param1,F32 param2, std::string type)
{
    if(type==DistributionBinomial::getKey()){
        _deriveddistribution= new DistributionBinomial(param1,param2);
    }else if(type==DistributionNormal::getKey()){
        _deriveddistribution= new DistributionNormal(param1,param2);
    }else if(type==DistributionUniformInt::getKey()){
        _deriveddistribution= new DistributionUniformInt(param1,param2);
    }
    else if(type==DistributionUniformReal::getKey())
        _deriveddistribution= new DistributionUniformReal(param1,param2);
    else
        std::cerr<<std::string("In Distribution::Distribution(F32 param1,F32 param2 , const char* type), type must be equal to one accepted distribution: ")+DistributionBinomial::getKey()+" "+DistributionNormal::getKey()+" "+DistributionUniformInt::getKey()+" "+DistributionUniformReal::getKey();
}

Distribution::Distribution(const Mat2F32& param,std::string type){
    if(type==std::string(DistributionRegularStep::getKey()) ){
        _deriveddistribution= new DistributionRegularStep(param);
    }else if(type==DistributionIntegerRegularStep::getKey()){
        _deriveddistribution= new DistributionIntegerRegularStep(param);
    }
    else
        std::cerr<<std::string("In Distribution::Distribution(Mat2F32 param,const char* type), type must be equal to one accepted distribution: ")+DistributionRegularStep::getKey()+" "+DistributionIntegerRegularStep::getKey();
}

Distribution::Distribution(const char* param,std::string type){
    if(type==DistributionExpression::getKey() ){
        _deriveddistribution= new DistributionExpression(param);
    }else
        std::cerr<<std::string("In Distribution::Distribution(const char* param, const char* type), type must be equal to one accepted distribution: ")+DistributionExpression::getKey();
}


Distribution::~Distribution()
{
    if(_deriveddistribution!=NULL)
        delete _deriveddistribution;
}

void Distribution::setStep(F32 )const{

}


F32 Distribution::getStep()const{
    if(_deriveddistribution!=NULL)
        return _deriveddistribution->getStep();
    else
        return 0.01;
}

F32 Distribution::getXmin()const{
    if(_deriveddistribution!=NULL)
        return _deriveddistribution->getXmin();
    else
        return -NumericLimits<F32>::maximumRange();
}
F32 Distribution::getXmax()const{
    if(_deriveddistribution!=NULL)
        return _deriveddistribution->getXmax();
    else
        return NumericLimits<F32>::maximumRange();
}
F32 Distribution::randomVariable(F32 value)const 
{
    if(_deriveddistribution!=NULL)
        return _deriveddistribution->operator ()(value);
    else{
        std::cerr<<"In Distribution::randomVariable(F32 value), empty distributio";
        return 0;
    }
}



F32 Distribution::operator()(F32 value)const {
    if(_deriveddistribution!=NULL)
        return _deriveddistribution->operator ()(value);
    else{
        std::cerr<<"In Distribution::operator()(F32 value), empty distribution";
        return 0;
    }
}

F32 Distribution::randomVariable()const {
    if(_deriveddistribution!=NULL)
        return _deriveddistribution->randomVariable();
    else{
        std::cerr<<"In Distribution::randomVariable(), empty distribution";
        return 0;
    }
}
F32 Distribution::operator()()const {
    return this->randomVariable();
}

Distribution * Distribution::clone()const {
    if(_deriveddistribution!=NULL)
        return _deriveddistribution->clone();
    else
        std::cerr<<"In Distribution::clone(), empty distribution";
    return NULL;
}

const Distribution * Distribution::___getPointerImplementation()const{

    return _deriveddistribution;

}

Distribution * Distribution::___getPointerImplementation(){

    return _deriveddistribution;

}

void Distribution::___setPointererImplementation(Distribution * d){

    delete _deriveddistribution;
    _deriveddistribution = d;
}

Distribution  Distribution::rho(const Distribution &d)const{
    DistributionArithmeticComposition *dist = new DistributionArithmeticComposition;
    dist->setDistributionLeft(*this);
    dist->setDistributionRight(d);
    Distribution dout;
    dout.___setPointererImplementation(dist);
    return dout;
}
Distribution  Distribution::operator +(const Distribution& d)const
{
    DistributionArithmeticAddition *dist = new DistributionArithmeticAddition;
    dist->setDistributionLeft(*this);
    dist->setDistributionRight(d);

    Distribution dout;
    dout.___setPointererImplementation(dist);
    return dout;
}
Distribution  Distribution::operator -(const Distribution& d)const
{
    DistributionArithmeticSubtraction *dist = new DistributionArithmeticSubtraction;
    dist->setDistributionLeft(*this);
    dist->setDistributionRight(d);
    Distribution dout;
    dout.___setPointererImplementation(dist);
    return dout;
}
Distribution  Distribution::operator *(const Distribution& d)const
{
    DistributionArithmeticMultiplication *dist = new DistributionArithmeticMultiplication;
    dist->setDistributionLeft(*this);
    dist->setDistributionRight(d);
    Distribution dout;
    dout.___setPointererImplementation(dist);
    return dout;
}
Distribution  Distribution::operator /(const Distribution& d)const
{
    DistributionArithmeticDivision *dist = new DistributionArithmeticDivision;
    dist->setDistributionLeft(*this);
    dist->setDistributionRight(d);
    Distribution dout;
    dout.___setPointererImplementation(dist);
    return dout;
}
Distribution  Distribution::operator -()const
{
    DistributionArithmeticSubtraction *dist = new DistributionArithmeticSubtraction;
    Distribution  exp ("0","EXPRESSION");
    dist->setDistributionLeft(exp);
    dist->setDistributionRight(*this);
    Distribution dout;
    dout.___setPointererImplementation(dist);
    return dout;
}

Mat2RGBUI8 Distribution::multiDisplay( Distribution & d1,Distribution & d2,Distribution & d3,F32 xmin,F32 xmax,F32 ymin,F32 ymax,int sizex,int sizey){
    std::vector<Distribution> d;
    d.push_back(d1);
    d.push_back(d2);
    d.push_back(d3);
    return Distribution::multiDisplay( d,    xmin, xmax, ymin, ymax, sizex, sizey);
}
Mat2RGBUI8 Distribution::multiDisplay( Distribution & d1,Distribution & d2,F32 xmin,F32 xmax,F32 ymin,F32 ymax,int sizex,int sizey){
    std::vector<Distribution> d;
    d.push_back(d1);
    d.push_back(d2);
    return Distribution::multiDisplay( d,    xmin, xmax, ymin, ymax, sizex, sizey);
}
Mat2RGBUI8 Distribution::multiDisplay( std::vector<Distribution> & d,F32 xmin,F32 xmax,F32 ymin,F32 ymax,int sizex,int sizey){
    if(xmin == -NumericLimits<F32>::maximumRange()){
        xmin = d[0].getXmin();
        if(xmin == -NumericLimits<F32>::maximumRange())
            xmin = 0;
    }
    if(xmax == NumericLimits<F32>::maximumRange()){
        xmax = d[0].getXmax();
        if(xmax == NumericLimits<F32>::maximumRange())
            xmax = 1;
    }
    if(ymin == -NumericLimits<F32>::maximumRange()){
        ymin = pop::Statistics::minValue(d[0],xmin,xmax);
    }
    if(ymax == NumericLimits<F32>::maximumRange()){
        ymax = pop::Statistics::maxValue(d[0],xmin,xmax);
    }
    std::vector<RGBUI8> v_RGB;
    for(int i=0;i<(int)d.size();i++)
        v_RGB.push_back(RGBUI8::randomRGB());
    MatN<2,RGBUI8> img (sizey,sizex);
    const char * c = "Simple plot: left(right)-arrow to move x, up(down)-to move y and +(-) to (un)zoom";

    MatNDisplay main_disp;


    main_disp.display(img);
    main_disp.set_title(c);
    while (!main_disp.is_closed() ) {

        if(main_disp.is_keyARROWDOWN()){
            F32 diff =ymax-ymin;
            ymin -= diff*0.02;
            ymax -= diff*0.02;
        }
        if(main_disp.is_keyARROWUP())
        {
            F32 diff =ymax-ymin;
            ymin += diff*0.02;
            ymax += diff*0.02;
        }
        if(main_disp.is_keyARROWLEFT()){
            F32 diff =xmax-xmin;
            xmin -= diff*0.02;
            xmax -= diff*0.02;
        }
        if(main_disp.is_keyARROWRIGHT())
        {
            F32 diff =xmax-xmin;
            xmin += diff*0.02;
            xmax += diff*0.02;
        }
        if(main_disp.is_keyPADADD())
        {
            F32 diffx =xmax-xmin;
            F32 diffy =ymax-ymin;
            xmin += diffx*0.02;
            xmax -= diffx*0.02;
            ymin += diffy*0.02;
            ymax -= diffy*0.02;
        }
        if(main_disp.is_keyPADSUB())
        {
            F32 diffx =xmax-xmin;
            F32 diffy =ymax-ymin;
            xmin -= diffx*0.02;
            xmax += diffx*0.02;
            ymin -= diffy*0.02;
            ymax += diffy*0.02;
        }

        img=RGBUI8(0);
        Draw::axis(img, xmin, xmax, ymin, ymax,255);


        for(int i =0;i<(int)d.size();i++){
            VecN<2,int> x1,x2;
            x1(1)=img.getDomain()(1)-40;x2(1)=img.getDomain()(1)-20;
            x1(0)=(i+1)*10;x2(0)=(i+1)*10;

            Draw::line(img,x1,x2,v_RGB[i],1);
            Draw::text(img,BasicUtility::Any2String(i).c_str(),Vec2I32(x2(0)-5,x2(1)+5),v_RGB[i]);
            Draw::distribution(d[i], xmin, xmax, ymin, ymax,v_RGB[i],img);

        }
        main_disp.display(img).waitTime();
    }
    return img;
}

Mat2RGBUI8 Distribution::display(F32 xmin,F32 xmax,F32 ymin,F32 ymax,int sizewidth,int sizeheight){
    if(xmin == -NumericLimits<F32>::maximumRange()){
        xmin = this->getXmin();
        if(xmin == -NumericLimits<F32>::maximumRange())
            xmin = 0;
    }
    if(xmax == NumericLimits<F32>::maximumRange()){
        xmax = this->getXmax();
        if(xmax == NumericLimits<F32>::maximumRange())
            xmax = 1;
    }
    if(ymin == -NumericLimits<F32>::maximumRange()){
        ymin = pop::Statistics::minValue(*this,xmin,xmax);
    }
    if(ymax == NumericLimits<F32>::maximumRange()){
        ymax = pop::Statistics::maxValue(*this,xmin,xmax);
    }


    MatN<2,unsigned char> img (sizeheight, sizewidth);
    Draw::distribution(*this, xmin, xmax, ymin, ymax,255,img);
    Draw::axis(img, xmin, xmax, ymin, ymax,255);

    MatNDisplay main_disp;

    main_disp.set_title("Simple plot: left(right)-arrow to move x, up(down)-to move y and +(-) to (un)zoom");
    main_disp.display(img);
    while (!main_disp.is_closed() ) {

        if(main_disp.is_keyARROWDOWN()){
            F32 diff =ymax-ymin;
            ymin -= diff*0.02;
            ymax -= diff*0.02;
        }
        else if(main_disp.is_keyARROWUP())
        {
            F32 diff =ymax-ymin;
            ymin += diff*0.02;
            ymax += diff*0.02;
        }
        else if(main_disp.is_keyARROWLEFT()){
            F32 diff =xmax-xmin;
            xmin -= diff*0.02;
            xmax -= diff*0.02;
        }
        else if(main_disp.is_keyARROWRIGHT())
        {
            F32 diff =xmax-xmin;
            xmin += diff*0.02;
            xmax += diff*0.02;
        }
        else if(main_disp.is_keyPADADD())
        {
            F32 diffx =xmax-xmin;
            F32 diffy =ymax-ymin;
            xmin += diffx*0.02;
            xmax -= diffx*0.02;
            ymin += diffy*0.02;
            ymax -= diffy*0.02;
        }
        else if(main_disp.is_keyPADSUB())
        {
            F32 diffx =xmax-xmin;
            F32 diffy =ymax-ymin;
            xmin -= diffx*0.02;
            xmax += diffx*0.02;
            ymin -= diffy*0.02;
            ymax += diffy*0.02;
        }else if(main_disp.is_keyS())
        {
            img.save("snaphot.png");
        }

        img=0;
        Draw::distribution(*this, xmin, xmax, ymin, ymax,255,img);
        Draw::axis(img, xmin, xmax, ymin, ymax,255);
        main_disp.display(img).waitTime();
    }
    return img;
}
pop::Distribution maximum(const pop::Distribution & d1, const pop::Distribution & d2){
    pop::DistributionArithmeticMax *dist = new pop::DistributionArithmeticMax;
    dist->setDistributionLeft(d1);
    dist->setDistributionRight(d2);
    pop::Distribution dout;
    dout.___setPointererImplementation(dist);
    return dout;
}

pop::Distribution minimum(const pop::Distribution & d1, const pop::Distribution & d2){
    pop::DistributionArithmeticMin *dist = new pop::DistributionArithmeticMin;
    dist->setDistributionLeft(d1);
    dist->setDistributionRight(d2);
    pop::Distribution dout;
    dout.___setPointererImplementation(dist);
    return dout;
}
}
