
/*

 * CDistributionCatalog.cpp
 *
 *  Created on: 05-Dec-2009
 *      Author: vincent
 */
#include<cmath>
#include<algorithm>
#include"data/distribution/DistributionAnalytic.h"
#include"data/utility/BasicUtility.h"


#include"data/typeF/TypeF.h"
namespace pop
{
DistributionDiscrete::DistributionDiscrete(F32 step)
    :Distribution(),_step(step){

}
DistributionDiscrete::DistributionDiscrete(const DistributionDiscrete & discrete)
    :Distribution(discrete),_step(discrete._step){

}
void DistributionDiscrete::setStep(F32 step)const{
    _step = step;
}

F32 DistributionDiscrete::getStep()const{
    return _step;
}
bool DistributionDiscrete::isInStepIntervale(F32 value, F32 hitvalue)const{

    if(  value>hitvalue-(_step*1.01)*0.5&&value<hitvalue+(_step*0.99)*0.5)
        return true;
    else
        return false;
}

DistributionSign::DistributionSign()
:DistributionDiscrete(0.1){
}

DistributionSign::DistributionSign(const DistributionSign & d)
    :DistributionDiscrete(d){
}

DistributionSign * DistributionSign::clone()const {
    return new DistributionSign;
}

F32 DistributionSign::operator()(F32 value)const {
    if(this->isInStepIntervale(value,-1))
        return  1/(this->getStep()*2);
    else if(this->isInStepIntervale(value,1))
        return  1/(this->getStep()*2);
    else
        return 0;
}

F32 DistributionSign::randomVariable()const {
    if(irand()%2==1)
        return 1;
    else
        return -1;
}
std::string DistributionUniformReal::getKey(){
    return "UNIFORMREAL";
}
DistributionUniformReal * DistributionUniformReal::clone()const 
{
    return new DistributionUniformReal(_xmin,  _xmax);
}

DistributionUniformReal::DistributionUniformReal(F32 min, F32 max)
    :Distribution(),_xmin(min),_xmax(max)
{
}
DistributionUniformReal::DistributionUniformReal(const DistributionUniformReal & dist)
    :Distribution(),_xmin(dist.getXmin()),_xmax(dist.getXmax())
{
}



F32 DistributionUniformReal::getXmin()const{
    return _xmin;
}
F32 DistributionUniformReal::getXmax()const{
    return _xmax;
}
F32 DistributionUniformReal::operator()(F32 value)const 
{
    if(value>=_xmin&&value<_xmax)return 1/(_xmin-_xmax);
    else return 0;
}
F32 DistributionUniformReal::randomVariable()const 
{
    F32 value = (_xmax-_xmin)* static_cast<F32>(irand())/ static_cast<F32>(MTRand_int32::maxValue());
    return _xmin + value;
}

void DistributionUniformReal::reset(F32 xmin, F32 xmax){
    this->_xmin =xmin;
    this->_xmax =xmax;
}



//Uniform Distribution

std::string DistributionUniformInt::getKey(){return "UNIFORMINT";}
DistributionUniformInt * DistributionUniformInt::clone()const 
{
    return new DistributionUniformInt(_xmin,  _xmax);
}

DistributionUniformInt::DistributionUniformInt(int min, int max)
    :DistributionDiscrete(0.1),_xmin(min),_xmax(max)
{
}

DistributionUniformInt::DistributionUniformInt()
:DistributionDiscrete(0.1)
{
}

DistributionUniformInt::DistributionUniformInt(const DistributionUniformInt & dist)
    :DistributionDiscrete(0.1),_xmin(dist.getXmin()),_xmax(dist.getXmax())
{
}


F32 DistributionUniformInt::getXmin()const{
    return _xmin;
}

F32 DistributionUniformInt::getXmax()const{
    return _xmax;
}
void DistributionUniformInt::setXmin(F32 xmin){
    _xmin = xmin;
}

void DistributionUniformInt::setXmax(F32 xmax){
    _xmax = xmax;
}

F32 DistributionUniformInt::randomVariable()const 
{
    return _xmin + irand.operator ()()%(1+_xmax-_xmin);
}
F32 DistributionUniformInt::operator()(F32 value)const 
{
    if(value>=_xmin&&value<_xmax+1){
        if(this->isInStepIntervale(value,std::floor(value)))
            return  1/(this->getStep()*(_xmax-_xmin));

        if(this->isInStepIntervale(value,std::ceil(value)))
            return  1/(this->getStep()*(_xmax-_xmin));

    }
    return 0;
}
void DistributionUniformInt::reset(int xmin, int xmax){
    this->_xmin =xmin;
    this->_xmax =xmax;
}


//Normal distribtuion
std::string DistributionNormal::getKey(){return "NORMAL";}
DistributionNormal * DistributionNormal::clone()const 
{
    return new DistributionNormal(_mean, _standard_deviation);
}

F32 DistributionNormal::getMean()const{
    return _mean;
}

F32 DistributionNormal::getStandartDeviation()const{
    return _standard_deviation;
}


DistributionNormal::DistributionNormal(F32 mean, F32 standard_deviation)
    :_mean(mean),_standard_deviation(standard_deviation),_real(0,1)
{
}
DistributionNormal::DistributionNormal(const DistributionNormal & dist)
    :Distribution(),_mean(dist.getMean()),_standard_deviation(dist.getStandartDeviation()),_real(0,1)
{
}


F32 DistributionNormal::randomVariable()const 
{
    F32 x1, x2, w;

    do {
        x1 = 2.0 * _real.randomVariable()- 1.0;
        x2 = 2.0 * _real.randomVariable() - 1.0;
        w = x1 * x1 + x2 * x2;

    } while ( w >= 1.0 );

    w = std::sqrt( (-2.0 * std::log ( w ) ) / w );
    return (x1 * w)*_standard_deviation + _mean;
}
F32 DistributionNormal::operator()(F32 value)const 
{

    return (1/std::sqrt((2*3.141592654*_standard_deviation*_standard_deviation)))*std::exp(-(value-_mean)*(value-_mean)/(2*_standard_deviation*_standard_deviation));
}
void DistributionNormal::reset(F32 mean, F32 standard_deviation){
    this->_mean =mean;
    this->_standard_deviation=standard_deviation;
}

F32 DistributionNormal::getXmin()const{
    return _mean - 5 *_standard_deviation;
}
F32 DistributionNormal::getXmax()const{
    return _mean + 5 *_standard_deviation;
}


//Binomial
std::string DistributionBinomial::getKey(){return "BINOMIAL";}
DistributionBinomial * DistributionBinomial::clone()const 
{
    return new DistributionBinomial(_probability, _number_times);
}

F32 DistributionBinomial::getProbability()const{
    return _probability;
}

int DistributionBinomial::getNumberTime()const{
    return _number_times;
}

DistributionBinomial::DistributionBinomial(F32 probability, int number_times)
    :DistributionDiscrete(1),_probability(probability),_number_times(number_times),distreal01(0,1)
{
}
DistributionBinomial::DistributionBinomial(const DistributionBinomial & dist)
    :DistributionDiscrete(1),_probability(dist.getProbability()),_number_times(dist.getNumberTime()),distreal01(0,1)
{
}
F32 DistributionBinomial::randomVariable()const 
{
    int sum =0;
    for(int i=0;i<_number_times;i++)
        if(distreal01.randomVariable()<_probability)sum++;
    return sum;
}
F32 DistributionBinomial::operator()(F32 )const 
{
    std::cerr<<"In DistributionBinomial::operator()(F32 ), not implemented";
    return 1;
}
void DistributionBinomial::reset(F32 probability, int number_times){
    this->_probability =probability;
    this->_number_times=number_times;

}


//exponentiel
std::string DistributionExponential::getKey(){return "EXPONENTIAL";}
DistributionExponential * DistributionExponential::clone()const 
{
    return new DistributionExponential(_lambda);
}

F32 DistributionExponential::getLambda()const{
    return _lambda;
}


DistributionExponential::DistributionExponential(F32 lambda)
    :_lambda(lambda),distreal01(0,1)
{
}
DistributionExponential::DistributionExponential(const DistributionExponential & dist)
    :Distribution(),_lambda(dist.getLambda()),distreal01(0,1)
{
}
F32 DistributionExponential::randomVariable()const 
{
    return -std::log(distreal01.randomVariable())/_lambda;
}
F32 DistributionExponential::operator()(F32 value)const 
{
    if(value>=0)return _lambda*std::exp(-_lambda*value);
    else return 0;
}
void DistributionExponential::reset(F32 lambda){
    this->_lambda =lambda;

}

F32 DistributionExponential::getXmin()const{
    return 0;
}
F32 DistributionExponential::getXmax()const{
    return 4/_lambda;
}


//Poisson
std::string DistributionPoisson::getKey(){return "POISSON";}
DistributionPoisson * DistributionPoisson::clone()const 
{
    return new DistributionPoisson(_lambda);
}

F32 DistributionPoisson::getLambda()const{
    return _lambda;
}
DistributionPoisson::~DistributionPoisson()
{
    if(this->flambdalargemult==NULL)delete this->flambdalargemult;
    if(this->flambdalargerest==NULL)delete this->flambdalargerest;

}
DistributionPoisson::DistributionPoisson(F32 lambda)
    :DistributionDiscrete(1),_lambda(lambda),_maxlambda(200),flambdalargemult(NULL),flambdalargerest(NULL),distreal01(0,1)
{
    this->init();
}
DistributionPoisson::DistributionPoisson(const DistributionPoisson & dist)
    :DistributionDiscrete(1),_lambda(dist.getLambda()),_maxlambda(200),flambdalargemult(NULL),flambdalargerest(NULL),distreal01(0,1)
{
    this->init();
}
void DistributionPoisson::init()
{
    if(flambdalargemult!=NULL)delete flambdalargemult;
    if(flambdalargerest!=NULL)delete flambdalargerest;
    if(_lambda<=_maxlambda)
    {
        F64 lambda = _lambda;
        F64 presicion=0.999;

        F64 probability =std::exp(-lambda);
        F64 cummulative_distribution=probability;
        v_table.push_back(cummulative_distribution);

        int k=1;
        while(cummulative_distribution<presicion)
        {
            probability = probability * lambda/k;
            cummulative_distribution +=probability;

            k++;
            v_table.push_back(cummulative_distribution);
        }
    }
    else
    {
        this->flambdalargemult = new DistributionPoisson(_maxlambda);
        this->mult= std::floor(_lambda/_maxlambda);
        F32 rest = _lambda - this->mult*_maxlambda;
        this->flambdalargerest = new DistributionPoisson(rest);
    }


}
F32 DistributionPoisson::randomVariable()const 
{
    if(_lambda==0)return 0;

    F32 uni=distreal01.randomVariable();
    if(_lambda<=_maxlambda)
    {
        return int(std::lower_bound(v_table.begin(),v_table.end(),uni)-v_table.begin());
    }
    else
    {

        int value=0;
        for(int i=0;i<this->mult;i++)
            value+=this->flambdalargemult->randomVariable();
        value+=this->flambdalargerest->randomVariable();
        return value;
    }
}

F32 DistributionPoisson::randomVariable(F32 lambda)const 
{

    F32 uni=distreal01.randomVariable();
    F64 _v=lambda;
    F64 probability =std::exp(-_v);
    F64 cummulative_distribution=probability;
    if(uni<cummulative_distribution)
    {
        return 0;
    }
    else
    {
        int k=0;

        while(uni>cummulative_distribution)
        {
            k++;
            probability = probability * lambda/k;
            cummulative_distribution +=probability;
        }
        return k;
    }
}


F32 DistributionPoisson::operator()(F32 value)const 
{
    int index=-1;
    if(value>=0)
        index = std::floor(value);
    else
        return 0;
    if(index==-1)
        return 0;
    F64 lambda=this->_lambda;
    F64 _exp = std::exp(-lambda);
    for(int k=index;k>=1;k--)
    {
        _exp*= lambda/k;
    }
    return _exp/this->getStep();

}

void DistributionPoisson::reset(F32 lambda){
    this->_lambda =lambda;
    this->init();

}
F32 DistributionPoisson::getXmin()const{
    return 0;
}
F32 DistributionPoisson::getXmax()const{
    return 3*_lambda;
}




//Dirac
std::string DistributionDirac::getKey(){return "DIRAC";}
DistributionDirac * DistributionDirac::clone()const 
{
    return new DistributionDirac(this->_x);
}


F32 DistributionDirac::getX()const{
    return _x;
}
void DistributionDirac::reset(F32 x){
    _x = x;
}


DistributionDirac::DistributionDirac(F32 x)
    :_x(x)
{
}
DistributionDirac::DistributionDirac(DistributionDirac &dist)
    :DistributionDiscrete(),_x(dist.getX())
{
}

F32 DistributionDirac::randomVariable()const 
{
    return _x ;
}
F32 DistributionDirac::operator()(F32 value)const 
{
    if(  value>=_x-this->getStep()/2&&value<_x+this->getStep()/2)
        return 1/this->getStep();
    return 0;
}

//Triangle
std::string DistributionTriangle::getKey(){return "TRIANGLE";}
DistributionTriangle * DistributionTriangle::clone()const 
{
    return new DistributionTriangle(*this);
}


DistributionTriangle::DistributionTriangle(F32 xmin,F32 xmax,F32 peak)
    :_x_min(xmin),_x_max(xmax),_x_peak(peak),_distreal01(0,1)
{
}
DistributionTriangle::DistributionTriangle(const DistributionTriangle &dist)
    :Distribution(),_x_min(dist._x_min),_x_max(dist._x_max),_x_peak(dist._x_peak),_distreal01(0,1)
{
}

F32 DistributionTriangle::randomVariable()const 
{
    F32 x = _distreal01.randomVariable();
    return (x-_x_min)*(x-_x_min)/((_x_peak-_x_min)*(_x_max-_x_min)  ) ;
}
F32 DistributionTriangle::operator()(F32 value)const 
{
    if(value<_x_min||value>_x_max){
        return value;
    }
    else if(value<=_x_peak){
         return (value-_x_min)/(_x_peak- _x_min);
    }else{
         return (_x_max-value)/(_x_max - _x_peak);
    }
}

}
