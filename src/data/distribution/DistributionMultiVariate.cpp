#include<ctime>
#include"data/distribution/DistributionMultiVariate.h"
#include"data/distribution/DistributionMultiVariateArithmetic.h"
#include"data/distribution/DistributionMultiVariateFromDataStructure.h"
#include"data/distribution/Distribution.h"

namespace pop{
unsigned long DistributionMultiVariate::init[] = {static_cast<unsigned long>(time(NULL)), 0x234, 0x345, 0x456};
unsigned long DistributionMultiVariate::length = 4;


const DistributionMultiVariate * DistributionMultiVariate::___getPointerImplementation()const{
    if(_deriveddistribution==NULL)
        return this;
    else
        return _deriveddistribution;

}

DistributionMultiVariate * DistributionMultiVariate::___getPointerImplementation(){
    if(_deriveddistribution==NULL)
        return this;
    else
        return _deriveddistribution;

}

void DistributionMultiVariate::___setPointererImplementation(DistributionMultiVariate * d){
    if(_deriveddistribution!=NULL)
        delete _deriveddistribution;
    _deriveddistribution = d;
}

MTRand_int32 DistributionMultiVariate::irand(DistributionMultiVariate::init, DistributionMultiVariate::length);
DistributionMultiVariate::DistributionMultiVariate()
    :_deriveddistribution(NULL)
{

}

DistributionMultiVariate::DistributionMultiVariate(const DistributionMultiVariate & d)
    :_deriveddistribution(NULL)
{
    ___setPointererImplementation(d.clone());
    //    else if(type=="COUPLED"){

    //    }

}

DistributionMultiVariate::DistributionMultiVariate(const Distribution & d,int nbr_variable_coupled)
    :_deriveddistribution(NULL)
{
    DistributionMultiVariateCoupled *coupled =new DistributionMultiVariateCoupled;
    coupled->setSingleDistribution(d);
    coupled->setNbrVariableCoupled(nbr_variable_coupled);
    ___setPointererImplementation(coupled);
}
DistributionMultiVariate::DistributionMultiVariate(std::string expresssion,std::string variable)
    :_deriveddistribution(NULL)
{
     DistributionMultiVariateExpression *exp =new DistributionMultiVariateExpression;
     exp->fromRegularExpression(std::make_pair(expresssion,variable));
     ___setPointererImplementation(exp);
}
DistributionMultiVariate::DistributionMultiVariate(VecF64 mean, Mat2F64 covariance)
    :_deriveddistribution(NULL)
{
     DistributionMultiVariateNormal *exp =new DistributionMultiVariateNormal;
     exp->fromMeanVecAndCovarianceMatrix(mean,covariance);
     ___setPointererImplementation(exp);
}
DistributionMultiVariate::DistributionMultiVariate(const Distribution & f)
    :_deriveddistribution(NULL)
{
    DistributionMultiVariateFromDistribution * d = new DistributionMultiVariateFromDistribution;
    d->fromDistribution(*f.clone());
    ___setPointererImplementation(d);
}
DistributionMultiVariate::DistributionMultiVariate(const DistributionMultiVariate & f,const DistributionMultiVariate & g)
    :_deriveddistribution(NULL)
{
    DistributionMultiVariateSeparationProduct *dist = new DistributionMultiVariateSeparationProduct;
    dist->setDistributionMultiVariateLeft(f);
    dist->setDistributionMultiVariateRight(g);
    ___setPointererImplementation(dist);
}

DistributionMultiVariate::~DistributionMultiVariate()
{

}

F64 DistributionMultiVariate::operator()(const VecF64& value)const {
    if(_deriveddistribution!=NULL)
        return _deriveddistribution->operator ()(value);
    else
        throw(pexception("In DistributionMultiVariate::operator()(const VecF64& value), empty distribution"));
}


VecF64 DistributionMultiVariate::randomVariable()const{
    if(_deriveddistribution!=NULL)
        return _deriveddistribution->randomVariable();
    else
        throw(pexception("In DistributionMultiVariate:randomVariable(), empty distribution"));
}


DistributionMultiVariate & DistributionMultiVariate::operator =(const DistributionMultiVariate& d){
    this->___setPointererImplementation( d.clone());
    return *this;
}
DistributionMultiVariate  DistributionMultiVariate::rho(const DistributionMultiVariate &d)const{
    DistributionMultiVariateArithmeticComposition *dist = new DistributionMultiVariateArithmeticComposition;
    dist->setDistributionMultiVariateLeft(*this);
    dist->setDistributionMultiVariateRight(d);
    DistributionMultiVariate dout;
    dout.___setPointererImplementation(dist);
    return dout;
}
DistributionMultiVariate  DistributionMultiVariate::operator +(const DistributionMultiVariate& d)const
{
    DistributionMultiVariateArithmeticAddition *dist = new DistributionMultiVariateArithmeticAddition;
    dist->setDistributionMultiVariateLeft(*this);
    dist->setDistributionMultiVariateRight(d);

    DistributionMultiVariate dout;
    dout.___setPointererImplementation(dist);
    return dout;
}
DistributionMultiVariate  DistributionMultiVariate::operator -(const DistributionMultiVariate& d)const
{
    DistributionMultiVariateArithmeticSubtraction *dist = new DistributionMultiVariateArithmeticSubtraction;
    dist->setDistributionMultiVariateLeft(*this);
    dist->setDistributionMultiVariateRight(d);
    DistributionMultiVariate dout;
    dout.___setPointererImplementation(dist);
    return dout;
}
DistributionMultiVariate  DistributionMultiVariate::operator *(const DistributionMultiVariate& d)const
{
    DistributionMultiVariateArithmeticMultiplication *dist = new DistributionMultiVariateArithmeticMultiplication;
    dist->setDistributionMultiVariateLeft(*this);
    dist->setDistributionMultiVariateRight(d);
    DistributionMultiVariate dout;
    dout.___setPointererImplementation(dist);
    return dout;
}
DistributionMultiVariate  DistributionMultiVariate::operator /(const DistributionMultiVariate& d)const
{
    DistributionMultiVariateArithmeticDivision *dist = new DistributionMultiVariateArithmeticDivision;
    dist->setDistributionMultiVariateLeft(*this);
    dist->setDistributionMultiVariateRight(d);
    DistributionMultiVariate dout;
    dout.___setPointererImplementation(dist);
    return dout;
}
DistributionMultiVariate  DistributionMultiVariate::operator -()const
{
    DistributionMultiVariateArithmeticSubtraction *dist = new DistributionMultiVariateArithmeticSubtraction;


    DistributionMultiVariateExpression d;
    d.fromRegularExpression("0","x");
    DistributionMultiVariate dd(d);
    dist->setDistributionMultiVariateLeft(dd);
    dist->setDistributionMultiVariateRight(*this);
    DistributionMultiVariate dout;
    dout.___setPointererImplementation(dist);
    return dout;
}




DistributionMultiVariate * DistributionMultiVariate::clone()const throw(pexception){
    if(_deriveddistribution!=NULL)
        return _deriveddistribution->clone();
    else
        throw(pexception("In DistributionMultiVariate::clone(), empty distribution"));
}

void DistributionMultiVariate::setStep(F64 step)const{
    _step =step;
    if(_deriveddistribution!=NULL)
        return _deriveddistribution->setStep(step);
}


int DistributionMultiVariate::getNbrVariable()const{
    if(_deriveddistribution!=NULL)
        return _deriveddistribution->getNbrVariable();
    else
        throw(pexception("In DistributionMultiVariate::getNbrVariable, undefined DistributionMultiVariate"));
}
pop::DistributionMultiVariate maximum(const pop::DistributionMultiVariate & d1, const pop::DistributionMultiVariate & d2){
    pop::DistributionMultiVariateArithmeticMax *dist = new pop::DistributionMultiVariateArithmeticMax;
    dist->setDistributionMultiVariateLeft(d1);
    dist->setDistributionMultiVariateRight(d2);
    pop::DistributionMultiVariate dout;
    dout.___setPointererImplementation(dist);
    return dout;
}

pop::DistributionMultiVariate minimum(const pop::DistributionMultiVariate & d1, const pop::DistributionMultiVariate & d2){
    pop::DistributionMultiVariateArithmeticMin *dist = new pop::DistributionMultiVariateArithmeticMin;
    dist->setDistributionMultiVariateLeft(d1);
    dist->setDistributionMultiVariateRight(d2);
    pop::DistributionMultiVariate dout;
    dout.___setPointererImplementation(dist);
    return dout;
}
}
