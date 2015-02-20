#include"data/distribution/DistributionArithmetic.h"
#include"data/distribution/DistributionAnalytic.h"
namespace pop{

namespace Private {

DistributionConcatenation::~DistributionConcatenation()
{
    if(_fleft!=NULL)delete _fleft;
    if(_fright!=NULL)delete _fright;
}
DistributionConcatenation::DistributionConcatenation(const Distribution &f_left,const Distribution& f_right){
    _fleft = f_left.clone();
    _fright = f_right.clone();
}

DistributionConcatenation& DistributionConcatenation::operator=(const DistributionConcatenation&a){
    if(_fleft!=NULL)delete _fleft;
    if(_fright!=NULL)delete _fright;
    _fleft = a._fleft->clone();
    _fright = a._fright->clone();
    return *this;
}
}

DistributionArithmeticAddition::DistributionArithmeticAddition(const Distribution &f_left,const Distribution& f_right)
    :Distribution(),DistributionConcatenation(f_left,f_right)
{}

DistributionArithmeticAddition * DistributionArithmeticAddition::clone()const {
    return new DistributionArithmeticAddition(*this->_fleft,*this->_fright);
}
F32 DistributionArithmeticAddition::operator ()( F32  value)const {
    return _fleft->operator ()(value)+ _fright->operator ()(value);
}
F32 DistributionArithmeticAddition::randomVariable()const{
    std::cerr<<"No random variable for addition"<<std::endl;
    return 0;
}


DistributionArithmeticSubtraction::DistributionArithmeticSubtraction(const Distribution &f_left,const Distribution& f_right)
    :Distribution(),DistributionConcatenation(f_left,f_right)
{}
DistributionArithmeticSubtraction * DistributionArithmeticSubtraction::clone()const {
    return new DistributionArithmeticSubtraction(*this->_fleft,*this->_fright);
}
F32 DistributionArithmeticSubtraction::operator ()( F32  value)const {
    return _fleft->operator ()(value)- _fright->operator ()(value);
}

F32 DistributionArithmeticSubtraction::randomVariable()const{
    std::cerr<<"No random variable for addition"<<std::endl;
    return 0;
}

DistributionArithmeticMultiplication::DistributionArithmeticMultiplication(const Distribution &f_left,const Distribution& f_right)
    :Distribution(),DistributionConcatenation(f_left,f_right)
{}
DistributionArithmeticMultiplication * DistributionArithmeticMultiplication::clone()const {
    return new DistributionArithmeticMultiplication(*this->_fleft,*this->_fright);
}
F32 DistributionArithmeticMultiplication::operator ()( F32  value)const {
    return _fleft->operator ()(value)* _fright->operator ()(value);
}
F32 DistributionArithmeticMultiplication::randomVariable()const{
    std::cerr<<"No random variable for addition"<<std::endl;
    return 0;
}


DistributionArithmeticDivision::DistributionArithmeticDivision(const Distribution &f_left,const Distribution& f_right)
    :Distribution(),DistributionConcatenation(f_left,f_right)
{}
DistributionArithmeticDivision * DistributionArithmeticDivision::clone()const {
    return new DistributionArithmeticDivision(*this->_fleft,*this->_fright);
}
F32 DistributionArithmeticDivision::operator ()( F32  value)const {
    return _fleft->operator ()(value)/_fright->operator ()(value);
}
F32 DistributionArithmeticDivision::randomVariable()const{
    std::cerr<<"No random variable for addition"<<std::endl;
    return 0;
}


DistributionArithmeticComposition::DistributionArithmeticComposition(const Distribution &f_left,const Distribution& f_right)
    :Distribution(),DistributionConcatenation(f_left,f_right)
{}
DistributionArithmeticComposition * DistributionArithmeticComposition::clone()const {
    return new DistributionArithmeticComposition(*this->_fleft,*this->_fright);
}
F32 DistributionArithmeticComposition::operator ()( F32  value)const {
    return _fleft->operator ()(_fright->operator ()(value));
}
F32 DistributionArithmeticComposition::randomVariable()const{
    std::cerr<<"No random variable for addition"<<std::endl;
    return 0;
}

DistributionArithmeticMax::DistributionArithmeticMax(const Distribution &f_left,const Distribution& f_right)
    :Distribution(),DistributionConcatenation(f_left,f_right)
{}
DistributionArithmeticMax * DistributionArithmeticMax::clone()const {
    return new DistributionArithmeticMax(*this->_fleft,*this->_fright);
}
F32 DistributionArithmeticMax::operator ()( F32  value)const {
    return (std::max)(_fleft->operator ()(value),_fright->operator ()(value));
}
F32 DistributionArithmeticMax::randomVariable()const{
    std::cerr<<"No random variable for addition"<<std::endl;
    return 0;
}

DistributionArithmeticMin::DistributionArithmeticMin(const Distribution &f_left,const Distribution& f_right)
    :Distribution(),DistributionConcatenation(f_left,f_right)
{}
DistributionArithmeticMin * DistributionArithmeticMin::clone()const {
    return new DistributionArithmeticMin(*this->_fleft,*this->_fright);
}
F32 DistributionArithmeticMin::operator ()( F32  value)const {
    return (std::min)(_fleft->operator ()(value),_fright->operator ()(value));
}
F32 DistributionArithmeticMin::randomVariable()const{
    std::cerr<<"No random variable for addition"<<std::endl;
    return 0;
}
DistributionArithmeticAddition operator +(const Distribution &d1,const Distribution &d2){
    DistributionArithmeticAddition d(d1,d2);
    return d;
}
DistributionArithmeticSubtraction operator -(const Distribution &d1,const Distribution &d2){
    DistributionArithmeticSubtraction d(d1,d2);
    return d;
}
DistributionArithmeticMultiplication operator *(const Distribution &d1,const Distribution &d2){
    DistributionArithmeticMultiplication d(d1,d2);
    return d;
}
DistributionArithmeticDivision operator /(const Distribution &d1,const Distribution &d2){
    DistributionArithmeticDivision d(d1,d2);
    return d;
}
DistributionArithmeticMin minimum(const Distribution &d1,const Distribution &d2){
    DistributionArithmeticMin d(d1,d2);
    return d;
}
DistributionArithmeticMax maximum(const Distribution &d1,const Distribution &d2){
    DistributionArithmeticMax d(d1,d2);
    return d;
}
DistributionArithmeticComposition f_rho_g(const Distribution &d1,const Distribution &d2){
    DistributionArithmeticComposition d(d1,d2);
    return d;
}

}
