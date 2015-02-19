#include"data/distribution/DistributionMultiVariateArithmetic.h"
namespace pop{

namespace Private{
DistributionMultiVariateConcatenation::~DistributionMultiVariateConcatenation()
{
    if(_fleft!=NULL)delete _fleft;
    if(_fright!=NULL)delete _fright;
}
DistributionMultiVariateConcatenation::DistributionMultiVariateConcatenation(const DistributionMultiVariate &f_left,const DistributionMultiVariate& f_right){
    _fleft = f_left.clone();
    _fright = f_right.clone();
}

DistributionMultiVariateConcatenation& DistributionMultiVariateConcatenation::operator=(const DistributionMultiVariateConcatenation&a){
    if(_fleft!=NULL)delete _fleft;
    if(_fright!=NULL)delete _fright;
    _fleft = a._fleft->clone();
    _fright = a._fright->clone();
    return *this;
}
}

DistributionMultiVariateArithmeticAddition::DistributionMultiVariateArithmeticAddition(const DistributionMultiVariate &f_left,const DistributionMultiVariate& f_right)
    :DistributionMultiVariate(),DistributionMultiVariateConcatenation(f_left,f_right)
{}

DistributionMultiVariateArithmeticAddition * DistributionMultiVariateArithmeticAddition::clone()const {
    return new DistributionMultiVariateArithmeticAddition(*this->_fleft,*this->_fright);
}
F32 DistributionMultiVariateArithmeticAddition::operator ()( const VecF32&  value)const {
    return _fleft->operator ()(value)+ _fright->operator ()(value);
}
VecF32 DistributionMultiVariateArithmeticAddition::randomVariable()const{
    std::cerr<<"No random variable for addition"<<std::endl;
    return VecF32();
}
unsigned int DistributionMultiVariateArithmeticAddition::getNbrVariable()const{
    return _fleft->getNbrVariable() +  _fright->getNbrVariable();
}

DistributionMultiVariateArithmeticSubtraction::DistributionMultiVariateArithmeticSubtraction(const DistributionMultiVariate &f_left,const DistributionMultiVariate& f_right)
    :DistributionMultiVariate(),DistributionMultiVariateConcatenation(f_left,f_right)
{}
DistributionMultiVariateArithmeticSubtraction * DistributionMultiVariateArithmeticSubtraction::clone()const {
    return new DistributionMultiVariateArithmeticSubtraction(*this->_fleft,*this->_fright);
}
F32 DistributionMultiVariateArithmeticSubtraction::operator ()( const VecF32&  value)const {
    return _fleft->operator ()(value)- _fright->operator ()(value);
}

VecF32 DistributionMultiVariateArithmeticSubtraction::randomVariable()const{
    std::cerr<<"No random variable for addition"<<std::endl;
    return VecF32();
}
unsigned int DistributionMultiVariateArithmeticSubtraction::getNbrVariable()const{
    return _fleft->getNbrVariable() +  _fright->getNbrVariable();
}


DistributionMultiVariateArithmeticMultiplication::DistributionMultiVariateArithmeticMultiplication(const DistributionMultiVariate &f_left,const DistributionMultiVariate& f_right)
    :DistributionMultiVariate(),DistributionMultiVariateConcatenation(f_left,f_right)
{}
DistributionMultiVariateArithmeticMultiplication * DistributionMultiVariateArithmeticMultiplication::clone()const {
    return new DistributionMultiVariateArithmeticMultiplication(*this->_fleft,*this->_fright);
}
F32 DistributionMultiVariateArithmeticMultiplication::operator ()( const VecF32&  value)const {
    return _fleft->operator ()(value)* _fright->operator ()(value);
}
VecF32 DistributionMultiVariateArithmeticMultiplication::randomVariable()const{
    std::cerr<<"No random variable for addition"<<std::endl;
    return VecF32();
}
unsigned int DistributionMultiVariateArithmeticMultiplication::getNbrVariable()const{
    return _fleft->getNbrVariable() +  _fright->getNbrVariable();
}

DistributionMultiVariateArithmeticDivision::DistributionMultiVariateArithmeticDivision(const DistributionMultiVariate &f_left,const DistributionMultiVariate& f_right)
    :DistributionMultiVariate(),DistributionMultiVariateConcatenation(f_left,f_right)
{}
DistributionMultiVariateArithmeticDivision * DistributionMultiVariateArithmeticDivision::clone()const {
    return new DistributionMultiVariateArithmeticDivision(*this->_fleft,*this->_fright);
}
F32 DistributionMultiVariateArithmeticDivision::operator ()( const VecF32&  value)const {
    return _fleft->operator ()(value)/_fright->operator ()(value);
}
VecF32 DistributionMultiVariateArithmeticDivision::randomVariable()const{
    std::cerr<<"No random variable for addition"<<std::endl;
    return VecF32();
}
unsigned int DistributionMultiVariateArithmeticDivision::getNbrVariable()const{
    return _fleft->getNbrVariable() +  _fright->getNbrVariable();
}

DistributionMultiVariateArithmeticMax::DistributionMultiVariateArithmeticMax(const DistributionMultiVariate &f_left,const DistributionMultiVariate& f_right)
    :DistributionMultiVariate(),DistributionMultiVariateConcatenation(f_left,f_right)
{}
DistributionMultiVariateArithmeticMax * DistributionMultiVariateArithmeticMax::clone()const {
    return new DistributionMultiVariateArithmeticMax(*this->_fleft,*this->_fright);
}
F32 DistributionMultiVariateArithmeticMax::operator ()( const VecF32&  value)const {
    return std::max(_fleft->operator ()(value),_fright->operator ()(value));
}
VecF32 DistributionMultiVariateArithmeticMax::randomVariable()const{
    std::cerr<<"No random variable for addition"<<std::endl;
    return VecF32();
}
unsigned int DistributionMultiVariateArithmeticMax::getNbrVariable()const{
    return _fleft->getNbrVariable() +  _fright->getNbrVariable();
}

DistributionMultiVariateArithmeticMin::DistributionMultiVariateArithmeticMin(const DistributionMultiVariate &f_left,const DistributionMultiVariate& f_right)
    :DistributionMultiVariate(),DistributionMultiVariateConcatenation(f_left,f_right)
{}
DistributionMultiVariateArithmeticMin * DistributionMultiVariateArithmeticMin::clone()const {
    return new DistributionMultiVariateArithmeticMin(*this->_fleft,*this->_fright);
}
F32 DistributionMultiVariateArithmeticMin::operator ()( const VecF32&  value)const {
    return std::min(_fleft->operator ()(value),_fright->operator ()(value));
}
VecF32 DistributionMultiVariateArithmeticMin::randomVariable()const{
    std::cerr<<"No random variable for addition"<<std::endl;
    return VecF32();
}
unsigned int DistributionMultiVariateArithmeticMin::getNbrVariable()const{
    return _fleft->getNbrVariable() +  _fright->getNbrVariable();
}

DistributionMultiVariateArithmeticAddition operator +(const DistributionMultiVariate &d1,const DistributionMultiVariate &d2){
    DistributionMultiVariateArithmeticAddition d(d1,d2);
    return d;
}
DistributionMultiVariateArithmeticSubtraction operator -(const DistributionMultiVariate &d1,const DistributionMultiVariate &d2){
    DistributionMultiVariateArithmeticSubtraction d(d1,d2);
    return d;
}
DistributionMultiVariateArithmeticMultiplication operator *(const DistributionMultiVariate &d1,const DistributionMultiVariate &d2){
    DistributionMultiVariateArithmeticMultiplication d(d1,d2);
    return d;
}
DistributionMultiVariateArithmeticDivision operator /(const DistributionMultiVariate &d1,const DistributionMultiVariate &d2){
    DistributionMultiVariateArithmeticDivision d(d1,d2);
    return d;
}
DistributionMultiVariateArithmeticMin minimum(const DistributionMultiVariate &d1,const DistributionMultiVariate &d2){
    DistributionMultiVariateArithmeticMin d(d1,d2);
    return d;
}
DistributionMultiVariateArithmeticMax maximum(const DistributionMultiVariate &d1,const DistributionMultiVariate &d2){
    DistributionMultiVariateArithmeticMax d(d1,d2);
    return d;
}



}
