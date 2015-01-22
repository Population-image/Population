#include"data/distribution/DistributionArithmetic.h"
#include"data/distribution/DistributionAnalytic.h"
namespace pop{
DistributionArithmetic::DistributionArithmetic()
{
}

void DistributionArithmetic::setDistributionLeft(const Distribution &f_left){
    _fleft=f_left;
}
void DistributionArithmetic::setStep(F32 step)const{
    _fleft.setStep(step);
    _fright.setStep(step);
}

void DistributionArithmetic::setDistributionRight( const Distribution & f_right){
    _fright=f_right;
}

Distribution & DistributionArithmetic::getDistributionLeft(){
    return  _fleft;
}
const Distribution & DistributionArithmetic::getDistributionLeft()const{
    return  _fleft;
}
Distribution & DistributionArithmetic::getDistributionRight(){
    return  _fright;
}

const Distribution & DistributionArithmetic::getDistributionRight()const{
    return  _fright;
}
F32 DistributionArithmetic::randomVariable()const 
{
    std::cerr<<"In DistributionArithmetic::randomVariable()const ,  no  probability distribution, you have to use pop::Statistics::toProbabilityDistribution";
    return 0;
}


DistributionArithmeticAddition::DistributionArithmeticAddition()
    :DistributionArithmetic()
{
}

DistributionArithmeticAddition::DistributionArithmeticAddition(const DistributionArithmeticAddition & dist)
    :DistributionArithmetic()
{
    this->setDistributionLeft(dist.getDistributionLeft());
    this->setDistributionRight(dist.getDistributionRight());
}

DistributionArithmeticAddition * DistributionArithmeticAddition::clone()const {
    return new DistributionArithmeticAddition(*this);
}
F32 DistributionArithmeticAddition::operator ()( F32  value)const {
    return this->getDistributionLeft().operator ()(value)+ this->getDistributionRight().operator ()(value);
}


DistributionArithmeticSubtraction::DistributionArithmeticSubtraction()
    :DistributionArithmetic()
{


}

DistributionArithmeticSubtraction::DistributionArithmeticSubtraction(const DistributionArithmeticSubtraction & dist)
    :DistributionArithmetic()
{
    this->setDistributionLeft(dist.getDistributionLeft());
    this->setDistributionRight(dist.getDistributionRight());
}

DistributionArithmeticSubtraction * DistributionArithmeticSubtraction::clone()const {
    return new DistributionArithmeticSubtraction(*this);
}
F32 DistributionArithmeticSubtraction::operator ()( F32  value)const {
    return this->getDistributionLeft().operator ()(value)- this->getDistributionRight().operator ()(value);
}



DistributionArithmeticMultiplication::DistributionArithmeticMultiplication()
    :DistributionArithmetic()
{
}

DistributionArithmeticMultiplication::DistributionArithmeticMultiplication(const DistributionArithmeticMultiplication & dist)
    :DistributionArithmetic()
{
    this->setDistributionLeft(dist.getDistributionLeft());
    this->setDistributionRight(dist.getDistributionRight());
}

DistributionArithmeticMultiplication * DistributionArithmeticMultiplication::clone()const {
    return new DistributionArithmeticMultiplication(*this);
}
F32 DistributionArithmeticMultiplication::operator ()( F32  value)const {
    return this->getDistributionLeft().operator ()(value)* this->getDistributionRight().operator ()(value);
}


DistributionArithmeticDivision::DistributionArithmeticDivision()
    :DistributionArithmetic()
{
}

DistributionArithmeticDivision::DistributionArithmeticDivision(const DistributionArithmeticDivision & dist)
    :DistributionArithmetic()
{
    this->setDistributionLeft(dist.getDistributionLeft());
    this->setDistributionRight(dist.getDistributionRight());
}

DistributionArithmeticDivision * DistributionArithmeticDivision::clone()const {
    return new DistributionArithmeticDivision(*this);
}
F32 DistributionArithmeticDivision::operator ()( F32  value)const {
    return this->getDistributionLeft().operator ()(value)/this->getDistributionRight().operator ()(value);
}



DistributionArithmeticComposition::DistributionArithmeticComposition()
    :DistributionArithmetic()
{
    DistributionUniformReal d(0,1);
}

DistributionArithmeticComposition::DistributionArithmeticComposition(const DistributionArithmeticComposition & dist)
    :DistributionArithmetic()
{
    this->setDistributionLeft(dist.getDistributionLeft());
    this->setDistributionRight(dist.getDistributionRight());
}

DistributionArithmeticComposition * DistributionArithmeticComposition::clone()const {
    return new DistributionArithmeticComposition(*this);
}
F32 DistributionArithmeticComposition::operator ()( F32  value)const {
    return this->getDistributionLeft().operator ()(this->getDistributionRight().operator ()(value));
}

DistributionArithmeticMax::DistributionArithmeticMax()
    :DistributionArithmetic()
{

}

DistributionArithmeticMax::DistributionArithmeticMax(const DistributionArithmeticMax & dist)
    :DistributionArithmetic()
{
    this->setDistributionLeft(dist.getDistributionLeft());
    this->setDistributionRight(dist.getDistributionRight());
}

DistributionArithmeticMax * DistributionArithmeticMax::clone()const {
    return new DistributionArithmeticMax(*this);
}
F32 DistributionArithmeticMax::operator ()( F32  value)const {
    return maximum(this->getDistributionLeft().operator ()(value),this->getDistributionRight().operator ()(value));
}


DistributionArithmeticMin::DistributionArithmeticMin()
    :DistributionArithmetic()
{
}

DistributionArithmeticMin::DistributionArithmeticMin( const DistributionArithmeticMin &dist)
    :DistributionArithmetic()
{
    this->setDistributionLeft(dist.getDistributionLeft());
    this->setDistributionRight(dist.getDistributionRight());
}

DistributionArithmeticMin * DistributionArithmeticMin::clone()const {
    return new DistributionArithmeticMin(*this);
}
F32 DistributionArithmeticMin::operator ()( F32  value)const {
    return minimum(this->getDistributionLeft().operator ()(value),this->getDistributionRight().operator ()(value));
}
}
