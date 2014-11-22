#include"data/distribution/DistributionArithmetic.h"
#include"data/distribution/DistributionAnalytic.h"
namespace pop{
DistributionArithmetic::DistributionArithmetic()
{
}

void DistributionArithmetic::setDistributionLeft(const Distribution &f_left){
    _fleft=f_left;
}
void DistributionArithmetic::setStep(F64 step)const{
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
F64 DistributionArithmetic::randomVariable()const throw(pexception)
{
        throw(pexception("In DistributionArithmetic::randomVariable()const ,  no  probability distribution, you have to use pop::Statistics::toProbabilityDistribution"));
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

DistributionArithmeticAddition * DistributionArithmeticAddition::clone()const throw(pexception){
    return new DistributionArithmeticAddition(*this);
}
F64 DistributionArithmeticAddition::operator ()( F64  value)const throw(pexception){
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

DistributionArithmeticSubtraction * DistributionArithmeticSubtraction::clone()const throw(pexception){
    return new DistributionArithmeticSubtraction(*this);
}
F64 DistributionArithmeticSubtraction::operator ()( F64  value)const throw(pexception){
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

DistributionArithmeticMultiplication * DistributionArithmeticMultiplication::clone()const throw(pexception){
    return new DistributionArithmeticMultiplication(*this);
}
F64 DistributionArithmeticMultiplication::operator ()( F64  value)const throw(pexception){
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

DistributionArithmeticDivision * DistributionArithmeticDivision::clone()const throw(pexception){
    return new DistributionArithmeticDivision(*this);
}
F64 DistributionArithmeticDivision::operator ()( F64  value)const throw(pexception){
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

DistributionArithmeticComposition * DistributionArithmeticComposition::clone()const throw(pexception){
    return new DistributionArithmeticComposition(*this);
}
F64 DistributionArithmeticComposition::operator ()( F64  value)const throw(pexception){
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

DistributionArithmeticMax * DistributionArithmeticMax::clone()const throw(pexception){
    return new DistributionArithmeticMax(*this);
}
F64 DistributionArithmeticMax::operator ()( F64  value)const throw(pexception){
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

DistributionArithmeticMin * DistributionArithmeticMin::clone()const throw(pexception){
    return new DistributionArithmeticMin(*this);
}
F64 DistributionArithmeticMin::operator ()( F64  value)const throw(pexception){
    return minimum(this->getDistributionLeft().operator ()(value),this->getDistributionRight().operator ()(value));
}
}
