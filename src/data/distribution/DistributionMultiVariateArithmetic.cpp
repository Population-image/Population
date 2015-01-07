#include"data/distribution/DistributionMultiVariateArithmetic.h"
#include"data/distribution/DistributionMultiVariateFromDataStructure.h"


namespace pop{
DistributionMultiVariateArithmetic::DistributionMultiVariateArithmetic()

{
}


void DistributionMultiVariateArithmetic::setDistributionMultiVariateLeft(const DistributionMultiVariate &f_left){
    _fleft=f_left;
}
void DistributionMultiVariateArithmetic::setStep(F64 step)const{
    _fleft.setStep(step);
    _fright.setStep(step);
}

void DistributionMultiVariateArithmetic::setDistributionMultiVariateRight(const DistributionMultiVariate & f_right){
    _fright=f_right;
}

DistributionMultiVariate & DistributionMultiVariateArithmetic::getDistributionMultiVariateLeft(){
    return  _fleft;
}
const DistributionMultiVariate & DistributionMultiVariateArithmetic::getDistributionMultiVariateLeft()const{
    return  _fleft;
}
DistributionMultiVariate & DistributionMultiVariateArithmetic::getDistributionMultiVariateRight(){
    return  _fright;
}

const DistributionMultiVariate & DistributionMultiVariateArithmetic::getDistributionMultiVariateRight()const{
    return  _fright;
}
VecF64 DistributionMultiVariateArithmetic::randomVariable()const  {
    std::cerr<<"In distributionMultiVariateArithmetic::randomVariable(), no  probability distribution, you have to use pop::Statistics::toProbabilityDistribution";
    return VecF64();
}




DistributionMultiVariateArithmeticAddition::DistributionMultiVariateArithmeticAddition()
    :DistributionMultiVariateArithmetic()
{
}

DistributionMultiVariateArithmeticAddition::DistributionMultiVariateArithmeticAddition( const DistributionMultiVariateArithmeticAddition &dist)
    :DistributionMultiVariateArithmetic()
{
    this->setDistributionMultiVariateLeft(dist.getDistributionMultiVariateLeft());
    this->setDistributionMultiVariateRight(dist.getDistributionMultiVariateRight());
}

DistributionMultiVariateArithmeticAddition * DistributionMultiVariateArithmeticAddition::clone()const {
    return new DistributionMultiVariateArithmeticAddition(*this);
}
F64 DistributionMultiVariateArithmeticAddition::operator()(const VecF64&  value)const{
    return this->getDistributionMultiVariateLeft().operator ()(value)+ this->getDistributionMultiVariateRight().operator ()(value);
}





DistributionMultiVariateArithmeticSubtraction::DistributionMultiVariateArithmeticSubtraction()
    :DistributionMultiVariateArithmetic()
{

}

DistributionMultiVariateArithmeticSubtraction::DistributionMultiVariateArithmeticSubtraction(const DistributionMultiVariateArithmeticSubtraction & dist)
    :DistributionMultiVariateArithmetic()
{

    this->setDistributionMultiVariateLeft(dist.getDistributionMultiVariateLeft());
    this->setDistributionMultiVariateRight(dist.getDistributionMultiVariateRight());
}

DistributionMultiVariateArithmeticSubtraction * DistributionMultiVariateArithmeticSubtraction::clone()const {
    return new DistributionMultiVariateArithmeticSubtraction(*this);
}
F64 DistributionMultiVariateArithmeticSubtraction::operator()(const VecF64&  value)const{
    return this->getDistributionMultiVariateLeft().operator ()(value)- this->getDistributionMultiVariateRight().operator ()(value);
}


DistributionMultiVariateArithmeticMultiplication::DistributionMultiVariateArithmeticMultiplication()
    :DistributionMultiVariateArithmetic()
{

}

DistributionMultiVariateArithmeticMultiplication::DistributionMultiVariateArithmeticMultiplication(const DistributionMultiVariateArithmeticMultiplication & dist)
    :DistributionMultiVariateArithmetic()
{

    this->setDistributionMultiVariateLeft(dist.getDistributionMultiVariateLeft());
    this->setDistributionMultiVariateRight(dist.getDistributionMultiVariateRight());
}

DistributionMultiVariateArithmeticMultiplication * DistributionMultiVariateArithmeticMultiplication::clone()const {
    return new DistributionMultiVariateArithmeticMultiplication(*this);
}
F64 DistributionMultiVariateArithmeticMultiplication::operator()(const VecF64&  value)const{
    return this->getDistributionMultiVariateLeft().operator ()(value)* this->getDistributionMultiVariateRight().operator ()(value);
}




DistributionMultiVariateArithmeticDivision::DistributionMultiVariateArithmeticDivision()
    :DistributionMultiVariateArithmetic()
{
}

DistributionMultiVariateArithmeticDivision::DistributionMultiVariateArithmeticDivision(const DistributionMultiVariateArithmeticDivision & dist)
    :DistributionMultiVariateArithmetic()
{
    this->setDistributionMultiVariateLeft(dist.getDistributionMultiVariateLeft());
    this->setDistributionMultiVariateRight(dist.getDistributionMultiVariateRight());
}

DistributionMultiVariateArithmeticDivision * DistributionMultiVariateArithmeticDivision::clone()const {
    return new DistributionMultiVariateArithmeticDivision(*this);
}
F64 DistributionMultiVariateArithmeticDivision::operator()(const VecF64&  value)const{
    return this->getDistributionMultiVariateLeft().operator ()(value)/this->getDistributionMultiVariateRight().operator ()(value);
}


DistributionMultiVariateArithmeticComposition::DistributionMultiVariateArithmeticComposition()
    :DistributionMultiVariateArithmetic()
{

}

DistributionMultiVariateArithmeticComposition::DistributionMultiVariateArithmeticComposition(const DistributionMultiVariateArithmeticComposition & dist)
    :DistributionMultiVariateArithmetic()
{
    this->setDistributionMultiVariateLeft(dist.getDistributionMultiVariateLeft());
    this->setDistributionMultiVariateRight(dist.getDistributionMultiVariateRight());
}

DistributionMultiVariateArithmeticComposition * DistributionMultiVariateArithmeticComposition::clone()const {
    return new DistributionMultiVariateArithmeticComposition(*this);
}
F64 DistributionMultiVariateArithmeticComposition::operator()(const VecF64&  )const{
    return 0;
    //    return this->getDistributionMultiVariateLeft().operator ()(this->getDistributionMultiVariateRight().operator ()(value));
}




DistributionMultiVariateArithmeticMin::DistributionMultiVariateArithmeticMin()
    :DistributionMultiVariateArithmetic()
{
}

DistributionMultiVariateArithmeticMin::DistributionMultiVariateArithmeticMin(const DistributionMultiVariateArithmeticMin & dist)
    :DistributionMultiVariateArithmetic()
{

    this->setDistributionMultiVariateLeft(dist.getDistributionMultiVariateLeft());
    this->setDistributionMultiVariateRight(dist.getDistributionMultiVariateRight());
}

DistributionMultiVariateArithmeticMin * DistributionMultiVariateArithmeticMin::clone()const {
    return new DistributionMultiVariateArithmeticMin(*this);
}
F64 DistributionMultiVariateArithmeticMin::operator()(const VecF64&  value)const{
    return minimum(this->getDistributionMultiVariateLeft().operator ()(value),this->getDistributionMultiVariateRight().operator ()(value));
}


DistributionMultiVariateArithmeticMax::DistributionMultiVariateArithmeticMax()
    :DistributionMultiVariateArithmetic()
{

}

DistributionMultiVariateArithmeticMax::DistributionMultiVariateArithmeticMax(const DistributionMultiVariateArithmeticMax & dist)
    :DistributionMultiVariateArithmetic()
{
    this->setDistributionMultiVariateLeft(dist.getDistributionMultiVariateLeft());
    this->setDistributionMultiVariateRight(dist.getDistributionMultiVariateRight());
}

DistributionMultiVariateArithmeticMax * DistributionMultiVariateArithmeticMax::clone()const {
    return new DistributionMultiVariateArithmeticMax(*this);
}
F64 DistributionMultiVariateArithmeticMax::operator()(const VecF64&  value)const{
    return maximum(this->getDistributionMultiVariateLeft().operator ()(value),this->getDistributionMultiVariateRight().operator ()(value));
}


DistributionMultiVariateSeparationProduct::DistributionMultiVariateSeparationProduct()
    :DistributionMultiVariateArithmetic()
{
}
int DistributionMultiVariateSeparationProduct::getNbrVariable()const{
    return this->getDistributionMultiVariateLeft().getNbrVariable()+this->getDistributionMultiVariateRight().getNbrVariable();
}
DistributionMultiVariateSeparationProduct::DistributionMultiVariateSeparationProduct(const DistributionMultiVariateSeparationProduct & dist)
    :DistributionMultiVariateArithmetic()
{
    this->setDistributionMultiVariateLeft(dist.getDistributionMultiVariateLeft());
    this->setDistributionMultiVariateRight(dist.getDistributionMultiVariateRight());
}

DistributionMultiVariateSeparationProduct * DistributionMultiVariateSeparationProduct::clone()const {
    return new DistributionMultiVariateSeparationProduct(*this);
}
F64 DistributionMultiVariateSeparationProduct::operator()(const VecF64&  v)const{
    VecF64 vleft(this->getDistributionMultiVariateLeft().getNbrVariable());
    for(int i=0;i<this->getDistributionMultiVariateLeft().getNbrVariable();i++)
        vleft(i)=v(i);

    VecF64 vright(this->getDistributionMultiVariateRight().getNbrVariable());
    for(int i=0;i<this->getDistributionMultiVariateRight().getNbrVariable();i++)
        vright(i)=v(i+this->getDistributionMultiVariateLeft().getNbrVariable());

    return this->getDistributionMultiVariateLeft().operator ()(vleft)*this->getDistributionMultiVariateRight().operator ()(vright);
}
VecF64 DistributionMultiVariateSeparationProduct::randomVariable()const {
    VecF64 vleft = this->getDistributionMultiVariateLeft().randomVariable();
    VecF64 vright = this->getDistributionMultiVariateRight().randomVariable();

    int size = vleft.size();
    vleft.resize(vleft.size()+vright.size());
    for(int i =0;i<(int)vright.size();i++)
        vleft(size+i) = vright(i);
    return vleft;

}
DistributionMultiVariateCoupled::DistributionMultiVariateCoupled()
    :DistributionMultiVariate()
{
    _nbr_variable_coupled =0;
}
DistributionMultiVariateCoupled::DistributionMultiVariateCoupled(const DistributionMultiVariateCoupled & dist)
    :DistributionMultiVariate()
{
    _single = dist.getSingleDistribution();
    _nbr_variable_coupled = dist.getNbrVariableCoupled();
}
void DistributionMultiVariateCoupled::setNbrVariableCoupled(int nbr_variable_coupled){
    _nbr_variable_coupled=nbr_variable_coupled;
}


int DistributionMultiVariateCoupled::getNbrVariableCoupled() const{
    return _nbr_variable_coupled;
}


void DistributionMultiVariateCoupled::setSingleDistribution(const Distribution &distsingle){
    _single = distsingle;
}
Distribution DistributionMultiVariateCoupled::getSingleDistribution()const{
    return _single;
}




DistributionMultiVariateCoupled * DistributionMultiVariateCoupled::clone()const {
    return new DistributionMultiVariateCoupled(*this);
}

F64 DistributionMultiVariateCoupled::operator()(const VecF64&  value)const{
    for(unsigned int i=1;i<value.size();i++){
        if(value(1)<value(0)-_step/2 || value(1)>value(0)+_step/2)
            return 0;
    }
    return _single.operator() (value(0));
}
VecF64 DistributionMultiVariateCoupled::randomVariable()const {
    F64 v = _single.randomVariable();
    VecF64 vout(_nbr_variable_coupled,v);
    return vout;
}

 int DistributionMultiVariateCoupled::getNbrVariable()const{
     return _nbr_variable_coupled;
}


}
