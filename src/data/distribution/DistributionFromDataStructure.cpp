#include<algorithm>
#include<cmath>

#include"data/distribution/DistributionFromDataStructure.h"
#include"data/utility/BasicUtility.h"
namespace pop{
DistributionRegularStep::DistributionRegularStep()
    :uni(0,1)
{}
DistributionRegularStep::DistributionRegularStep(const Mat2F32 & matrix)
    :uni(0,1)
{
    if(matrix.sizeI()>=2&& matrix.sizeJ()>=2){
        this->_spacing = matrix(1,0)-matrix(0,0);
        this->_xmin =  matrix(0,0);
        this->_xmax = matrix(matrix.sizeI()-1,0)+this->_spacing;
        this->_table.resize(matrix.sizeI());
        for(I32 i = 0;i<(I32)matrix.sizeI();i++)
            this->_table[i]= matrix(i,1);
        generateRepartition();
    }
}
Mat2F32 DistributionRegularStep::toMatrix()const{
    Mat2F32 m(this->_table.size(),2);
    for(unsigned int i=0;i<m.sizeI();i++){
        m(i,0) = this->_xmin + i*this->_spacing;
        m(i,1) = this->_table[i];
    }
    return m;
}


F32 DistributionRegularStep::getXmin() const{
    return _xmin;
}

F32 DistributionRegularStep::getXmax()const{
    return _xmax;
}

F32 DistributionRegularStep::getStep()const{
    return this->_spacing;
}



F32  DistributionRegularStep::fMinusOneForMonotonicallyIncreasing(F32 y)const{
    std::vector<F32>::const_iterator low=std::lower_bound (this->_table.begin(), this->_table.end(), y);
    return this->_spacing*     int(low- this->_table.begin())+this->_xmin ;
}

void DistributionRegularStep::smoothGaussian(F32 sigma){

    std::vector<F32 > _table_temp;
    _table_temp = this->_table;
    std::vector<F32> v_value_one_dimension;
    int radius_kernel=static_cast<int>(2*sigma);
    //initialisation one-dimension
    F32 sum=0;
    for(int i=0;i<2*radius_kernel+1;i++){
        F32  value =std::exp(-0.5f*(radius_kernel-i)*(radius_kernel-i)/(sigma*sigma));
        v_value_one_dimension.push_back(value);
        sum+=value;
    }
    //normalisation
    for(int i=0;i<(int)v_value_one_dimension.size();i++){
        v_value_one_dimension[i]/=sum;
    }
    for(I32 i = 0;i<(I32)this->_table.size();i++)
    {
        sum=0;
        for(I32 j = 0;j<(I32)v_value_one_dimension.size();j++)
        {
            int k= j- radius_kernel;
            F32 value;
            if((i+k)<0)
                value=_table_temp[-(i+k)-1];
            else if((i+k)>=(int)this->_table.size())
                value=_table_temp[(int)_table_temp.size()-1- ((i+k)-(int)_table_temp.size() ) ];
            else
                value = _table_temp[(i+k)];
            sum+=  v_value_one_dimension[j]*value;

        }
        this->_table[i]=sum;
    }
}

F32 DistributionRegularStep::operator ()(F32 value)const 
{
    int index = static_cast<int>(std::floor((value-_xmin)/_spacing));
    if(index<0||index>=(int)_table.size())
        return 0;
    else
        return _table[index];
}
DistributionRegularStep * DistributionRegularStep::clone()const 
{
    return new DistributionRegularStep(*this);
}


void DistributionRegularStep::generateRepartition(){
    _repartition.clear();

    F32 sum=0;
    for(I32 i = 0;i<(I32)this->_table.size();i++)
    {
        sum +=this->_table[i] ;
    }
    F32 sumtemp =0;
    for(I32 i = 0;i<(I32)this->_table.size();i++)
    {
        sumtemp +=this->_table[i] ;
        _repartition.push_back(sumtemp/sum);
    }

}
F32 DistributionRegularStep::randomVariable()const 
{
    F32 u = this->uni.randomVariable();
    std::vector<F32>::const_iterator  low=std::upper_bound (_repartition.begin(), _repartition.end(),u ); //
    I32 indice = I32(low- _repartition.begin()) ;
    return _xmin+indice*_spacing;
}


F32 DistributionExpression::operator()(F32 value)const 
{
    return fparser.Eval(& value);
}
F32 DistributionExpression::randomVariable()const {
    std::cerr<<"In DistributionExpression::randomVariable(),  no  probability distribution, you have to use pop::Statistics::toProbabilityDistribution";
    return 0;
}

DistributionExpression *DistributionExpression::clone()const 
{
    return new DistributionExpression(*this);
}


DistributionExpression::DistributionExpression(std::string regularexpression)
    :Distribution()
{
    this->fromRegularExpression(regularexpression);
}

bool DistributionExpression::fromRegularExpression(std::string function)
{
    if(function.empty()==true)
    {
        return false;
    }
    func = function;

    fparser.AddConstant("pi", static_cast<pop::F32>(3.1415926535897932));
    //
    I32 res = fparser.Parse(function, "x");
    //
    if(res >= 0)
    {
        std::string str = std::string(res+7, ' ') + "^\n"+ fparser.ErrorMsg()+ "\n\n";
        std::cerr<<"In DistributionExpression::fromRegularExpression(std::string function), the expression: "+function + " is not valid "+ str;
        fparser.Parse("x", "x");
        return false;
    }
    return true;
}
}
