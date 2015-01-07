#include<algorithm>
#include<cmath>

#include"data/distribution/DistributionFromDataStructure.h"
#include"data/utility/BasicUtility.h"
namespace pop{
std::string  DistributionRegularStep::getKey(){return "STEP";}

DistributionRegularStep::DistributionRegularStep()
    :uni(0,1),_table(1)
{
}
DistributionRegularStep::DistributionRegularStep(const Mat2F64 & m)
    :uni(0,1)
{
    this->fromMatrix(m);
}
F64 DistributionRegularStep::getXmin() const{
    return _xmin;
}

F64 DistributionRegularStep::getXmax()const{
    return _xmax;
}

F64 DistributionRegularStep::getStep()const{
    return this->_spacing;
}

DistributionRegularStep::DistributionRegularStep(const DistributionRegularStep & dist)
    :Distribution(),uni(0,1)
{
    this->fromMatrix(dist.toMatrix());
}


F64  DistributionRegularStep::fMinusOneForMonotonicallyIncreasing(F64 y)const{
    std::vector<F64>::const_iterator low=std::lower_bound (this->_table.begin(), this->_table.end(), y);
    return this->_spacing*     int(low- this->_table.begin())+this->_xmin ;
}

void DistributionRegularStep::smoothGaussian(double sigma){

    std::vector<F64 > _table_temp;
    _table_temp = this->_table;
    std::vector<F64> v_value_one_dimension;
    int radius_kernel=2*sigma;
    //initialisation one-dimension
    double sum=0;
    for(int i=0;i<2*radius_kernel+1;i++){
        F64  value =std::exp(-0.5*(radius_kernel-i)*(radius_kernel-i)/(sigma*sigma));
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
            double value;
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

F64 DistributionRegularStep::operator ()(F64 value)const 
{
    int index = std::floor((value-_xmin)/_spacing);
    if(index<0||index>=(int)_table.size())
        return 0;
    else
        return _table[index];
}
DistributionRegularStep * DistributionRegularStep::clone()const 
{
    return new DistributionRegularStep(*this);
}


void DistributionRegularStep::fromMatrix(const Mat2F64 &matrix,F64 step) {
    if(matrix.sizeI()>=2&& matrix.sizeJ()>=2){
        this->_spacing = matrix(1,0)-matrix(0,0);
        bool constant_step=true;
        for(I32 i = 1;i<(I32)matrix.sizeI();i++)
        {
            if( matrix(i,0)- matrix(i-1,0)-this->_spacing>0.0001){
                constant_step=false;
                this->_spacing =minimum(this->_spacing, matrix(i,0)- matrix(i-1,0));
            }
        }
        if(constant_step==true){
            this->_xmin =  matrix(0,0);
            this->_xmax = matrix(matrix.sizeI()-1,0)+this->_spacing;
            this->_table.resize(matrix.sizeI());
            for(I32 i = 0;i<(I32)matrix.sizeI();i++)
                this->_table[i]= matrix(i,1);
        }else{
            if(step==0)
                this->_spacing = this->_spacing/2;
            else
                this->_spacing = step;
            this->_xmin =  matrix(0,0);
            this->_xmax =  matrix(matrix.sizeI()-1,0)+(matrix(matrix.sizeI()-1,0)-matrix(matrix.sizeI()-2,0)) ;
            this->_table.resize((this->_xmax-this->_xmin)/this->_spacing);
            F64 x=this->_xmin;
            unsigned int index = 0;
            for(I32 i = 0;i<(I32)this->_table.size();i++,x+=this->_spacing){
                if(index<matrix.sizeI()-1){
                    if(x>= matrix(index+1,0))
                        index++;
                }
                this->_table[i]= matrix(index,1);
            }

        }
        generateRepartition();
    }
}

Mat2F64 DistributionRegularStep::toMatrix()const{
    Mat2F64 m((I32)this->_table.size(),2);
    for(I32 i = 0;i<(I32)this->_table.size();i++)
    {
        m(i,0)=this->_xmin+i*this->_spacing;
        m(i,1)=this->_table[i];
    }
    return m;
}
void DistributionRegularStep::generateRepartition(){
    _repartition.clear();

    F64 sum=0;
    for(I32 i = 0;i<(I32)this->_table.size();i++)
    {
        sum +=this->_table[i] ;
    }
    F64 sumtemp =0;
    for(I32 i = 0;i<(I32)this->_table.size();i++)
    {
        sumtemp +=this->_table[i] ;
        _repartition.push_back(sumtemp/sum);
    }

}
F64 DistributionRegularStep::randomVariable()const 
{
    F64 u = this->uni.randomVariable();
    std::vector<F64>::const_iterator  low=std::upper_bound (_repartition.begin(), _repartition.end(),u ); //
    I32 indice = I32(low- _repartition.begin()) ;
    return _xmin+indice*_spacing;
}

std::string DistributionIntegerRegularStep::getKey(){return "INTEGERSTEP";}
DistributionIntegerRegularStep::~DistributionIntegerRegularStep(){

}

DistributionIntegerRegularStep::DistributionIntegerRegularStep()
    :uni(0,1),_table(1)
{
}
DistributionIntegerRegularStep::DistributionIntegerRegularStep(const Mat2F64 & m)
    :uni(0,1)
{
    this->fromMatrix(m);
}
F64 DistributionIntegerRegularStep::getXmin() const{
    return _xmin;
}

F64 DistributionIntegerRegularStep::getXmax()const{
    return _xmax;
}
DistributionIntegerRegularStep::DistributionIntegerRegularStep(const DistributionIntegerRegularStep & dist)
    :DistributionDiscrete(dist.getStep()),uni(0,1)
{
    this->fromMatrix(dist.toMatrix());
}



F64 DistributionIntegerRegularStep::operator ()(F64 value)const 
{
    if(value>=_xmin&&value<_xmax+1){
        if(this->isInStepIntervale(value,std::floor(value))){
            int index = std::floor(std::floor(value)-_xmin);
            return  _table[index]/(this->getStep());
        }
        if(this->isInStepIntervale(value,std::ceil(value))){
            int index = std::floor(std::ceil(value)-_xmin);
            return  _table[index]/(this->getStep());
        }

    }
    return 0;
}
DistributionIntegerRegularStep * DistributionIntegerRegularStep::clone()const 
{
    return new DistributionIntegerRegularStep(*this);
}


void DistributionIntegerRegularStep::fromMatrix(const Mat2F64 &matrix)
{
    if(matrix.sizeI()<2|| matrix.sizeJ()<2)
        std::cerr<<"In DistributionIntegerRegularStep::fromMatrix(const Mat2F64 & matrix,F64 step), input matrix must have at least 2 columns and 2 rows";
    this->_xmin =  matrix(0,0);
    this->_xmax = matrix(matrix.sizeI()-1,0);
    this->_table.resize(matrix.sizeI());
    for(I32 i = 0;i<(I32)matrix.sizeI();i++)
        this->_table[i]= matrix(i,1);
    this->setStep(matrix(1,0)-matrix(0,0));
    generateRepartition();
}

Mat2F64 DistributionIntegerRegularStep::toMatrix()const{
    Mat2F64 m((I32)this->_table.size(),2);
    for(I32 i = 0;i<(I32)this->_table.size();i++)
    {
        m(i,0)=this->_xmin+i*1;
        m(i,1)=this->_table[i];
    }
    return m;
}
void DistributionIntegerRegularStep::generateRepartition(){
    _repartition.clear();

    F64 sum=0;
    for(I32 i = 0;i<(I32)this->_table.size();i++)
    {
        sum +=this->_table[i] ;
    }
    F64 sumtemp =0;
    for(I32 i = 0;i<(I32)this->_table.size();i++)
    {
        sumtemp +=this->_table[i] ;
        _repartition.push_back(sumtemp/sum);
    }

}
F64 DistributionIntegerRegularStep::randomVariable()const 
{
    F64 u = this->uni.randomVariable();
    std::vector<F64>::const_iterator low=std::upper_bound (_repartition.begin(), _repartition.end(),u ); //
    I32 indice = I32(low- _repartition.begin()) ;
    return _xmin+indice;
}


std::string DistributionExpression::getKey(){return "EXPRESSION";}
F64 DistributionExpression::operator()(F64 value)const 
{
    return fparser.Eval(& value);
}
F64 DistributionExpression::randomVariable()const {
    std::cerr<<"In DistributionExpression::randomVariable(),  no  probability distribution, you have to use pop::Statistics::toProbabilityDistribution";
    return 0;
}

DistributionExpression *DistributionExpression::clone()const 
{
    return new DistributionExpression(*this);
}

DistributionExpression::~DistributionExpression(){

}

DistributionExpression::DistributionExpression()
{
}
DistributionExpression::DistributionExpression(std::string regularexpression)
    :Distribution()
{
    this->fromRegularExpression(regularexpression);
}
DistributionExpression::DistributionExpression(const DistributionExpression & dist)
    :Distribution()
{ 
    this->fromRegularExpression(dist.toRegularExpression());
}

bool DistributionExpression::fromRegularExpression(std::string function)
{
    if(function.empty()==true)
    {
        return false;
    }
    func = function;

    fparser.AddConstant("pi", 3.1415926535897932);
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

std::string DistributionExpression::toRegularExpression()const{
    return func;
}
//const  std::string DistributionPencil::KEY = "DistributionPencil";
//F64 DistributionPencil::operator ()(F64 value)const {
//    std::vector<F64>::const_iterator it = std::lower_bound (_x_values.begin(),_x_values.end(),value);
//    I32 indice = I32(it- _x_values.begin()) ;
//    if(  value>=*it-this->getStep()/2&&value<*it+this->getStep()/2)
//        return _m(indice,1)/this->getStep();
//    else if(it+1!=_x_values.end()&& value>=*(it+1)-this->getStep()/2&&value<*(it+1)+this->getStep()/2)
//        return _m(indice+1,1)/this->getStep();
//    else return 0;


//}
//F64 DistributionPencil::randomVariable()const {
//    F64 u = this->uni.randomVariable();
//    std::vector<F64>::const_iterator low=std::upper_bound (_repartition.begin(), _repartition.end(),u ); //
//    I32 indice = I32(low- _repartition.begin()) ;
//    return  _m(indice,0);
//}

//void DistributionPencil::fromMatrix(const Mat2F64 & matrix){
//    if(matrix.sizeI()<2|| matrix.sizeJ()<2)
//        std::cerr<<"In DistributionPencil::fromMatrix(const Mat2F64 & matrix), input matrix must have at least 2 columns and 2 rows";
//    //tri a bulle
//    _m = matrix;
//    bool test = true;
//    while(test == true){
//        test = false;
//        for(unsigned int i =0;i<_m.sizeI()-1;i++){
//            if(_m(i,0)>_m(i+1,0))
//            {
//                test = true;
//                F64 v0=_m(i,0);
//                F64 v1=_m(i,1);
//                _m(i,0)=_m(i+1,0);
//                _m(i,1)=_m(i+1,1);
//                _m(i+1,0) = v0;
//                _m(i+1,1) = v1;
//            }
//        }
//    }


//    generateDistribution();
//}

//Mat2F64 DistributionPencil::toMatrix()const{
//    return _m;
//}
//DistributionPencil::~DistributionPencil()
//{}
//DistributionPencil::DistributionPencil()
//    :_index_old(0),uni(0,1)
//{
//    this->_key = DistributionPencil::KEY;
//}
//DistributionPencil::DistributionPencil(const Mat2F64 & m)
//    :DistributionDiscrete(),_index_old(0),  uni(0,1)
//{
//    this->_key = DistributionPencil::KEY;
//    this->fromMatrix(m);
//}

//DistributionPencil::DistributionPencil(const DistributionPencil & dist)
//    :DistributionDiscrete(),_index_old(0),  uni(0,1)
//{
//    this->_key = DistributionPencil::KEY;
//    this->fromMatrix(dist.toMatrix());
//}


//void DistributionPencil::generateDistribution(){
//    _repartition.clear();
//    _x_values.clear();
//    F64 sum=0;
//    F64 diff=NumericLimits<F64>::maximumRange();
//    for(unsigned int i = 0;i<_m.sizeI();i++)
//    {
//        if(i!=_m.sizeI()-1)
//        {
//            diff=minimum(diff, _m(i+1,0)-_m(i,0));
//        }
//        _x_values.push_back(_m(i,0));
//        sum +=_m(i,1) ;
//    }
//    this->setStep(diff/10);
//    F64 sumtemp =0;
//    for(I32 i = 0;i<(I32)_m.sizeI();i++)
//    {
//        sumtemp +=_m(i,1) ;
//        _repartition.push_back(sumtemp/sum);
//    }

//}
//void DistributionPencil::fromString(std::string str){
//    std::istringstream iss(str );
//    Mat2F64 m;
//    iss>>m;
//    this->fromMatrix(m);
//}

//std::string DistributionPencil::toString()const {
//    std::ostringstream oss;
//    oss << this->_m;
//    return oss.str();
//}
//DistributionPencil * DistributionPencil::clone()const {
//    return  new DistributionPencil(*this);
//}
//namespace
//{
//const bool registered_DistributionPencil = DistributionFactory::getInstance()->Register(  DistributionPencil::KEY, new DistributionPencil());
//}
}
