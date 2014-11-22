#include<algorithm>
#include<cmath>
#include"data/distribution/DistributionMultiVariateFromDataStructure.h"
#include"data/utility/BasicUtility.h"
#include"algorithm/LinearAlgebra.h"
namespace pop{

DistributionMultiVariateRegularStep::DistributionMultiVariateRegularStep()
    :uni(0,1)
{

}
DistributionMultiVariateRegularStep::DistributionMultiVariateRegularStep(const DistributionMultiVariateRegularStep & dist)
    :DistributionMultiVariate(),uni(0,1)
{
    this->_step =  dist._step;
    this->_repartition = dist._repartition;
    this->_xmin  =dist._xmin;
    this->_mat2d =dist._mat2d;

}

DistributionMultiVariateRegularStep::DistributionMultiVariateRegularStep(const MatN<2,F64> data_x_y, VecF64& xmin,F64 step)
    :DistributionMultiVariate(),uni(0,1)
{
    _xmin= xmin;
    _step  = step;
    _mat2d  = data_x_y;

    generateRepartition();
}
void DistributionMultiVariateRegularStep::generateRepartition(){
    if(_xmin.size()==2){
        _repartition.resize(_mat2d.getDomain().multCoordinate());
        double value = std::accumulate(_mat2d.begin(),_mat2d.end(),0.);

        F64 sumtemp =0;
        for(int i =0;i<_mat2d.getDomain().multCoordinate();i++){
            sumtemp+=_mat2d[i];
            _repartition[i]=sumtemp/value;
        }
    }
}




int DistributionMultiVariateRegularStep::getNbrVariable()const{
    return _xmin.size();
}

F64 DistributionMultiVariateRegularStep::operator ()(const VecF64&  v)const{
    if(_xmin.size()==2){
        Vec2F64 x = (Vec2F64(v(0),v(1))-Vec2F64(_xmin(0),_xmin(1)))/_step;
        if(_mat2d.isValid(x))
            return _mat2d(x);
        else
            return 0;

    }
    else
                throw(pexception("work only for two variates"));
}


DistributionMultiVariateRegularStep * DistributionMultiVariateRegularStep::clone()const throw(pexception){
    return new DistributionMultiVariateRegularStep(*this);
}

VecF64 DistributionMultiVariateRegularStep::randomVariable()const throw(pexception){
    F64 u = this->uni.randomVariable();
    std::vector<F64>::const_iterator low=std::upper_bound (_repartition.begin(), _repartition.end(),u ); //
    I32 indice = I32(low- _repartition.begin()) ;
    if(_xmin.size()==2){
        Vec2F64 v=VecNIndice<2>::Indice2VecN(_mat2d.getDomain(),indice);
        VecF64 vv(2);
        vv(0)=v(0)*_step+_xmin(0);vv(1)=v(1)*_step+_xmin(1);
        return vv;
    }
    else
        throw(pexception("work only for two variates"));
}


DistributionMultiVariateFromDistribution::~DistributionMultiVariateFromDistribution(){
}

DistributionMultiVariateFromDistribution::DistributionMultiVariateFromDistribution()
{
}

DistributionMultiVariateFromDistribution::DistributionMultiVariateFromDistribution(const DistributionMultiVariateFromDistribution & dist)
    :DistributionMultiVariate()
{
    fromDistribution( dist.toDistribution());
}

DistributionMultiVariateFromDistribution * DistributionMultiVariateFromDistribution::clone()const throw(pexception){
    return new DistributionMultiVariateFromDistribution(*this);
}

F64 DistributionMultiVariateFromDistribution::operator ()(const VecF64&  value)const{
    return _f.operator ()(value(0));
}

VecF64 DistributionMultiVariateFromDistribution::randomVariable()const throw(pexception){
    VecF64 v(1);
    v(0)=_f.randomVariable();
    return v;
}

void DistributionMultiVariateFromDistribution::setStep(F64 step)const{
    return _f.setStep(step);
}
void DistributionMultiVariateFromDistribution::fromDistribution(const Distribution &d){
    _f = d;
}
int DistributionMultiVariateFromDistribution::getNbrVariable()const{
    return 1;
}
Distribution  DistributionMultiVariateFromDistribution::toDistribution()const{
    return _f;
}


DistributionMultiVariateNormal::DistributionMultiVariateNormal()
    :_standard_normal(0,1)
{
}
DistributionMultiVariateNormal::DistributionMultiVariateNormal(const DistributionMultiVariateNormal & dist)
    :DistributionMultiVariate(),_standard_normal(0,1)
{
    this->fromMeanVecAndCovarianceMatrix(dist.toMeanVecAndCovarianceMatrix());
}
F64 DistributionMultiVariateNormal::operator ()(const VecF64&  value)const{

    VecF64 V= value - _mean;
    VecF64 V1 = _sigma_minus_one*V;
    F64 v = -0.5*productInner(V,V1);
    v =std::exp(v);
    v/=(std::sqrt(absolute(_determinant_sigma)));
    v/=(std::pow(2*3.141592654,this->getNbrVariable()/2));
    return v;
}
int DistributionMultiVariateNormal::getNbrVariable()const{
    return _mean.size();
}

VecF64 DistributionMultiVariateNormal::randomVariable()const throw(pexception){
    VecF64 V(this->getNbrVariable());
    for(int i = 0;i<this->getNbrVariable();i++){
        V(i)=_standard_normal.randomVariable();
    }
    return _mean + _a*V;
}

DistributionMultiVariateNormal * DistributionMultiVariateNormal::clone()const throw(pexception){
    return new DistributionMultiVariateNormal(*this);
}

void DistributionMultiVariateNormal::fromMeanVecAndCovarianceMatrix(VecF64 mean, Mat2F64 covariance){
    _mean = mean;
    _sigma = covariance;
    _sigma_minus_one=LinearAlgebra::inverseGaussianElimination(_sigma);
    _determinant_sigma = _sigma.determinant();
    _a = LinearAlgebra::AATransposeEqualMDecomposition(_sigma);

}

void DistributionMultiVariateNormal::fromMeanVecAndCovarianceMatrix(std::pair<VecF64, Mat2F64> meanvectorAndcovariancematrix){
    fromMeanVecAndCovarianceMatrix(meanvectorAndcovariancematrix.first,meanvectorAndcovariancematrix.second);
}

std::pair<VecF64,Mat2F64> DistributionMultiVariateNormal::toMeanVecAndCovarianceMatrix()const{
    return std::make_pair(_mean,_sigma);
}

F64 DistributionMultiVariateExpression::operator()( const VecF64&  value)const
{
    return fparser.Eval(static_cast<const F64*>( &(*(value.begin()))));
}
VecF64 DistributionMultiVariateExpression::randomVariable()const throw(pexception){
    throw(pexception("In distributionMultiVariateArithmetic::randomVariable(), no  probability distribution, you have to use pop::Statistics::toProbabilityDistribution"));
}

int DistributionMultiVariateExpression::getNbrVariable()const{
    return _nbrvariable;
}



DistributionMultiVariateExpression *DistributionMultiVariateExpression::clone() const throw(pexception)
{
    return new DistributionMultiVariateExpression(*this);
}


DistributionMultiVariateExpression::DistributionMultiVariateExpression()
{
}
DistributionMultiVariateExpression::DistributionMultiVariateExpression(const DistributionMultiVariateExpression & dist)
    :DistributionMultiVariate()
{
    this->_nbrvariable = dist._nbrvariable;
    this->fromRegularExpression(dist.toRegularExpression());
}

bool DistributionMultiVariateExpression::fromRegularExpression(std::pair<std::string,std::string> regularexpressionAndconcatvar){
    _func = regularexpressionAndconcatvar.first;
    _concatvar = regularexpressionAndconcatvar.second;
    _nbrvariable = 1;
    for(int i=0;i<(int)_concatvar.size();i++){
        if(_concatvar[i]==',')
            _nbrvariable++;
    }
    fparser.AddConstant("pi", 3.1415926535897932);
    //
    I32 res = fparser.Parse(regularexpressionAndconcatvar.first, regularexpressionAndconcatvar.second);
    //
    if(res >= 0)
    {

        //        std::cout << std::string(res+7, ' ') << "^\n"<< fparser.ErrorMsg() << "\n\n";
        fparser.Parse("x", "x");
        return false;
    }
    return true;
}

bool DistributionMultiVariateExpression::fromRegularExpression(std::string expression,std::string var1){
    _nbrvariable = 1;
    return fromRegularExpression(std::make_pair(expression,var1));
}

bool DistributionMultiVariateExpression::fromRegularExpression(std::string expression,std::string var1,std::string var2){
    _nbrvariable =  2;
    return fromRegularExpression(std::make_pair(expression,var1+","+var2));
}

bool DistributionMultiVariateExpression::fromRegularExpression(std::string expression,std::string var1,std::string var2,std::string var3){
    _nbrvariable =  3;
    return fromRegularExpression(std::make_pair(expression,var1+","+var2+","+var3));
}

bool DistributionMultiVariateExpression::fromRegularExpression(std::string expression,std::string var1,std::string var2,std::string var3,std::string var4){
    _nbrvariable =  4;
    return fromRegularExpression(std::make_pair(expression,var1+","+var2+","+var3+","+var4));
}



std::pair<std::string,std::string> DistributionMultiVariateExpression::toRegularExpression()const{
    return std::make_pair(_func,_concatvar);
}

int DistributionMultiVariateUnitSphere::getDIM()const{
    return _dim;
}
DistributionMultiVariateUnitSphere::DistributionMultiVariateUnitSphere(int dimension)
    :DistributionMultiVariate(),d2pi(0,2*3.14159265),d2(-1,1)
{
    _dim=dimension;
}


DistributionMultiVariateUnitSphere::DistributionMultiVariateUnitSphere(const DistributionMultiVariateUnitSphere & dist)
    :DistributionMultiVariate(),d2pi(0,2*3.14159265),d2(-1,1)
{
    _dim = dist.getDIM();
}

VecF64 DistributionMultiVariateUnitSphere::randomVariable()const throw(pexception){
    VecF64 v(_dim);
    if(_dim==3){
        // Marsaglia 3d method
        bool test=false;
        while(test==false){
            double x1 =d2.randomVariable();
            double x2 =d2.randomVariable();
            if(x1*x1+x2*x2<1){
                test=true;
                v(0)=2*x1*std::sqrt(1-x1*x1-x2*x2);
                v(1)=2*x2*std::sqrt(1-x1*x1-x2*x2);
                v(2)=1-2*(x1*x1+x2*x2);
            }
        }
    }else{
        double theta =d2pi.randomVariable();
        v(0)=std::cos(theta);
        v(1)=std::sin(theta);
    }
    return v;
}

DistributionMultiVariateUnitSphere * DistributionMultiVariateUnitSphere::clone()const throw(pexception){
    return new DistributionMultiVariateUnitSphere(_dim);
}

int DistributionMultiVariateUniformInt::getDIM()const{
    return _xmin.size();
}
DistributionMultiVariateUniformInt::DistributionMultiVariateUniformInt(const VecI32& xmin,const VecI32& xmax )
    :DistributionMultiVariate(),_d(0,1),_xmin(xmin),_xmax(xmax)
{
}


DistributionMultiVariateUniformInt::DistributionMultiVariateUniformInt(const DistributionMultiVariateUniformInt & dist)
    :DistributionMultiVariate(),_d(0,1),_xmin(dist._xmin),_xmax(dist._xmin)
{

}

VecF64 DistributionMultiVariateUniformInt::randomVariable()const throw(pexception){
    VecF64 v(_xmin.size());

    for(unsigned int i =0;i<v.size();i++){
        v(i)=std::floor(_d.randomVariable()*(_xmax(i)-_xmin(i))+_xmin(i));
    }
    return v;
}

DistributionMultiVariateUniformInt * DistributionMultiVariateUniformInt::clone()const throw(pexception){
    return new DistributionMultiVariateUniformInt(*this);
}
}
