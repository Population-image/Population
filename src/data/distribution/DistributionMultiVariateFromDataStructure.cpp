#include<algorithm>
#include<cmath>
#include"data/distribution/DistributionMultiVariateFromDataStructure.h"
#include"data/utility/BasicUtility.h"
#include"algorithm/LinearAlgebra.h"
namespace pop{


DistributionMultiVariateRegularStep::DistributionMultiVariateRegularStep(const MatN<2,F32> data_x_y, VecF32& xmin,F32 step)
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
        F32 value = std::accumulate(_mat2d.begin(),_mat2d.end(),0.);

        F32 sumtemp =0;
        for(int i =0;i<_mat2d.getDomain().multCoordinate();i++){
            sumtemp+=_mat2d[i];
            _repartition[i]=sumtemp/value;
        }
    }
}




unsigned int DistributionMultiVariateRegularStep::getNbrVariable()const{
    return static_cast<unsigned int>(_xmin.size());
}

F32 DistributionMultiVariateRegularStep::operator ()(const VecF32&  v)const{
    if(_xmin.size()==2){
        Vec2F32 x = (Vec2F32(v(0),v(1))-Vec2F32(_xmin(0),_xmin(1)))/_step;
        if(_mat2d.isValid(x))
            return _mat2d(x);
        else
            return 0;

    }
    else
        std::cerr<<"work only for two variates";
    return 0;
}


DistributionMultiVariateRegularStep * DistributionMultiVariateRegularStep::clone()const {
    return new DistributionMultiVariateRegularStep(*this);
}

VecF32 DistributionMultiVariateRegularStep::randomVariable()const {
    F32 u = this->uni.randomVariable();
    std::vector<F32>::const_iterator low=std::upper_bound (_repartition.begin(), _repartition.end(),u ); //
    I32 indice = I32(low- _repartition.begin()) ;
    if(_xmin.size()==2){
        Vec2F32 v=VecNIndice<2>::Indice2VecN(_mat2d.getDomain(),indice);
        VecF32 vv(2);
        vv(0)=v(0)*_step+_xmin(0);vv(1)=v(1)*_step+_xmin(1);
        return vv;
    }
    else{
        std::cerr<<"work only for two variates";
        return VecF32();
    }

}


DistributionMultiVariateNormal::DistributionMultiVariateNormal(VecF32 mean, Mat2F32 covariance)
    :DistributionMultiVariate(),_standard_normal(0,1)
{
    _mean = mean;
    _sigma = covariance;
    _sigma_minus_one=LinearAlgebra::inverseGaussianElimination(_sigma);
    _determinant_sigma = _sigma.determinant();

    _a = LinearAlgebra::AATransposeEqualMDecomposition(_sigma);

}

F32 DistributionMultiVariateNormal::operator ()(const VecF32&  value)const{

    VecF32 V= value - _mean;
    VecF32 V1 = _sigma_minus_one*V;
    F32 v = -0.5*productInner(V,V1);
    v =std::exp(v);
    v/=(std::sqrt(absolute(_determinant_sigma)));
    v/=(std::pow(2*3.141592654,this->getNbrVariable()/2));
    return v;
}
unsigned int DistributionMultiVariateNormal::getNbrVariable()const{
    return static_cast<unsigned int>(_mean.size());
}

VecF32 DistributionMultiVariateNormal::randomVariable()const {
    VecF32 V(this->getNbrVariable());
    for(unsigned int i = 0;i<this->getNbrVariable();i++){
        V(i)=_standard_normal.randomVariable();
    }
    return _mean + _a*V;
}

DistributionMultiVariateNormal * DistributionMultiVariateNormal::clone()const {
    return new DistributionMultiVariateNormal(*this);
}

F32 DistributionMultiVariateExpression::operator()( const VecF32&  value)const
{
    return fparser.Eval(static_cast<const F32*>( &(*(value.begin()))));
}
VecF32 DistributionMultiVariateExpression::randomVariable()const {
    std::cerr<<"In distributionMultiVariateArithmetic::randomVariable(), no  probability distribution, you have to use pop::Statistics::toProbabilityDistribution";
    return VecF32();
}

unsigned int DistributionMultiVariateExpression::getNbrVariable()const{
    return _nbrvariable;
}



DistributionMultiVariateExpression *DistributionMultiVariateExpression::clone() const 
{
    return new DistributionMultiVariateExpression(*this);
}



bool DistributionMultiVariateExpression::fromRegularExpression(std::pair<std::string,std::string> regularexpressionAndconcatvar){
    _func = regularexpressionAndconcatvar.first;
    _concatvar = regularexpressionAndconcatvar.second;
    _nbrvariable = 1;
    for(int i=0;i<(int)_concatvar.size();i++){
        if(_concatvar[i]==',')
            _nbrvariable++;
    }
    fparser.AddConstant("pi", 3.1415926535897932f);
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
DistributionMultiVariateExpression::DistributionMultiVariateExpression(std::string expression,std::string var1)
    :DistributionMultiVariate()
{
    _nbrvariable = 1;
    fromRegularExpression(std::make_pair(expression,var1));
}
DistributionMultiVariateExpression::DistributionMultiVariateExpression(std::string expression,std::string var1,std::string var2)
    :DistributionMultiVariate()
{
    _nbrvariable =  2;
    fromRegularExpression(std::make_pair(expression,var1+","+var2));
}

DistributionMultiVariateExpression::DistributionMultiVariateExpression(std::string expression,std::string var1,std::string var2,std::string var3)
    :DistributionMultiVariate()
{
    _nbrvariable =  3;
    fromRegularExpression(std::make_pair(expression,var1+","+var2+","+var3));
}

DistributionMultiVariateExpression::DistributionMultiVariateExpression(std::string expression,std::string var1,std::string var2,std::string var3,std::string var4)
    :DistributionMultiVariate()
{
    _nbrvariable =  4;
    fromRegularExpression(std::make_pair(expression,var1+","+var2+","+var3+","+var4));
}





unsigned int DistributionMultiVariateUnitSphere::getNbrVariable()const{
    return _dim;
}
DistributionMultiVariateUnitSphere::DistributionMultiVariateUnitSphere(int dimension)
    :DistributionMultiVariate(),d2pi(0,static_cast<pop::F32>(2*3.14159265)),d2(-1,1)
{
    _dim=dimension;
}
 F32 DistributionMultiVariateUnitSphere::operator()(const VecF32& )const{
    std::cerr<<"Not implemented "<<std::endl;
    return 0;
}

VecF32 DistributionMultiVariateUnitSphere::randomVariable()const {
    VecF32 v(_dim);
    if(_dim==3){
        // Marsaglia 3d method
        bool test=false;
        while(test==false){
            F32 x1 =d2.randomVariable();
            F32 x2 =d2.randomVariable();
            if(x1*x1+x2*x2<1){
                test=true;
                v(0)=2*x1*std::sqrt(1-x1*x1-x2*x2);
                v(1)=2*x2*std::sqrt(1-x1*x1-x2*x2);
                v(2)=1-2*(x1*x1+x2*x2);
            }
        }
    }else{
        F32 theta =d2pi.randomVariable();
        v(0)=std::cos(theta);
        v(1)=std::sin(theta);
    }
    return v;
}

DistributionMultiVariateUnitSphere * DistributionMultiVariateUnitSphere::clone()const {
    return new DistributionMultiVariateUnitSphere(_dim);
}

unsigned int DistributionMultiVariateUniformInt::getNbrVariable()const{
    return static_cast<unsigned int>(_xmin.size());
}
DistributionMultiVariateUniformInt::DistributionMultiVariateUniformInt(const VecI32& xmin,const VecI32& xmax )
    :DistributionMultiVariate(),_d(0,1),_xmin(xmin),_xmax(xmax)
{
}

VecF32 DistributionMultiVariateUniformInt::randomVariable()const {
    VecF32 v(static_cast<int>(_xmin.size()));

    for(unsigned int i =0;i<v.size();i++){
        v(i)=std::floor(_d.randomVariable()*(_xmax(i)-_xmin(i))+_xmin(i));
    }
    return v;
}

DistributionMultiVariateUniformInt * DistributionMultiVariateUniformInt::clone()const {
    return new DistributionMultiVariateUniformInt(_xmin,_xmax);
}
F32 DistributionMultiVariateUniformInt::operator()(const VecF32& )const{
   std::cerr<<"Not implemented "<<std::endl;
   return 0;
}
DistributionMultiVariateProduct::~DistributionMultiVariateProduct(){
    for(unsigned int i=0;i<_v_dist.size();i++){
        if(_v_dist[i]!=NULL)delete _v_dist[i];
    }
    _v_dist.clear();
}
DistributionMultiVariateProduct& DistributionMultiVariateProduct::operator=(const DistributionMultiVariateProduct&a){
    for(unsigned int i=0;i<_v_dist.size();i++){
        if(_v_dist[i]!=NULL)delete _v_dist[i];
    }
    _v_dist.clear();
    for(unsigned int i=0;i<a._v_dist.size();i++){
        _v_dist.push_back(a._v_dist(i)->clone());
    }
	return *this;
}

DistributionMultiVariateProduct::DistributionMultiVariateProduct(const Distribution & dist1,const Distribution & dist2){
    _v_dist.push_back(dist1.clone());
    _v_dist.push_back(dist2.clone());
}

DistributionMultiVariateProduct::DistributionMultiVariateProduct(const Distribution & dist1,const Distribution & dist2,const Distribution & dist3){
    _v_dist.push_back(dist1.clone());
    _v_dist.push_back(dist2.clone());
    _v_dist.push_back(dist3.clone());
}

DistributionMultiVariateProduct::DistributionMultiVariateProduct(const Vec<Distribution*> v_dist){
    for(unsigned int i=0;i<v_dist.size();i++){
        _v_dist.push_back(v_dist(i)->clone());
    }
}

DistributionMultiVariateProduct * DistributionMultiVariateProduct::clone()const {
     return new DistributionMultiVariateProduct(_v_dist);
}
 F32 DistributionMultiVariateProduct::operator()(const VecF32&  value)const{
     double sum=0;
     for(unsigned int i=0;i<_v_dist.size();i++){
         sum =_v_dist(i)->operator ()(value(i));
     }
     return sum;
 }
VecF32 DistributionMultiVariateProduct::randomVariable()const{
    VecF32 random(this->getNbrVariable());
    for(unsigned int i=0;i<_v_dist.size();i++){
        random(i) =_v_dist(i)->randomVariable();
    }
    return random;
}
unsigned int DistributionMultiVariateProduct::getNbrVariable()const{
    return static_cast<unsigned int>(_v_dist.size());
}
}
