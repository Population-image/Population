#include"Population.h"//Single header
using namespace pop;//Population namespace


class DistributionPolynomial : public Distribution
{
public:
    DistributionPolynomial(){

    }

    F32 operator()(F32 value)const{
        if(_v_a.empty()==true){
            return 0;
        }else{
            F32 sum = _v_a(0);
            for(unsigned int i=1;i<_v_a.size();i++){
                sum += value*_v_a(i);
                value*=value;
            }
            return sum;
        }
    }

    void setTerms(const Vec<F32> v_a){
        _v_a = v_a;
    }
    Vec<F32> getTerms(const Vec<F32> v_a)const{
        return _v_a;
    }
    F32 randomVariable()const{
        std::cerr<<"No randomVariable for the polynomial expression";
        return 0;
    }
    Distribution * clone()const{
        return new DistributionPolynomial(*this);
    }
    int getDegree()const
    {

        return _v_a.size()-1;
    }
private:
    Vec<F32> _v_a;
};
template<typename InputData,typename OutputData>
struct Learning{
    virtual ~Learning(){}
    virtual void setObservedData(const Vec<InputData>& input_values,const Vec<InputData>& target_values )=0;
    virtual OutputData process(const InputData& input_values)=0;
};


DistributionPolynomial PolynomialCurveFittingQuadraticError(int degree, const Vec<F32>& input_values,const Vec<F32>& target_values ){
    VecF32 T(degree+1);
    for(unsigned int k=0;k<T.size();k++){
        F32 sum=0;
        for(unsigned int j=0;j<input_values.size();j++){
            sum+=target_values(j)*std::pow(input_values(j),k);
        }
        T(k)=sum;
    }

    Mat2F32 M(degree+1,degree+1);

    for(unsigned int k=0;k<M.sizeI();k++){

        for(unsigned int i=0;i<M.sizeJ();i++){
            F32 sum=0;
            for(unsigned int j=0;j<input_values.size();j++){
                sum+=std::pow(input_values(j),k+i);
            }
            M(k,i)=sum;
        }
    }
    VecF32 W= LinearAlgebra::inverseGaussianElimination(M)*T;
    DistributionPolynomial d;
    d.setTerms(W);
    //        std::cout<<W<<std::endl;
    return d;
}

int main(){
    VecF32 v_x;
    VecF32 v_y;

    v_x.push_back(0);v_y.push_back(0);
    v_x.push_back(1);v_y.push_back(2);
    v_x.push_back(2);v_y.push_back(1);
    v_x.push_back(10);v_y.push_back(13);
    DistributionPolynomial d = PolynomialCurveFittingQuadraticError(2,v_x,v_y);
    std::cout<<d(0)<<std::endl;
    std::cout<<d(1)<<std::endl;
    std::cout<<d(2)<<std::endl;
    std::cout<<d(10)<<std::endl;
    return 0;


}
