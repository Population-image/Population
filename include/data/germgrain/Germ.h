#ifndef GERM_H_
#define GERM_H_
#include"data/vec/Vec.h"
#include"data/mat/Mat2x.h"
#include"algorithm/GeometricalTransformation.h"
namespace pop
{
template<int DIM>
class POP_EXPORTS Germ
{

public:
    RGBUI8 color;
    VecN<DIM,F64> x;

    Germ()
        :color(255,255,255)
    {
    }
    virtual ~Germ()
    {
    }
    void translation(const VecN<DIM,F64> & trans){x = trans +x;}


    virtual bool intersectionPoint(const VecN<DIM,F64> &    )  {
        std::cout<<"you are in the master class for the method intersction VecN"<<std::endl;
        return true;
    }
    virtual F64 getRadiusBallNorm0IncludingGrain(){
        std::cout<<"you are in the master class radius"<<std::endl;
        return true;
    }
    virtual Germ<DIM> * clone()const{
        return new Germ<DIM>(*this);
    }
    void setGerm(const Germ& germ){
        color = germ.color;
        x     = germ.x;
    }
};

namespace Details {

template<int DIM>
struct Rot{
    inline static Mat2x<F64,DIM,DIM> rotation(F64 angleradian,int coordinate);
};
template<>
struct Rot<2>{
    inline static Mat2x<F64,2,2> rotation(F64 angleradian,int ){
        return GeometricalTransformation::rotation2D(angleradian);
    }
};
template<>
struct Rot<3>{
    inline static  Mat2x<F64,3,3> rotation(F64 angleradian,int coordinate){
        return GeometricalTransformation::rotation3D(angleradian,coordinate);
    }
};

}
template<int DIM>
class POP_EXPORTS OrientationEulerAngle
{
private:
    Vec<F64> angle;
    Vec<Mat2x<F64,DIM,DIM> > M_minus;
public:

    OrientationEulerAngle(){
        if(DIM==2){
            angle.resize(1);
            M_minus.resize(1);
            setAngle_ei(0,0);
        }else{
            angle.resize(3);
            M_minus.resize(3);
            setAngle_ei(0,0);
            setAngle_ei(0,1);
            setAngle_ei(0,2);
        }

    }
    virtual ~OrientationEulerAngle(){

    }

    OrientationEulerAngle(const OrientationEulerAngle& o ){    this->angle = o.angle;this->M_minus = o.M_minus;}
    OrientationEulerAngle & operator =(const OrientationEulerAngle& o){this->angle = o.angle;this->M_minus = o.M_minus;return *this;}
    void randomAngle(){
        DistributionUniformReal uni(-pop::PI,pop::PI);

        if(DIM==2){
            setAngle_ei(uni.randomVariable(),0);
        }else{
            setAngle_ei(uni.randomVariable(),0);
            setAngle_ei(uni.randomVariable(),1);
            setAngle_ei(uni.randomVariable(),2);
        }

    }
    void setAngle_ei(F64 angleradian,int coordinate=0){
        angle(coordinate) = angleradian;
        M_minus(coordinate) = Details::Rot<DIM>::rotation(-angleradian,coordinate);

    }
    void setAngle(Vec<F64> angles_radian){
        for(unsigned int i =0;i<angles_radian.size();i++)
            setAngle_ei(angles_radian(i),i);
    }
    VecN<DIM, F64> inverseRotation(const VecN<DIM, F64> & x){
        if(DIM==2){
            return M_minus(0)*x;
        }else{
            return M_minus(0)*M_minus(1)*M_minus(2)*x;
        }
    }
    void setAngle(const OrientationEulerAngle<DIM> * o){
        this->angle = o->angle;this->M_minus = o->M_minus;
    }
};
}
#endif
