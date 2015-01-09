#include"data/germgrain/GermGrain.h"
namespace pop{
GrainCylinder::GrainCylinder()
    : maxradius(0)
{
}
F64 GrainCylinder::getRadiusBallNorm0IncludingGrain(){
    if(maxradius==0)maxradius=std::sqrt(radius*radius+height*height/4);
    return  maxradius;
}
bool GrainCylinder::intersectionPoint(const VecN<3,F64> &  x_value)
{
    VecN<3,F64> p = this->x -x_value;
    p = this->orientation.inverseRotation(p);
    if(p[0]*p[0]+p[1]*p[1]>this->radius*this->radius)
        return false;
    else if(p[2]>height/2.||p[2]<-height/2.)
        return false;
    else
        return true;
}
Germ<3> * GrainCylinder::clone() const
{
    return new GrainCylinder( *this );
}
//OrientationEulerAngle<3>::OrientationEulerAngle(){
//    setAngle_ei(0,0);
//    setAngle_ei(0,1);
//    setAngle_ei(0,2);
//}

//void OrientationEulerAngle<3>::setAngle_ei(F64 angleradian,int index_coordinate)
//{
//    if(index_coordinate==0){
//        this->anglex = angleradian;
//        Mx_minus1 =GeometricalTransformation::rotation3D(-angleradian,0);
//    }
//    else if(index_coordinate==1){
//        this->angley = angleradian;
//        My_minus1 =  GeometricalTransformation::rotation3D(-angleradian,1);
//    }
//    else if(index_coordinate==2){
//        this->anglez = angleradian;
//        Mz_minus1 =  GeometricalTransformation::rotation3D(-angleradian,2);
//    }
//    else std::cout<<"Problem only three axes for 3D for the grains";
//}
//void OrientationEulerAngle<3>::setAngle(Vec<F64> vangleradian)
//{
//    if((int)vangleradian.size()!=3)
//    {
//        std::cerr<<"Three angles for the grain orientation for DIM=3"<<std::endl;
//    }
//    this->setAngle_ei(vangleradian[0], 0);
//    this->setAngle_ei(vangleradian[1], 1);
//    this->setAngle_ei(vangleradian[2], 2);
//}
//void OrientationEulerAngle<3>::randomAngle()
//{
//    DistributionUniformReal uni(-pop::PI,pop::PI);
//    this->setAngle_ei( uni.randomVariable(),0);
//    this->setAngle_ei( uni.randomVariable(),1);
//    this->setAngle_ei( uni.randomVariable(),2);
//}
//VecN<3, F64> OrientationEulerAngle<3>::inverseRotation(VecN<3, F64> & x)
//{
//    VecN<3, F64> v(x);
//    v= Mz_minus1*v;
//    v= My_minus1*v;
//    v= Mx_minus1*v;
//    return v;
//}
//void OrientationEulerAngle<3>::setAngle(const OrientationEulerAngle<3> * grain)
//{
//    this->setAngle_ei(grain->anglex, 0);
//    this->setAngle_ei(grain->angley, 1);
//    this->setAngle_ei(grain->anglez, 2);
//}

//OrientationEulerAngle<2>::OrientationEulerAngle(){
//    setAngle_ei(0,0);
//}
//OrientationEulerAngle<2>::OrientationEulerAngle(const OrientationEulerAngle<2>& o){
//    this->angle = o.angle;
//    this->M_minus1 = o.M_minus1;
//}

//OrientationEulerAngle<2> & OrientationEulerAngle<2>::operator =(const OrientationEulerAngle<2>& o){
//    this->angle = o.angle;
//    this->M_minus1 = o.M_minus1;
//    return *this;
//}

//OrientationEulerAngle<3>::OrientationEulerAngle(const OrientationEulerAngle<3>& o){
//    this->anglex = o.anglex;
//    this->angley = o.angley;
//    this->anglez = o.anglez;
//    this->Mx_minus1= o.Mx_minus1;
//    this->My_minus1= o.My_minus1;
//    this->Mz_minus1= o.Mz_minus1;
//}

//OrientationEulerAngle<3> & OrientationEulerAngle<3>::operator =(const OrientationEulerAngle<3>& o){
//    this->anglex = o.anglex;
//    this->angley = o.angley;
//    this->anglez = o.anglez;
//    this->Mx_minus1= o.Mx_minus1;
//    this->My_minus1= o.My_minus1;
//    this->Mz_minus1= o.Mz_minus1;
//    return *this;
//}

//void OrientationEulerAngle<2>::setAngle_ei(F64 angleradian,int index_coordinate)
//{
//    if(index_coordinate==0){
//        angle = angleradian;
//        M_minus1 = GeometricalTransformation::rotation2D(-angleradian);
//    }
//    else std::cout<<"Problem only one axe of rotation for 2D Grains";
//}
//void OrientationEulerAngle<2>::setAngle(Vec<F64> vangle)
//{
//    if((int)vangle.size()!=1)
//    {
//        std::cerr<<"One angle for the grain orientation for DIM=2"<<std::endl;
//    }
//    this->setAngle_ei(vangle[0], 0);
//}
//void OrientationEulerAngle<2>::randomAngle()
//{
//    DistributionUniformReal uni(-pop::PI,pop::PI);
//    setAngle_ei(uni.randomVariable(),0);
//}
//VecN<2, F64> OrientationEulerAngle<2>::inverseRotation(const VecN<2, F64> & x)
//{
//    return M_minus1*x;
//}
//void OrientationEulerAngle<2>::setAngle(const OrientationEulerAngle<2> * grain)
//{
//    setAngle_ei(grain->angle,0);
//}

GrainEquilateralRhombohedron::GrainEquilateralRhombohedron()
{
    angleequi = 15*pop::PI/180;
    setAnglePlane(angleequi);
}
void GrainEquilateralRhombohedron::setAnglePlane(F64 angleradian)
{
    angleequi = angleradian;
    cosangle = std::cos(angleequi);
    normalplanx[0]=(std::cos(angleequi));normalplanx[1]=(0);normalplanx[2]=(-std::sin(angleequi));
    normalplany[0]=(-std::sin(angleequi));normalplany[1]=(std::cos(angleequi));normalplany[2]=(0);
    normalplanz[0]=(0);normalplanz[1]=(-std::sin(angleequi));normalplanz[2]=(std::cos(angleequi));
}
F64 GrainEquilateralRhombohedron::getRadiusBallNorm0IncludingGrain(){
    return 2*radius/cosangle;
}
bool GrainEquilateralRhombohedron::intersectionPoint(const VecN<3,F64> &  x_value)
{
    VecN<3,F64> p = this->x -x_value;
    p = this->orientation.inverseRotation(p);

    F64 signe;
    signe = (productInner(normalplanx,p)+this->radius)*(productInner(normalplanx,p)-this->radius);
    if(signe>=0)
    {
        return false;
    }
    else
    {
        signe = (productInner(normalplany,p)+this->radius)*(productInner(normalplany,p)-this->radius);
        if(signe>=0)
            return false;
        else
        {
            signe = (productInner(normalplanz,p)+this->radius)*(productInner(normalplanz,p)-this->radius);
            if(signe>=0)return false;
            else return true;
        }
    }
}

Germ<3> * GrainEquilateralRhombohedron::clone()const
{
    return new GrainEquilateralRhombohedron(*this);
}
}

