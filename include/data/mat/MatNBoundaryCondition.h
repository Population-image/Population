/******************************************************************************\
|*                   Population library for C++ X.X.X                         *|
|*----------------------------------------------------------------------------*|
The Population License is similar to the MIT license in adding this clause:
for any writing public or private that has resulted from the use of the
software population, the reference of this book "Population library, 2012,
Vincent Tariel" shall be included in it.

So, the terms of the Population License are:

Copyright © 2012-2015, Tariel Vincent

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to
deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software and for any writing
public or private that has resulted from the use of the software population,
the reference of this book "Population library, 2012, Vincent Tariel" shall
be included in it.

The Software is provided "as is", without warranty of any kind, express or
implied, including but not limited to the warranties of merchantability,
fitness for a particular purpose and noninfringement. In no event shall the
authors or copyright holders be liable for any claim, damages or other
liability, whether in an action of contract, tort or otherwise, arising
from, out of or in connection with the software or the use or other dealings
in the Software.
\***************************************************************************/
#ifndef FUNCTIONMatNBOUNDARYCONDITION_HPP
#define FUNCTIONMatNBOUNDARYCONDITION_HPP
#include"PopulationConfig.h"
#include"data/vec/VecN.h"
namespace pop
{
enum  MatNBoundaryConditionType{
    MATN_BOUNDARY_CONDITION_BOUNDED  = 0,
    MATN_BOUNDARY_CONDITION_PERIODIC = 1,
    MATN_BOUNDARY_CONDITION_MIRROR   = 2
};



class POP_EXPORTS MatNBoundaryConditionBounded
{
public:
    template<typename Vec1,typename Vec2>
    static bool isValid(const Vec1 & domain,const Vec2 & x){
        if(x.allSuperiorEqual(0)&&x.allInferior(domain)){
            return true;
        }
        else{
            return false;
        }
    }
    template<typename Vec1,typename Vec2>
    static bool isValid(const Vec1 & domain,const Vec2 & x,int direction){
        if(x(direction)>=0&&x(direction)<domain(direction)){
            return true;
        }
        else{
            return false;
        }
    }
    template<typename Vec1,typename Vec2>
    static void apply(const Vec1 & ,Vec2 & )
    {

    }

};
class POP_EXPORTS MatNBoundaryConditionMirror
{
public:
    template<typename Vec1,typename Vec2>
    static bool isValid(const Vec1 & ,const Vec2 & ){
        return true;
    }
    template<typename Vec1,typename Vec2>
    static bool isValid(const Vec1 & ,const Vec2 & ,int){
        return true;
    }
    template<typename Vec1,typename Vec2>
    static void apply(const Vec1 & domain,Vec2 & x)
    {
        for(int direction=0;direction<Vec1::DIM;direction++)
        {
            apply(domain,x,direction);
        }
    }
    template<typename Vec1,typename Vec2>
    static void apply(const Vec1 & domain,Vec2 & x,int direction)
    {

        if(x(direction)<0){
            x(direction)= -x(direction)-1;
        }
        else if(x(direction)>=domain(direction))
            x(direction)= 2*domain(direction)-x(direction)-1;
    }
};
class POP_EXPORTS MatNBoundaryConditionPeriodic
{
public:
    template<typename Vec1,typename Vec2>
    static bool isValid(const Vec1 & ,const Vec2 & ){
        return true;
    }
    template<typename Vec1,typename Vec2>
    static bool isValid(const Vec1 & ,const Vec2 & ,int){
        return true;
    }
    template<typename Vec1,typename Vec2>
    static void apply(const Vec1 & domain,Vec2 & x)
    {
        for(int direction=0;direction<Vec1::DIM;direction++)
        {
            apply(domain,x,direction);
        }
    }
    template<typename Vec1,typename Vec2>
    static void apply(const Vec1 & domain,Vec2 & x,int direction)
    {
        if(x(direction)<0)
            x(direction)= domain(direction)+x(direction);
        if(x(direction)>=domain(direction))
            x(direction)= x(direction)-domain(direction);
    }
};

class POP_EXPORTS MatNBoundaryCondition
{
public:
    MatNBoundaryCondition(MatNBoundaryConditionType condition=MATN_BOUNDARY_CONDITION_BOUNDED)
        :_condition(condition)
    {}
    template<typename Vec1,typename Vec2>
    bool isValid(const Vec1 & domain,const Vec2 & x){
        switch(_condition)
        {
        case MATN_BOUNDARY_CONDITION_BOUNDED :
            return MatNBoundaryConditionBounded::isValid(domain,x);
        case MATN_BOUNDARY_CONDITION_PERIODIC :
            return MatNBoundaryConditionPeriodic::isValid(domain,x);
        case MATN_BOUNDARY_CONDITION_MIRROR :
            return MatNBoundaryConditionMirror::isValid(domain,x);
        default :
            return true;
        }
    }
    template<typename Vec1,typename Vec2>
    bool isValid(const Vec1 & domain,const Vec2 & x,int direction){
        switch(_condition)
        {
        case MATN_BOUNDARY_CONDITION_BOUNDED :
            return MatNBoundaryConditionBounded::isValid(domain,x,direction);
        case MATN_BOUNDARY_CONDITION_PERIODIC :
            return MatNBoundaryConditionPeriodic::isValid(domain,x,direction);
        case MATN_BOUNDARY_CONDITION_MIRROR :
            return MatNBoundaryConditionMirror::isValid(domain,x,direction);
        default :
            return true;
        }
    }
    template<typename Vec1,typename Vec2>
    void  apply(const Vec1 & domain,Vec2 & x){
        switch(_condition)
        {
        case MATN_BOUNDARY_CONDITION_BOUNDED :
            MatNBoundaryConditionBounded::apply(domain,x);
            break;
        case MATN_BOUNDARY_CONDITION_PERIODIC :
            MatNBoundaryConditionPeriodic::apply(domain,x);
            break;
        case MATN_BOUNDARY_CONDITION_MIRROR :
            MatNBoundaryConditionMirror::apply(domain,x);
            break;
        }
    }
    template<typename Vec1,typename Vec2>
    void  apply(const Vec1 & domain,Vec2 & x,int direction){
        switch(_condition)
        {
        case MATN_BOUNDARY_CONDITION_BOUNDED :
            MatNBoundaryConditionBounded::apply(domain,x,direction);
            break;
        case MATN_BOUNDARY_CONDITION_PERIODIC :
            MatNBoundaryConditionPeriodic::apply(domain,x,direction);
            break;
        case MATN_BOUNDARY_CONDITION_MIRROR :
            MatNBoundaryConditionMirror::apply(domain,x,direction);
            break;
        default:
            MatNBoundaryConditionPeriodic::apply(domain,x,direction);
            break;
        }
    }
private:
    MatNBoundaryConditionType _condition;
};
template<int DIM>
inline VecN<PowerGP<2,DIM>::value,std::pair<double,VecN<DIM,int> > > interpolationBilinearWeigh(const VecN<DIM,int> &x_domain ,const VecN<DIM,F64> &x_point,MatNBoundaryCondition boundary = MATN_BOUNDARY_CONDITION_BOUNDED){
    return  _interpolationBilinearWeigh(x_domain,x_point,boundary, Int2Type<DIM>()) ;
}
template<int DIM>
VecN<PowerGP<2,DIM>::value,std::pair<double,VecN<DIM,int> > > _interpolationBilinearWeigh(const VecN<DIM,int> ,const VecN<DIM,F64> &,MatNBoundaryCondition , Int2Type<DIM>)
{

}

inline VecN<4,std::pair<double,Vec2I32 > > _interpolationBilinearWeigh(const Vec2I32 x_domain,const Vec2F64 &x_point,MatNBoundaryCondition boundary, Int2Type<2>)
{
    VecN<4,std::pair<double,Vec2I32 > > v_out(std::make_pair(0,Vec2I32(0,0)));
    Vec2F64 x;
    x(0)=x_point(0)+EPSILON;
    x(1)=x_point(1)+EPSILON;
    bool all_hit=true;
    double sum=0;
    Vec2I32 x1;
    x1(0)=std::floor(x(0));
    x1(1)=std::floor(x(1));
    if(boundary.isValid(x_domain,x1)){
        boundary.apply(x_domain,x1);
        double norm = (1-(x(0)-x1(0)))*(1-(x(1)-x1(1)));
        sum+= norm;
        v_out(0).second=x1;
        v_out(0).first =norm;
    }else{
       v_out(0).second=Vec2I32(0,0);
       v_out(0).first =0;
       all_hit=false;
    }
    x1(0)=std::ceil(x(0));
    if(boundary.isValid(x_domain,x1)){
        boundary.apply(x_domain,x1);
        double norm = (1-(x1(0)-x(0)))* (1-(x(1)-x1(1)));
        sum+= norm;
        v_out(1).second=x1;
        v_out(1).first =norm;
    }else{
        v_out(1).second=Vec2I32(0,0);
        v_out(1).first =0;
        all_hit=false;
     }
    x1(1)=std::ceil(x(1));
    if(boundary.isValid(x_domain,x1)){
        boundary.apply(x_domain,x1);
        double norm = (1-(x1(0)-x(0)))*(1-(x1(1)-x(1)));
        sum+= norm;
        v_out(2).second=x1;
        v_out(2).first =norm;
    }else{
        v_out(2).second=Vec2I32(0,0);
        v_out(2).first =0;
        all_hit=false;
     }
    x1(0)=std::floor(x(0));
    if(boundary.isValid(x_domain,x1)){
        boundary.apply(x_domain,x1);
        double norm = (1-(x(0)-x1(0)))*(1-(x1(1)-x(1)));
        sum+= norm;
        v_out(3).second=x1;
        v_out(3).first =norm;
    }else{
        v_out(3).second=Vec2I32(0,0);
        v_out(3).first =0;
        all_hit=false;
     }
    if(all_hit==false)
    {
        if(sum!=0){
            v_out(0).first /=sum;v_out(1).first /=sum;v_out(2).first /=sum;v_out(3).first /=sum;
        }
    }
    return v_out;
}


inline VecN<8,std::pair<double,Vec3I32 > > _interpolationBilinearWeigh(const Vec3I32 x_domain,const Vec3F64 &x_point,MatNBoundaryCondition boundary ,Int2Type<2>)
{
    VecN<8,std::pair<double,Vec3I32 > > v_out(std::make_pair(0,Vec3I32(0,0)));
    Vec3F64 x;
    x(0)=x_point(0)+EPSILON;
    x(1)=x_point(1)+EPSILON;
    x(2)=x_point(1)+EPSILON;
    bool all_hit=true;
    double sum=0;
    Vec3I32 x1;
    x1(0)=std::floor(x(0));
    x1(1)=std::floor(x(1));
    x1(2)=std::floor(x(2));
    if(boundary.isValid(x_domain,x1)){
        boundary.apply(x_domain,x1);
        double norm = (1-(x(0)-x1(0)))*(1-(x(1)-x1(1)))*(1-(x(2)-x1(2)));
        sum+= norm;
        v_out(0).second=x1;
        v_out(0).first =norm;
    }else{
       v_out(0).second=Vec3I32(0,0,0);
       v_out(0).first =0;
       all_hit=false;
    }
    x1(0)=std::ceil(x(0));
    if(boundary.isValid(x_domain,x1)){
        boundary.apply(x_domain,x1);
        double norm = (1-(x1(0)-x(0)))* (1-(x(1)-x1(1)))*(1-(x(2)-x1(2)));
        sum+= norm;
        v_out(1).second=x1;
        v_out(1).first =norm;
    }else{
        v_out(1).second=Vec3I32(0,0,0);
        v_out(1).first =0;
        all_hit=false;
     }
    x1(1)=std::ceil(x(1));
    if(boundary.isValid(x_domain,x1)){
        boundary.apply(x_domain,x1);
        double norm = (1-(x1(0)-x(0)))*(1-(x1(1)-x(1)))*(1-(x(2)-x1(2)));
        sum+= norm;
        v_out(2).second=x1;
        v_out(2).first =norm;
    }else{
        v_out(2).second=Vec3I32(0,0,0);
        v_out(2).first =0;
        all_hit=false;
     }
    x1(0)=std::floor(x(0));
    if(boundary.isValid(x_domain,x1)){
        boundary.apply(x_domain,x1);
        double norm = (1-(x(0)-x1(0)))*(1-(x1(1)-x(1)))*(1-(x(2)-x1(2)));
        sum+= norm;
        v_out(3).second=x1;
        v_out(3).first =norm;
    }else{
        v_out(3).second=Vec3I32(0,0,0);
        v_out(3).first =0;
        all_hit=false;
     }
    x1(0)=std::floor(x(0));
    x1(1)=std::floor(x(1));
    x1(2)=std::ceil(x(2));
    if(boundary.isValid(x_domain,x1)){
        boundary.apply(x_domain,x1);
        double norm = (1-(x(0)-x1(0)))*(1-(x(1)-x1(1)))*(1-(x1(2)-x(2)));
        sum+= norm;
        v_out(4).second=x1;
        v_out(4).first =norm;
    }else{
       v_out(4).second=Vec3I32(0,0,0);
       v_out(4).first =0;
       all_hit=false;
    }
    x1(0)=std::ceil(x(0));
    if(boundary.isValid(x_domain,x1)){
        boundary.apply(x_domain,x1);
        double norm = (1-(x1(0)-x(0)))* (1-(x(1)-x1(1)))*(1-(x1(2)-x(2)));
        sum+= norm;
        v_out(5).second=x1;
        v_out(5).first =norm;
    }else{
        v_out(5).second=Vec3I32(0,0,0);
        v_out(5).first =0;
        all_hit=false;
     }
    x1(1)=std::ceil(x(1));
    if(boundary.isValid(x_domain,x1)){
        boundary.apply(x_domain,x1);
        double norm = (1-(x1(0)-x(0)))*(1-(x1(1)-x(1)))*(1-(x1(2)-x(2)));
        sum+= norm;
        v_out(6).second=x1;
        v_out(6).first =norm;
    }else{
        v_out(6).second=Vec3I32(0,0,0);
        v_out(6).first =0;
        all_hit=false;
     }
    x1(0)=std::floor(x(0));
    if(boundary.isValid(x_domain,x1)){
        boundary.apply(x_domain,x1);
        double norm = (1-(x(0)-x1(0)))*(1-(x1(1)-x(1)))*(1-(x1(2)-x(2)));
        sum+= norm;
        v_out(7).second=x1;
        v_out(7).first =norm;
    }else{
        v_out(7).second=Vec3I32(0,0,0);
        v_out(7).first =0;
        all_hit=false;
     }

    if(all_hit==false)
    {
        if(sum!=0){
            v_out(0).first /=sum;v_out(1).first /=sum;v_out(2).first /=sum;v_out(3).first /=sum;
            v_out(4).first /=sum;v_out(5).first /=sum;v_out(6).first /=sum;v_out(7).first /=sum;
        }
    }
    return v_out;
}

}
#endif // FUNCTIONMatNBOUNDARYCONDITION_HPP
