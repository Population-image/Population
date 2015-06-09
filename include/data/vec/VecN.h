/******************************************************************************\
|*                   Population library for C++ X.X.X                         *|
|*----------------------------------------------------------------------------*|
The Population License is similar to the MIT license in adding this clause:
for any writing public or private that has  Typeed from the use of the
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
public or private that has  Typeed from the use of the software population,
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

#ifndef VecN_HPP
#define VecN_HPP

#include<cmath>
#include <iostream>
#include <sstream>
#include"data/utility/BasicUtility.h"
#include"data/vec/Vec.h"
#include"data/typeF/TypeTraitsF.h"
namespace pop
{
/*! \ingroup Vector
* \defgroup VecN Vec{2,3}{I32,F32}
* \brief template class for tuple of numbers of fixed size for representing a point or a vector
*/
template <int D=2, class  CoordinateType=int>
class  POP_EXPORTS VecN
{
    /*!
        \class pop::VecN
        \brief tuple of numbers of fixed  for the representation of a point or a vector as element position of a matrix
        \author Tariel Vincent
        \ingroup VecN
        \tparam D Space dimension (size of the vector)
        \tparam  CoordinateType  element type

        The template class \a VecN is a fixed-size arrays of consecutive elements of the same type taking two template parameters:
         - one constant value to set the array size
         - one type of the consecutive elements.

       For instance with a constant value equal 2 and a type equal to integer type, it allows to access of an element of a matrix.
       We have x=(x(0),x(1))=(i,j) where
      the first one represents the vertical from top to down and the second one represents the horizontal from left to right .\n

        \image html PointData.png

       To facilitate its utilisation, we use some typedef declarations to define the usual types to allow coding in C-style as these ones:
        -  Vec2I32: an integer pair (x(0),x(1)) \f$\in\mathbf{Z}^2\f$
        -  Vec3I32: an integer triplet (x(0),x(1),x(2)) \f$\in\mathbf{Z}^3\f$
        -  Vec2F32: a float pair (x(0),x(1)) \f$\in\mathbf{R}^2\f$
        -  Vec3F32: a float triplet (x(0),x(1),x(2)) \f$\in\mathbf{R}^3\f$




        For instance for a 2d grey-level matrix, this code: \n
        \code
        Mat2UI8 img(3,3);
        Vec2I32 x;
        x(0)=1;x(1)=1;//x=(1,1)
        img(x)=255;
        std::cout<<img;
        \endcode
        produce this output\n
        0 0 0\n
        0 255 0\n
        0 0 0\n
        \n\n
        A fixed-size array allows code optimization with a specialization for D=2 and D=3.
    */
private:
    CoordinateType _dat[D];
public:
    /*! \var DIM
     * Space dimension
     */
    enum{DIM=D};
    /*!
    \typedef E
    * index type to access elements
    */
    typedef int E;
    /*!
    \typedef F
    * type of the consecutive elements
    */
    typedef  CoordinateType F;

    /*!
    *
    * default constructor
    */
    VecN()
    {
        for(int i = 0;i<D;i++)
            this->_dat[i]=CoordinateType();
    }

    /*!
    * \param p copied VecN
    *
    * copy constructor
    */
    template < int D1,class G>
    VecN(const VecN<D1,G> & p )
    {
        for(int i = 0;i<minimum(D,D1);i++)
            this->_dat[i]=static_cast<CoordinateType>(p[i]);
    }
    /*!
    * \param value input value
    *
    * all elements are equal to this value
    */
    VecN(  CoordinateType  value )
    {
        for(int i = 0;i<D;i++)
            this->_dat[i]=value;
    }
    /*!
    * \param value0 input value for the first channel
    * \param value1 input value for the second channel
    *
    */
    VecN(  CoordinateType  value0,CoordinateType  value1 )
    {
        this->_dat[0]=value0;
        this->_dat[1]=value1;
    }
    /*!
    * \param value0 input value for the first channel
    * \param value1 input value for the second channel
    * \param value2 input value for the third channel
    *
    */
    VecN(  CoordinateType  value0,CoordinateType  value1,CoordinateType  value2 )
    {
        this->_dat[0]=value0;
        this->_dat[1]=value1;
        this->_dat[2]=value2;
    }
    /*!
    * \param value0 input value for the first channel
    * \param value1 input value for the second channel
    * \param value2 input value for the third channel
    * \param value3 input value for the fourth channel
    *
    */
    VecN(  CoordinateType  value0,CoordinateType  value1,CoordinateType  value2,CoordinateType value3 )
    {
        this->_dat[0]=value0;
        this->_dat[1]=value1;
        this->_dat[2]=value2;
        this->_dat[3]=value3;
    }
    /*!
    * \param x input Vec
    *
    * constructor with a vector
    * \sa Vec
    */
    template < class G>
    VecN(const Vec<G>& x){
        POP_DbgAssert( x.size()==DIM);
        for(int i=0;i<D;i++)
            this->_dat[i]=x(i);
    }
    /*!
    * \param x other VecN
    * \return this VecN
    *
    * Basic assignment of this VecN by \a other
    */
    template < class G>
    inline VecN  &operator =(const VecN<D,G>& x)
    {
        for(int i = 0;i<D;i++)
            this->_dat[i]=static_cast<CoordinateType>(x[i]);
        return *this;
    }
    /*!
    * \param value factor
    * \return this VecN
    *
    * Basic assignment of all elements of VecN by the factor \a value
    */
    inline VecN  &operator =(  CoordinateType value)
    {

        for(int i = 0;i<D;i++)
            this->_dat[i]=value;
        return *this;
    }
    /*!
    * \param x other VecN
    * \return this VecN
    *
    * Adds the contents of \a other to this VecN.
    */
    inline VecN & operator +=(const VecN & x)
    {
        for(int i = 0;i<D;i++)
            this->_dat[i]+=x[i];
        return *this;
    }
    /*!
    * \param value value
    * \return this VecN
    *
    * Adds all element of this VecN of this VecN by the factor \a value
    */
    inline VecN & operator +=( CoordinateType value)
    {
        for(int i = 0;i<D;i++)
            this->_dat[i]+=value;
        return *this;
    }
    /*!
    * \param x other VecN
    * \return output VecN
    *
    * addition of this VecNs by the other VecN
    */
    inline VecN operator+(const VecN&  x)const{
        VecN  u(*this);
        u+=x;
        return u;
    }
    /*!
    * \param value factor
    * \return output VecN
    *
    * addition of the factor \a value to all elements of this VecN
    */
    inline VecN  operator+( CoordinateType value)const{
        VecN  x(*this);
        x+=value;
        return x;
    }
    /*!
    * \param x other VecN
    * \return this VecN
    *
    * Subtracts the contents of \a other to this VecN.
    */
    inline VecN & operator -=(const VecN& x)
    {
        for(int i = 0;i<D;i++)
            this->_dat[i]-=x[i];
        return *this;
    }
    /*!
    * \param value factor
    * \return this VecN
    *
    * Subtracts all element of this VecN of this VecN by the factor \a value
    */
    inline VecN & operator -=( CoordinateType value)
    {
        for(int i = 0;i<D;i++)
            this->_dat[i]-=value;
        return *this;
    }
    /*!
    * \param x other VecN
    * \return output VecN
    *
    * subtraction of this VecN \a u by the other VecN \a v
    */
    inline VecN   operator-(const VecN&  x)const{
        VecN  u(*this);
        u-=x;
        return u;
    }
    /*!
    * \param value factor
    * \return output VecN
    *
    * Subtraction  to all elements of this VecN by the factor \a value  \f$(u_1-value,u_2-value)  \f$
    */
    inline VecN   operator-( CoordinateType  value)const{
        VecN  x(*this);
        x-=value;
        return x;
    }
    /*!
    * \return minus
    *
    * minus operator
    */
    inline VecN   operator-()const{
        VecN  x;
        x-=*this;
        return x;
    }

    /*!
    * \param x other VecN
    * \return this VecN
    *
    * Multiply the contents of \a other to this VecN.
    */
    inline VecN &  operator *=(const VecN&  x){
        for(int i = 0;i<D;i++)
            this->_dat[i]*=x[i];
        return *this;
    }
    /*!
    * \param value factor
    * \return this VecN
    *
    * Multiplies all elements of this VecN by factor.
    */
    inline VecN & operator *=( CoordinateType value)
    {
        for(int i = 0;i<D;i++)
            this->_dat[i]*=value;
        return *this;
    }
    /*!
    * \param x other VecN
    * \return this VecN
    *
    * Multiplies term by term
    * \sa productInner(const pop::VecN<D, Type1>&  x1,const pop::VecN<D, Type2>&  x2)
    */
    inline VecN   operator *(const VecN&  x)const{
        VecN  u(*this);
        u*=x;
        return u;
    }
    /*!
    * \param value factor
    * \return output VecN
    *
    * multiplication of the factor \a value to all elements of this VecN
    */
    inline VecN  operator*( CoordinateType  value)const{
        VecN  x(*this);
        x*=value;
        return x;
    }

    /*!
    * \param x other VecN
    * \return this VecN
    *
    * Divide term by term the contents of \a other to this VecN.
    */
    inline VecN &  operator /=(const VecN&  x){
        for(int i = 0;i<D;i++)
            this->_dat[i]/=x[i];
        return *this;
    }

    /*!
    * \param value factor
    * \return this VecN
    *
    * Divides all elements of this VecN by factor \a value
    */

    inline VecN & operator /=(  CoordinateType value)
    {
        for(int i = 0;i<D;i++)
            this->_dat[i]/=value;
        return *this;
    }
    /*!
    * \param x other VecN
    * \return this VecN
    *
    * Divide term by term the contents of \a other to this VecN.
    */
    inline VecN   operator /(const VecN&  x)const{
        VecN  u(*this);
        u/=x;
        return u;
    }


    /*!
    * \param value factor
    * \return output VecN
    *
    * division by the factor \a value to all elements of this VecN \a  \f$(u_1/value,u_2/value) \f$
    */
    inline VecN   operator/( CoordinateType value)const{
        VecN  x(*this);
        x/=value;
        return x;
    }
    /*!
    * \param v VecN
    * \param value factor
    * \return output VecN
    *
    * division of the factor \a value to all elements of the VecN \a  \f$(value/v_1,value/v_2) \f$
    */
    template<int D1,class  Type1>
    friend inline VecN<D1, Type1>   operator/(  Type1  value,const VecN<D1, Type1>&  v);

    /*!
    * \param x other VecN
    * \return boolean
    *
    * return true for each elements of this VecN is strictly greater or equal than to the element of same index of the other VecN, false otherwise
    */
    bool  allSuperiorEqual(const VecN& x)const
    {
        for(int i = 0;i<D;i++)
            if(this->_dat[i]<x[i]) return false;
        return true;
    }
    /*!
    * \param x other VecN
    * \return boolean
    *
    * relation order, return true if (it exists i>=0 this[i]>x[i] and for all 0<=j<i this[j]=x[j]) or (forall i this[i]=x[i]), false otherwise
    */
    bool  operator >=(const VecN& x)const
    {
        for(int i = 0;i<D;i++){
            if(this->_dat[i]>x[i])
                return true;
            else if(this->_dat[i]<x[i])
                return false;
        }
        return true;
    }
    /*!
    * \param x other VecN
    * \return boolean
    *
    * return true for each elements of this VecN is strictly less or equal than to the element of same index of the other VecN, false otherwise
    */
    inline bool allInferiorEqual(const VecN& x)const
    {
        for(int i = 0;i<D;i++)
            if(this->_dat[i]>x[i]) return false;
        return true;
    }
    /*!
    * \param x other VecN
    * \return boolean
    *
    * relation order, return true if it exists i>=0 this[i]<x[i] and for all 0<=j<i this[j]=x[j] or (forall i this[i]=x[i]), false otherwise
    */
    inline bool operator<=(const VecN& x)const
    {
        for(int i = 0;i<D;i++){
            if(this->_dat[i]<x[i])
                return true;
            else if(this->_dat[i]>x[i])
                return false;
        }
        return true;
    }
    /*!
    * \param x other VecN
    * \return boolean
    *
    * return true for each elements of this VecN is strictly greater than to the element of same index of the other VecN, false otherwise
    */
    inline bool allSuperior(const VecN& x)const
    {
        for(int i = 0;i<D;i++)
            if(this->_dat[i]<=x[i]) return false;
        return true;
    }

    /*!
     * \param x other VecN
     * \return boolean
     *
     * relation order, return true if it exists i>=0 this[i]>x[i] and for all 0<=j<i this[j]=x[j], false otherwise
    */
    inline bool operator>(const VecN& x)const
    {
        for(int i = 0;i<D;i++){
            if(this->_dat[i]>x[i])
                return true;
            else if(this->_dat[i]<x[i])
                return false;
        }
        return false;
    }
    /*!
    * \param x other VecN
    * \return boolean
    *
    * return true for each elements of this VecN is strictly less than to the element of same index of the other VecN, false otherwise
    */
    inline bool allInferior(const VecN& x)const
    {
        for(int i = 0;i<D;i++)
            if(this->_dat[i]>=x[i]) return false;
        return true;
    }

    /*!
     * \param x other VecN
     * \return boolean
     *
     * relation order, return true if it exists i>=0 this[i]<x[i] and for all 0<=j<i this[j]=x[j], false otherwise
    */
    inline bool operator<(const VecN& x)const
    {
        for(int i = 0;i<D;i++){
            if(this->_dat[i]<x[i])
                return true;
            else if(this->_dat[i]>x[i])
                return false;
        }
        return false;
    }

    /*!
    * \param x other VecN
    * \return boolean
    *
    * return true for each elements of this VecN is equal to the element of same index of the other VecN, false otherwise
    */
    inline bool operator ==(const VecN& x)const
    {
        for(int i = 0;i<D;i++)
            if(this->operator ()(i)!=x[i]) return false;
        return true;
    }
    /*!
    * \param value factor
    * \return boolean
    *
    * return true for all elements of this VecN is equal to the factor, false otherwise
    */
    inline bool operator ==(  CoordinateType value)const
    {
        for(int i = 0;i<D;i++)
            if(this->_dat[i]!=value) return false;
        return true;
    }
    /*!
    * \param x other VecN
    * \return boolean
    *
    * return true for at least one elements of this VecN is different to the element of same index of the other VecN, false otherwise
    */
    inline bool operator !=(const VecN& x)const
    {
        return !this->operator ==(x);
    }

    inline const  CoordinateType & operator []( int i)
    const
    {
        POP_DbgAssert( i>=0&& i<DIM);
        return   this->_dat[i];
    }
    /*!
    * \param i index
    * \return element
    *
    * Access to the element at the given index
    */
    inline   CoordinateType & operator []( int i)

    {
        POP_DbgAssert( i>=0&& i<DIM);
        return   this->_dat[i];
    }
    inline const  CoordinateType & operator ()( int i)
    const
    {
        POP_DbgAssert( i>=0&& i<DIM);
        return   this->_dat[i];
    }
    /*!
    * \param i index
    * \return element
    *
    * Access to the element at the given index
    */
    inline   CoordinateType & operator ()( int i)

    {
        POP_DbgAssert( i>=0&& i<DIM);
        return   this->_dat[i];
    }
    /*!
    * \return coordinate multiplication
    *
    * Multiplies all elements of this VecNs for instance x=(5,6,20) x.multCoordinate()=5*6*20=600
    */
    inline  CoordinateType multCoordinate()const
    {
        CoordinateType value=1;
        for(int i = 0;i<D;i++)
            value*=this->_dat[i];
        return value;
    }
    /*!
    \ param p norm
    * \return p-norm
    *
    * returns the p-norm of this vector for instance for x = (7,-2), we have x.norm(1)=abs(7)+abs(-2)=9
    *
    */
    F32 norm(int p=2)const
    {
        F32 value =0;
        if(p==0){
            for(int i = 0;i<D;i++)
            {
                value=maximum(value,(F32)normValue(this->_dat[i]));
            }

        }else if(p==1){
            for(int i = 0;i<D;i++)
            {
                value+=absolute(this->_dat[i]);
            }
        }else if(p==2){
            for(int i = 0;i<D;i++)
            {
                value+=this->_dat[i]*this->_dat[i];
            }
            value = std::sqrt(value);
        }else{
            for(int i = 0;i<D;i++)
            {
                value+=std::pow(static_cast<F32>(absolute(this->_dat[i])),static_cast<F32>(p));
            }
            value = std::pow(value,1.f/p);
        }
        return value;
    }

    /*!
    * \return p-norm power p
    *
    * returns the p-norm at power 2pof this VecN for instance for x = (7,-2), we have x.normPower(2)=7*7+-2*-2=53
    *
    */
    F32 normPower(int p=2)const
    {
        F32 value =0;
        if(p==0){
            for(int i = 0;i<D;i++)
            {
                value=maximum(value,(F32)normValue(this->_dat[i]));
            }

        }else if(p==1){
            for(int i = 0;i<D;i++)
            {
                value+=absolute(this->_dat[i]);
            }
        }else if(p==2){
            for(int i = 0;i<D;i++)
            {
                value+=this->_dat[i]*this->_dat[i];
            }
        }else{
            for(int i = 0;i<D;i++)
            {
                value+=std::pow(absolute(this->_dat[i]),static_cast<F32>(p));
            }
        }
        return value;
    }

    /*!
    * \return min coordinate
    *
    * returns the min coordinate for  x = (7,-2), we have x.minCoordinate()=min(7,-2)=-2
    *
    */
    CoordinateType minCoordinate()const
    {
        CoordinateType value =this->_dat[0];
        for(int i = 1;i<D;i++)
            value=minimum(this->_dat[i],value);
        return value;
    }
    /*!
    * \return min coordinate
    *
    * returns the min coordinate for  x = (7,-2), we have x.minCoordinate()=max(7,-2)=7
    *
    */
    CoordinateType maxCoordinate()const
    {
        CoordinateType value =this->_dat[0];
        for(int i = 1;i<D;i++)
            value=maximum(this->_dat[i],value);
        return value;
    }

    /*!
    * \param file input file
    *
    * load the values elements from the input file
    */
    void load(const char * file)
    {
        std::ifstream fs(file);
        if (!fs.fail())
            fs>>*this;
        else
            std::cerr<<"In VecN::load, cannot open this file"+std::string(file);
        fs.close();
    }
    /*!
    * \param file input file
    *
    * save the values elements to the input file
    */
    void save(const char * file)const
    {
        std::ofstream fs(file);
        fs<<*this;
        fs.close();
    }
    /*!
    *
    * clone the VecN
    */
    VecN*  clone()const
    {
        return new VecN(*this);
    }

    VecN<DIM+1,CoordinateType> addCoordinate(int coordinate, CoordinateType value ){
        POP_DbgAssert( coordinate>=0&& coordinate<=DIM);
        VecN<DIM+1,CoordinateType> x;
        for(int i = 0;i<=D;i++){
            if(i<coordinate)
                x(i)=this->_dat[i];
            else if(i==coordinate)
                x(i)= value;
            else
                x(i)= this->_dat[i-1];;
        }
        return x;
    }
    VecN<DIM-1,CoordinateType> removeCoordinate(int coordinate){
        POP_DbgAssert( coordinate>=0&& coordinate<DIM);
        VecN<DIM-1,CoordinateType> x;
        for(int i = 0;i<D;i++){
            if(i<coordinate)
                x(i)=this->_dat[i];
            else if(i>coordinate)
                x(i-1)= this->_dat[i];
        }
        return x;
    }

#ifdef HAVE_SWIG
    void setValue(int index,  CoordinateType value){
        _dat[index]=value;
    }
    CoordinateType getValue(int index)const{
        return _dat[index];
    }

    VecN(const VecN & p )
    {
        for(int i = 0;i<D;i++)
            this->_dat[i]=p[i];
    }
#endif
    void display(){
        std::cout<<*this<<std::endl;
    }
    /*!
    * Return a ptr to the first vector value
    *
    *
    * direct access to the vector data
    */
    CoordinateType * data(){
        return _dat;
    }
    /*!
    * Return a ptr to the first vector value
    *
    *
    * direct access to the vector data
    */
    const CoordinateType * data()const{
        return _dat;
    }
};


/*!
* \param x  input vector
* \param value factor
* \return output vector
*
* addition of the factor \a value to all elements of the VecN \a u
*/
template<int D,class  Type1>
inline VecN<D, Type1>   operator+( Type1  value,const VecN<D, Type1>&  x)
{
    VecN<D, Type1>  xt(value);
    xt+=x;
    return xt;
}
/*!

* \param x  input vector
* \param value factor
* \return output VecN
*
* Subtraction of the factor \a value to all elements of the VecN \a  \f$(value-u_1,value-u_2) \f$
*/
template<int D,class  Type1>
inline VecN<D, Type1>   operator-( Type1  value,const VecN<D, Type1>&  x)
{
    VecN<D, Type1>  xt(value);
    xt-=x;
    return xt;
}

template<int D,class  Type1>
inline VecN<D, Type1>   operator*( Type1  value,const VecN<D, Type1>&  x)
{
    VecN<D, Type1>  xt (value);
    xt*=x;
    return xt;
}
template<int D,class  Type1>
inline VecN<D, Type1>   operator/( Type1  value,const VecN<D, Type1>&  x)
{
    VecN<D, Type1>  xt (value);
    xt/=x;
    return xt;
}

typedef VecN<2,I32> Vec2I32;
typedef VecN<3,I32> Vec3I32;
typedef VecN<4,I32> Vec4I32;
typedef VecN<2,F32> Vec2F32;
typedef VecN<2,F64> Vec2F64;
typedef VecN<3,F32> Vec3F32;
typedef VecN<4,F32> Vec4F32;





template<typename TypeIn,typename TypeOut,int DIM>
struct FunctionTypeTraitsSubstituteF<VecN<DIM,TypeIn>,TypeOut>
{
    typedef VecN<DIM,typename FunctionTypeTraitsSubstituteF<TypeIn,TypeOut>::Result>  Result;
};




template<int DIM,typename Type>
struct isVectoriel<VecN<DIM,Type> >{
    enum { value =true};
};

template<int DIM, typename R, typename T>
struct ArithmeticsSaturation<VecN<DIM,R>,VecN<DIM,T> >
{
    static VecN<DIM,R> Range(VecN<DIM,T> p)
    {
        VecN<DIM,R> r;
        for(int i =0;i<DIM;i++)
            if(p(i)>=NumericLimits<R>::maximumRange())r(i) =NumericLimits<R>::maximumRange();
            else if(p(i)<NumericLimits<R>::minimumRange())r(i) = NumericLimits<R>::minimumRange();
            else r(i) =p(i);
        return r;
    }
};
template<int DIM, typename T>
struct ArithmeticsSaturation<UI8,VecN<DIM,T> >
{
    static UI8 Range(VecN<DIM,T> p)
    {
        T t = p.norm();
        if(t>=NumericLimits<UI8>::maximumRange())return NumericLimits<UI8>::maximumRange();
        else if(t<NumericLimits<UI8>::minimumRange())return NumericLimits<UI8>::minimumRange();
        else return t;
    }
};
template<int DIM, typename T>
struct ArithmeticsSaturation<UI16,VecN<DIM,T> >
{
    static UI16 Range(VecN<DIM,T> p)
    {
        T t = p.norm();
        if(t>=NumericLimits<UI16>::maximumRange())return NumericLimits<UI16>::maximumRange();
        else if(t<NumericLimits<UI16>::minimumRange())return NumericLimits<UI16>::minimumRange();
        else return t;
    }
};
template<int DIM, typename T>
struct ArithmeticsSaturation<UI32,VecN<DIM,T> >
{
    static UI32 Range(VecN<DIM,T> p)
    {
        T t = p.norm();
        if(t>=NumericLimits<UI32>::maximumRange())return NumericLimits<UI32>::maximumRange();
        else if(t<NumericLimits<UI32>::minimumRange())return NumericLimits<UI32>::minimumRange();
        else return t;
    }
};
template<int DIM, typename T>
struct ArithmeticsSaturation<F32,VecN<DIM,T> >
{
    static F32 Range(VecN<DIM,T> p)
    {
        T t = p.norm();
        if(t>=NumericLimits<F32>::maximumRange())return NumericLimits<F32>::maximumRange();
        else if(t<NumericLimits<F32>::minimumRange())return NumericLimits<F32>::minimumRange();
        else return t;
    }
};

template<class Type>
class RGB;
template< int DIM, typename R, typename T>
struct ArithmeticsSaturation<VecN<DIM,R>,RGB<T> >
{
    static VecN<DIM,R> Range(RGB<T> p)
    {
        return VecN<DIM,R>(p.lumi());
    }
};
template<class Type>
class Complex;
template< int DIM, typename R, typename T>
struct ArithmeticsSaturation<VecN<DIM,R>,Complex<T> >
{
    static VecN<DIM,R> Range(Complex<T> p)
    {
        return VecN<DIM,R>(p.norm());
    }
};
template<int DIM, typename R,typename T>
struct ArithmeticsSaturation<VecN<DIM,R>,T >
{
    static VecN<DIM,T> Range(R p)
    {
        return VecN<DIM,T>(ArithmeticsSaturation<R,T >::Range(p));
    }
};
template<int DIM,typename Type>
struct NumericLimits<pop::VecN<DIM,Type> >
{
    static const bool is_specialized = true;

    static pop::VecN<DIM,Type> minimumRange() throw()
    { return pop::VecN<DIM,Type>(NumericLimits<Type >::minimumRange());}
    static pop::VecN<DIM,Type> maximumRange() throw()
    { return pop::VecN<DIM,Type>(NumericLimits<Type >::maximumRange()); }
    static const bool is_integer = NumericLimits<Type>::is_integer;
};

/*!
* \ingroup  VecN
* \brief maximum of  VecN \a u by the VecN \a v \f$\max(u,v)=(\max(u_0,v_0),\max(u_1,v_1))\f$ for each coordinate
* \param p1 first VecN
* \param p2 second VecN
* \return output VecN
*
* \code
* pop::Vec2F32 x1(2.5,4.5);
* pop::Vec2F32 x2(6.3,2.5);
* pop::Vec2F32 x = maximum(x1,x2);//x=(6.3,2.5)
* \endcode
*
*/
template<int DIM,typename Type1>
inline pop::VecN<DIM,Type1> maximum(const pop::VecN<DIM,Type1>&  p1, const pop::VecN<DIM,Type1> &  p2)
{
    pop::VecN<DIM,Type1> _max;
    for(int i =0;i<DIM;i++)
        _max(i)=maximum(p1(i),p2(i));
    return _max;
}

template<typename Type1>
pop::VecN<2,Type1> maximum(const pop::VecN<2,Type1>&  p1, const pop::VecN<2,Type1> &  p2)
{
    return pop::VecN<2,Type1>(maximum(p1(0),p2(0)),maximum(p1(1),p2(1)));
}
template<typename Type1>
pop::VecN<3,Type1> maximum(const pop::VecN<3,Type1>&  p1, const pop::VecN<3,Type1> &  p2)
{
    return pop::VecN<3,Type1>(maximum(p1(0),p2(0)),maximum(p1(1),p2(1)),maximum(p1(2),p2(2)));
}
template<typename Type1>
pop::VecN<4,Type1> maximum(const pop::VecN<4,Type1>&  p1, const pop::VecN<4,Type1> &  p2)
{
    return pop::VecN<4,Type1>(maximum(p1(0),p2(0)),maximum(p1(1),p2(1)),maximum(p1(2),p2(2)),maximum(p1(3),p2(3)));
}
/*!
* \ingroup  VecN
* \brief minimum of  VecN \a u by the VecN \a v \f$\min(u,v)=(\min(u_0,v_0),\min(u_1,v_1))\f$ for each coordinate
* \param p1 first VecN
* \param p2 second VecN
* \return output VecN
*
*
* \code
* pop::Vec2F32 x1(2.5,4.5);
* pop::Vec2F32 x2(6.3,2.5);
* pop::Vec2F32 x = minimum(x1,x2);//x=(2.5,2.5)
* \endcode
*/
template<int DIM,typename Type1>
pop::VecN<DIM,Type1> minimum(const pop::VecN<DIM,Type1>&  p1, const pop::VecN<DIM,Type1> &  p2)
{
    pop::VecN<DIM,Type1> _min;
    for(int i =0;i<DIM;i++)
        _min(i)=minimum(p1(i),p2(i));
    return _min;
}
template<typename Type1>
pop::VecN<2,Type1> minimum(const pop::VecN<2,Type1>&  p1, const pop::VecN<2,Type1> &  p2)
{
    return pop::VecN<2,Type1>(minimum(p1(0),p2(0)),minimum(p1(1),p2(1)));
}
template<typename Type1>
pop::VecN<3,Type1> minimum(const pop::VecN<3,Type1>&  p1, const pop::VecN<3,Type1> &  p2)
{
    return pop::VecN<3,Type1>(minimum(p1(0),p2(0)),minimum(p1(1),p2(1)),minimum(p1(2),p2(2)));
}
template<typename Type1>
pop::VecN<4,Type1> minimum(const pop::VecN<4,Type1>&  p1, const pop::VecN<4,Type1> &  p2)
{
    return pop::VecN<4,Type1>(minimum(p1(0),p2(0)),minimum(p1(1),p2(1)),minimum(p1(2),p2(2)),minimum(p1(3),p2(3)));
}

/*!
* \ingroup  VecN
* \brief absolute value of the  VecN abs(u)=(abs(u_0),abs(u_1)) for each coordinate
* \param p1  VecN
* \return output VecN
*
* \code
* pop::Vec2F32 x1(2.5,-4.5);
* pop::Vec2F32 x = absolute(x1);//x=(2.5,2.5)
* \endcode
*/
template<int DIM,typename Type1>
inline pop::VecN<DIM,Type1> absolute(const pop::VecN<DIM,Type1>&  p1)
{
    pop::VecN<DIM,Type1> _abs;
    for(int i =0;i<DIM;i++)
        _abs(i)=absolute(p1(i));
    return _abs;
}
template<typename Type1>
pop::VecN<2,Type1> absolute(const pop::VecN<2,Type1>&  p1)
{
    return pop::VecN<2,Type1>(absolute(p1(0)),absolute(p1(1)));
}
template<typename Type1>
pop::VecN<3,Type1> absolute(const pop::VecN<3,Type1>&  p1)
{
    return pop::VecN<3,Type1>(absolute(p1(0)),absolute(p1(1)),absolute(p1(2)));
}
template<typename Type1>
pop::VecN<4,Type1> absolute(const pop::VecN<4,Type1>&  p1)
{
    return pop::VecN<4,Type1>(absolute(p1(0)),absolute(p1(1)),absolute(p1(2)),absolute(p1(3)));
}

/*!
* \ingroup  VecN
* \brief  floor value of the  VecN for each coordinate
* \param p1  VecN
* \return output VecN
*
*
*/
template<int DIM,typename Type1>
inline pop::VecN<DIM,Type1> floor(const pop::VecN<DIM,Type1>&  p1)
{
    pop::VecN<DIM,Type1> _abs;
    for(int i =0;i<DIM;i++)
        _abs(i)=std::floor(p1(i));
    return _abs;
}
template<typename Type1>
pop::VecN<2,Type1> floor(const pop::VecN<2,Type1>&  p1)
{
    return pop::VecN<2,Type1>(std::floor(p1(0)),std::floor(p1(1)));
}
template<typename Type1>
pop::VecN<3,Type1> floor(const pop::VecN<3,Type1>&  p1)
{
    return pop::VecN<3,Type1>(std::floor(p1(0)),std::floor(p1(1)),std::floor(p1(2)));
}
template<typename Type1>
pop::VecN<4,Type1> floor(const pop::VecN<4,Type1>&  p1)
{
    return pop::VecN<4,Type1>(std::floor(p1(0)),std::floor(p1(1)),std::floor(p1(2)),std::floor(p1(3)));
}

/*!
* \ingroup VecN
* \brief norm of the VecN \f$\vert u \vert_p=(\sum|u_i|^p)^{1/p}\f$
* \param x input vector
* \param p p-norm
* \return output scalar value
*
*/
template<int DIM,typename Type1>
F32 normValue(const pop::VecN<DIM,Type1>&  x, int p=2)
{
    return x.norm(p);
}
/*!
* \ingroup VecN
* \brief distance between two vectors \f$\vert u-v \vert^p\f$
* \param u  VecN
* \param v  VecN
* \param p  p-norm
* \return output scalar value
*
*/
template<int DIM,typename Type1>
F32 distance(const pop::VecN<DIM,Type1>&  u, const pop::VecN<DIM,Type1>&  v,int p=2)
{
    return normValue(u-v,p);
}
/*!
* \ingroup VecN
* \brief norm of the VecN \f$\vert u \vert_p=\sum|u_i|^p\f$
* \param p1 input vector
* \param p p-norm
* \return output scalar value
*
*/
template<int DIM,typename Type1>
F32 normPowerValue(const pop::VecN<DIM,Type1>&  p1, int p=2)
{
    return p1.normPower(p);
}
/*!
* \ingroup VecN
* \brief  round functions return the integral value nearest to v1 rounding half-way cases away for each coordinate
* \param v1  VecN
* \return output VecN
*
*/

template<int DIM,typename Type1>
pop::VecN<DIM,Type1> round(const pop::VecN<DIM,Type1>& v1){
    pop::VecN<DIM,Type1> round;
    for(int i=0;i<DIM;i++)
        round(i)= round(v1(i));
    return round;
}
template<typename Type1>
pop::VecN<2,Type1> round(const pop::VecN<2,Type1>&  p1)
{
    return pop::VecN<2,Type1>(round(p1(0)),round(p1(1)));
}
template<typename Type1>
pop::VecN<3,Type1> round(const pop::VecN<3,Type1>&  p1)
{
    return pop::VecN<3,Type1>(round(p1(0)),round(p1(1)),round(p1(2)));
}
template<typename Type1>
pop::VecN<4,Type1> round(const pop::VecN<4,Type1>&  p1)
{
    return pop::VecN<4,Type1>(round(p1(0)),round(p1(1)),round(p1(2)),round(p1(3)));
}

/*!
* \ingroup VecN
* \brief inner product between two 3d vector
* \param x1 first vector
* \param x2 second vector
* \return output vector
*
* inner  product of the two vectors \f$<u |v>= \sum_i u_i v_i.\f$
*/

template<int D,class  Type1>
Type1   productInner(const pop::VecN<D, Type1>&  x1,const pop::VecN<D, Type1>&  x2)
{
    Type1 value=0;
    for(int i = 0;i<D;i++)
    {
        value+=productInner(x1[i],x2[i]);
    }
    return value;
}
template<class  Type1>
Type1   productInner(const pop::VecN<2, Type1>&  x1,const pop::VecN<2, Type1>&  x2)
{
    return productInner(x1[0],x2[0])+productInner(x1[1],x2[1]);
}
template<class  Type1>
Type1   productInner(const pop::VecN<3, Type1>&  x1,const pop::VecN<3, Type1>&  x2)
{
    return productInner(x1[0],x2[0])+productInner(x1[1],x2[1])+productInner(x1[2],x2[2]);
}
template<class  Type1>
Type1   productInner(const pop::VecN<4, Type1>&  x1,const pop::VecN<4, Type1>&  x2)
{
    return productInner(x1[0],x2[0])+productInner(x1[1],x2[1])+productInner(x1[2],x2[2])+productInner(x1[3],x2[3]);
}

/*!
* \ingroup VecN
* \brief vectoriel product between two 3d vector
* \param x1 first vector
* \param x2 second vector
* \return output vector
*
* vectoriel product of the two vectors \f$u\wedge v=\begin{pmatrix}
u_1v_2-u_2v_1\\ u_2v_0-u_0v_2\\ u_0v_1-u_1v_0
\end{pmatrix}.\f$
*/
template<class  Type1>
inline pop::VecN<3, Type1>   productVectoriel(const pop::VecN<3, Type1>&  x1,const pop::VecN<3, Type1>&  x2)
{
    pop::VecN<3, Type1>  x;
    x(0)=x1[1]*x2[2]-x1[2]*x2[1];
    x(1)=x1[2]*x2[0]-x1[0]*x2[2];
    x(2)=x1[0]*x2[1]-x1[1]*x2[0];
    return x;
}


/*!
* \ingroup VecN
* \param out output stream
* \param x VecN
*
* stream insertion from the VecN x
*/
template <int D, class  Type>
std::ostream& operator << (std::ostream& out, const pop::VecN<D, Type>& x)
{
    for(int i = 0;i<D;i++)
    {
        out<<x[i]<<"<P>";

    }
    return out;
}

/*!
* \ingroup VecN
* \param in input stream
* \param x VecN
*
* stream extraction  to the VecN x
*/
template <int D, class  Type>
std::istream& operator >> (std::istream& in,  pop::VecN<D, Type>& x)
{
    for(int i = 0;i<D;i++)
    {
        std::string str;
        str = pop::BasicUtility::getline( in, "<P>" );
        pop::BasicUtility::String2Any(str,x[i]);
    }
    return in;
}



template<int DIM>
struct VecNIndice
{
    static   int VecN2Indice(const VecN<DIM,int> & size, const VecN<DIM,int> & x)
    {
        int value=0;
        int temp=1;
        for(int i=0;i<DIM;i++)
        {
            value+=temp*x(i);
            temp*=size(i);
        }
        return value;

    }
    static   int VecN2Indice(const VecN<DIM,int> & size, const Vec<int> & x)
    {
        int value=0;
        int temp=1;
        for(int i=0;i<DIM;i++)
        {
            value+=temp*x(i);
            temp*=size(i);
        }
        return value;

    }
    static   VecN<DIM,int> Indice2VecN(const VecN<DIM,int> & size, int indice)
    {
        VecN<DIM,int> x;
        for(int i=0;i<DIM;i++)
        {
            int temp=1;
            for(int j=0;j<DIM-(i+1);j++)
                temp*=size(j);
            x(DIM-(i+1)) = indice/temp ;
            indice -= (x(DIM-(i+1)) *temp);
        }
        return x;
    }
};
template<>
struct VecNIndice<2>
{
    static   int VecN2Indice(const Vec2I32 & size, const Vec2I32 & x)
    {
        return x(1)+x(0)*size(1);
    }
    static   int VecN2Indice(const Vec2I32 & size, const Vec<int> & x)
    {
        return x(1)+x(0)*size(1);
    }
    static   Vec2I32 Indice2VecN(const Vec2I32 & size, int indice)
    {
        Vec2I32 x;
        x(0)=indice/size(1);
        x(1)=indice - x(0)*size(1);
        return x;
    }
};
template<>
struct VecNIndice<3>
{
    static   int VecN2Indice(const VecN<3,int> & size, const VecN<3,int> & x)
    {
        return x(1)+size(1)*(x(0)+size(0)*x(2));
    }
    static   int VecN2Indice(const VecN<3,int> & size, const Vec<int> & x)
    {
        return x(1)+size(1)*(x(0)+size(0)*x(2));
    }
    static   VecN<3,int> Indice2VecN(const VecN<3,int> & size, int indice)
    {
        VecN<3,int> x;
        int s =size(1)*size(0);
        x(2)=indice/s;
        indice-=x(2)*s;
        x(0)=indice/size(1);
        x(1)=indice - x(0)*size(1);
        return x;
    }
};

template<int DIM>
struct VecN2IndiceByTable
{
        Vec< int> _table;
    VecN2IndiceByTable(const VecN<DIM,int>  & =VecN<DIM,int> (0))
    {
    }

    int VecN2Indice(const VecN<DIM,int> & size, const VecN<DIM,int> & x)const
    {
        int value=0;
        int temp=1;
        for(int i=0;i<DIM;i++)
        {
            value+=temp*x(i);
            temp*=size(i);
        }
        return value;

    }
    int VecN2Indice(const VecN<DIM,int> & size, const Vec<int> & x)const
    {
        int value=0;
        int temp=1;
        for(int i=0;i<DIM;i++)
        {
            value+=temp*x(i);
            temp*=size(i);
        }
        return value;

    }
    VecN<DIM,int> Indice2VecN(const VecN<DIM,int> & size, int indice)const
    {
        VecN<DIM,int> x;
        for(int i=0;i<DIM;i++)
        {
            int temp=1;
            for(int j=0;j<DIM-(i+1);j++)
                temp*=size(j);
            x(DIM-(i+1)) = indice/temp ;
            indice -= (x(DIM-(i+1)) *temp);
        }
        return x;
    }
};




}
#endif
