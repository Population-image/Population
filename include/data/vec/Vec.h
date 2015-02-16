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

#ifndef Vec_H
#define Vec_H
#include<vector>
#include<iostream>
#include<fstream>
#include<algorithm>
#include <numeric>
#include<cmath>
#include <functional> 
#include"data/typeF/TypeF.h"
#include"data/utility/BasicUtility.h"
#include"data/typeF/Complex.h"
#include"data/typeF/TypeTraitsF.h"
#include"data/functor/FunctorF.h"
namespace pop
{

class VecNIteratorEDomain
{
private:
    int _domain;
    int _x;

public:
    typedef int  Domain;
    VecNIteratorEDomain(const VecNIteratorEDomain& it)
        :_domain(it.getDomain())
    {
        init();
    }

    VecNIteratorEDomain(const int & domain)
        :    _domain(domain){init();}
    inline void init(){ _x=-1;}
    inline bool next(){
        _x++;
        if(_x<_domain)return true;
        else return false;
    }
    inline int & x(){ return _x;}
    inline int getDomain()const { return _domain;}
};


template<int DIM, typename CoordinateType>
class VecN;

/*! \ingroup Data
* \defgroup Vector Coordinate vector
* \brief  coordinate vector as tuple of numbers
*/


/*! \ingroup Vector
* \defgroup Vec Vec{I32,F32}
* \brief  template class for tuple of numbers of variable size
*/
template<typename CoordinateType>
class POP_EXPORTS Vec : public std::vector<CoordinateType>
{
private:
    /*!
     * \class pop::Vec
     * \ingroup Vec
     * \brief tuple of numbers of variable size for the representation of the coordinate vector
     * \author Tariel Vincent
     * \tparam CoordinateType number type
     *
     * Vec is an extension of the std::vector to include classical arithmetic operations of linear algebra.
     * \code
        VecF32 v1(3),v2(3);//VecF32=Vec<F32>
        v1(0)= 1;v1(1)= 0.5;v1(2)=3.5;
        v2(0)=-1;v2(1)=-0.5;v2(2)=2.5;
        v1 = (v1+v2)/normValue(v1+v2);//add two vector and normalized by the norm
        std::cout<<v1<<std::endl;//0,0,1
        Mat2F32 m =GeometricalTransformation::rotation3D(PI/2,1);//rotation along the y-axis
        v1 = m*v1;//multiply the matrix by the vector
        std::cout<<v1<<std::endl;//-1,0,0
     * \endcode
     * \sa pop::LinearAlgebra pop::GeometricalTransformation
    */
public:


    /*!
    \typedef E
    * index type to access an element in the ordered list of numbers
    */
    typedef int    E;


    /*!
    \typedef F
    * number type
    */
    typedef CoordinateType F;
    /*!
    \typedef Domain
    * Domain=int The domain of definition of a Vec is the number of elements
    */
    typedef unsigned int Domain;


    typedef VecNIteratorEDomain IteratorEDomain;
    typedef typename std::vector<CoordinateType>::value_type					 value_type;
    typedef typename std::vector<CoordinateType>::pointer           pointer;
    typedef typename std::vector<CoordinateType>::const_pointer     const_pointer;
    typedef typename std::vector<CoordinateType>::reference         reference;
    typedef typename std::vector<CoordinateType>::const_reference   const_reference;
    typedef typename std::vector<CoordinateType>::iterator iterator;
    typedef typename std::vector<CoordinateType>::const_iterator const_iterator;
    typedef typename std::vector<CoordinateType>::const_reverse_iterator  const_reverse_iterator;
    typedef typename std::vector<CoordinateType>::reverse_iterator		 reverse_iterator;
    typedef typename std::vector<CoordinateType>::size_type					 size_type;
    typedef typename std::vector<CoordinateType>::difference_type				 difference_type;
    typedef typename std::vector<CoordinateType>::allocator_type                        		 allocator_type;
    /*!
    *
    * default constructor
    */
    Vec();

    /*!
    *
    *  copy constructor
    */
    template<typename CoordinateType1>
    Vec(const Vec<CoordinateType1> &v);

    Vec(const Vec &v);

    /*!
    *
    *  constructor the Vec from a std::vector
    */
    template<typename CoordinateType1>
    Vec(const std::vector<CoordinateType1> &v);

    /*!
    * \param  size_vec number of elements
    * \param  value affection value for all elements
    *
    *  constructor a vector with the given size
    */
    explicit Vec(int size_vec, CoordinateType value=CoordinateType());


    /*!
    * \param x input VecN
    *
    *  constructor from a VecN
    * \sa VecN
    */
    template<typename CoordinateType1>
    Vec(const VecN<1,CoordinateType1>& x){
        (*this).resize(1);

        (*this)[0]=x[0];
    }
    /*!
    * \param x input VecN
    *
    *  constructor from a VecN for type conversion
    * \sa VecN
    */
    template<typename CoordinateType1>
    Vec(const VecN<2,CoordinateType1>& x){
        (*this).resize(2);
        (*this)[0]=x[0];
        (*this)[1]=x[1];
    }
    /*!
    * \param x input VecN
    *
    *  constructor from a VecN for type conversion
    * \sa VecN
    */
    template<typename CoordinateType1>
    Vec(const VecN<3,CoordinateType1>& x){
        (*this).resize(3);
        (*this)[0]=x[0];
        (*this)[1]=x[1];
        (*this)[2]=x[2];
    }
    template<typename CoordinateType1>
    Vec(const VecN<4,CoordinateType1>& x){
        (*this).resize(4);
        (*this)[0]=x[0];
        (*this)[1]=x[1];
        (*this)[2]=x[2];
        (*this)[3]=x[3];
    }
    template<typename CoordinateType1>
    Vec(const VecN<5,CoordinateType1>& x){
        (*this).resize(5);
        (*this)[0]=x[0];
        (*this)[1]=x[1];
        (*this)[2]=x[2];
        (*this)[3]=x[3];
        (*this)[4]=x[4];
    }
    /*!
    * \param __first input range
    * \param __last  last range
    * \param __a allocator type
    *
    *  range constructor
    Constructs a container with as many elements as the range [first,last),
    with each element constructed from its corresponding element in that range, in the same order.
    * \sa VecN
    */
    template<typename _InputIterator>
      Vec(_InputIterator __first, _InputIterator __last,
         const allocator_type& __a =  allocator_type())
  : std::vector<CoordinateType>(__first,__last,__a)
      { }


    /*!
    \return number of elements
    *
    * return the number of elements
    */
    Domain getDomain()const;

    /*!
    * \param i element entry
    * \return value reference
    *
    *  return the value reference of the i entry
    */
    CoordinateType & operator ()(unsigned int  i);

    /*!
    * \param i element entry
    * \return const value reference
    *
    *  return the const value reference of the i entry
    */
    const CoordinateType & operator ()(unsigned int  i)const;

    /*!
    * \param v  input vector
    * \return reference of the output vector
    *
    *  Addition assignment
    */
    Vec<CoordinateType>&  operator+=(const Vec<CoordinateType>& v);

    /*!
    * \param value   value
    * \return reference of the output vector
    *
    *  vout(i)=vin(i)+v
    */
    Vec&  operator+=(CoordinateType value);

    /*!
    * \param v other vector
    * \return output vector
    *
    *  Addition mout(i)=(*this)(i)+v1(i)
    */
    Vec<CoordinateType>  operator+(const Vec<CoordinateType>& v)const ;
    /*!
    * \return output vector
    *
    *  Addition mout(i)=v(i)+value
    */
    Vec  operator+(CoordinateType value)const;

    /*!
    * \param v  input vector
    * \return reference of the output vector
    *
    *  Subtraction assignment
    */
    Vec<CoordinateType>&  operator-=(const Vec<CoordinateType>& v);
    /*!
    * \param value  scalar value
    * \return reference of the outputVec vector
    *  vout(i)=vin(i)-v
    */
    Vec&  operator-=(CoordinateType value);
    /*!
    * \param v other vector
    * \return output vector
    *
    *  Subtraction mout(i)=(*this)(i)-v1(i)
    */
    Vec<CoordinateType>  operator-(const Vec<CoordinateType>& v)const ;
    /*!
     * \return output vector
     *
     *   unary - operator -> mout(i)=-(*this)(i)
     */
    Vec<CoordinateType>  operator-();

    /*!
     * \param value input value
     * \return output vector
     *
     *  Division mout(i)=(*this)(i)-value
     */
    Vec  operator-(CoordinateType value)const;


    /*!
    * \param v  input vector
    * \return reference of the output vector
    *
    *  Multiplication term by term
    *
    * vout(i)*=v(i)
    *
    */
    Vec<CoordinateType>&  operator*=(const Vec<CoordinateType>& v);
    /*!
     * \param value  input value
     * \return reference of the output vector
     *
     *  v(i)*=value
     */
    Vec&  operator*=(CoordinateType value);

    /*!
     * \param value input value
     * \return output vector
     *
     *  Multiplication mout(i)=v(i)*value
     */
    Vec  operator*(CoordinateType value)const;

    /*!
     * \param v input vector
     * \return output vector
     *
     *  Multiplication term by term mout(i)=this(i)*v(i)*value
     * \sa productInner(const pop::Vec<T1>& v1,const pop::Vec<T1>& v2)
     */
    Vec<CoordinateType>  operator*(const Vec<CoordinateType>& v)const;

    /*!
     * \param v input vector
     * \return reference of the output vector
     *
     *  vout(i)/=v(i)
     */
    Vec<CoordinateType>& operator/=(const Vec<CoordinateType>& v);

    /*!
     * \param value  input value
     * \return reference of the output vector
     *
     *  v(i)/=value
     */
    Vec& operator/=(CoordinateType value);

    /*!
     * \param v input vector
     * \return output vector
     *
     *  Division mout(i)=this(i)/v(i)
     */
    Vec<CoordinateType> operator/(const Vec<CoordinateType>& v)const;

    /*!
     * \param value input value
     * \return output vector
     *
     *  Division mout(i)=v(i)/value
     */
    Vec operator/(CoordinateType value)const;





    /*!
    * \param p norm (2=euclidean)
    * \return the euclidean norm of the vector
    *
    *  return \f$ (\sum_i |v(i)| ^p)^{1/p}\f$
    */
    F32 norm(int p=2)const;

    /*!
    * \param p norm  (2=euclidean)
    * \return the euclidean norm of the vector
    *
    *  return \f$ \sum_i |v(i)|^p)\f$
    */
    F32 normPower(int p=2)const;

    /*!
    * \return the multiplication of all elements
    *
    *  return \f$ \Pi_i v(i) \f$
    */
    CoordinateType multCoordinate();

    /*!
    \param file input file
    \exception  std::string the input file does not exist or it is not .v format
    *
    * The loader attempts to read the Vec using the specified format v
    */
    void load(std::string file);
    /*!
    \param file input file
    \exception  std::string the input file does not exist or it is not .v format
    *
    * The save attempts to save the vector using the specified format v
    */
    void save(std::string file)const ;
    /*!
    * \return  clone
    *
    *  return an exact copy of the object
    *
    */
    Vec * clone();
#ifdef HAVE_SWIG
    void setValue(int index, CoordinateType value){
        (*this)[index]=value;
    }
    CoordinateType getValue(int index)const{
        return (*this)[index];
    }
#endif
    void display();

    IteratorEDomain getIteratorEDomain()const
    {
        return IteratorEDomain(getDomain());
    }

};
typedef Vec<ComplexF32> VecComplexF32;
typedef Vec<F32> VecF32;
typedef Vec<I32> VecI32;
template<typename CoordinateType>
Vec<CoordinateType>::Vec()
{
}
template<typename CoordinateType>
Vec<CoordinateType>::Vec(const Vec<CoordinateType> &v)
    :std::vector<CoordinateType>(v)
{
}

template<typename CoordinateType>template<typename CoordinateType1>
Vec<CoordinateType>::Vec(const Vec<CoordinateType1> &v)
{
    this->resize(v.size());
    std::transform(v.begin(),v.end(),this->begin(),ArithmeticsSaturation<CoordinateType,CoordinateType1>::Range);
}
template<typename CoordinateType>template<typename CoordinateType1>
Vec<CoordinateType>::Vec(const std::vector<CoordinateType1> &v)
    :std::vector<CoordinateType>(v)
{
}
template<typename CoordinateType>
Vec<CoordinateType> * Vec<CoordinateType>::clone(){
    return new Vec(*this);
}

template<typename CoordinateType>
Vec<CoordinateType>::Vec(int size_vec,CoordinateType value)
    :std::vector<CoordinateType>(size_vec,value)
{
}
template<typename CoordinateType>
typename Vec<CoordinateType>::Domain Vec<CoordinateType>::getDomain()const{
    return (int)(*this).size();
}

template<typename CoordinateType>
const CoordinateType & Vec<CoordinateType>::operator ()(unsigned int  i)const{
    POP_DbgAssert(  i<this->size());
    return (*this)[i];
}
template<typename CoordinateType>
CoordinateType & Vec<CoordinateType>::operator ()(unsigned int  i){
    POP_DbgAssert( i<this->size());
    return (*this)[i];
}

template<typename CoordinateType>
void Vec<CoordinateType>::load(std::string file){
    std::ifstream  in(file.c_str());
    if (in.fail())
    {
        std::cerr<<"In Matrix::load, Matrix: cannot open file: "+file;
    }
    else
    {
        in>>*this;
    }
}
template<typename CoordinateType>
void Vec<CoordinateType>::save(std::string file)const {
    std::ofstream  out(file.c_str());
    if (out.fail())
    {
        std::cerr<<"In Matrix::save, cannot open file: "+file;
    }
    else
    {

        out<<*this;
    }
}
template<typename CoordinateType>
F32 Vec<CoordinateType>::norm(int p)const{
    Private::sumNorm<CoordinateType> op(p);
    if(p==0||p==1)
        return std::accumulate(this->begin(),this->end(),0.f,op);
    if(p==2)
        return std::sqrt(std::accumulate(this->begin(),this->end(),0.f,op));
    else
        return std::pow(std::accumulate(this->begin(),this->end(),0.f,op),1.f/p);
}
template<typename CoordinateType>
F32 Vec<CoordinateType>::normPower(int p)const{
    Private::sumNorm<CoordinateType> op(p);
    return std::accumulate(this->begin(),this->end(),0.f,op);

}
template<typename CoordinateType>
CoordinateType Vec<CoordinateType>::multCoordinate(){
    CoordinateType sum=1;
    for(unsigned int i=0;i<this->size();i++)
        sum*=this->operator ()(i);
    return sum;
}
template<typename CoordinateType>
void Vec<CoordinateType>::display(){
    std::cout<<*this<<std::endl;
}
template<typename CoordinateType>
Vec<CoordinateType>&  Vec<CoordinateType>::operator+=(const Vec<CoordinateType>& v)
{
    POP_DbgAssert( this->size()==v.size());
    std::transform(this->begin(),this->end(),v.begin(),this->begin(),std::plus<CoordinateType>());
    return *this;
}
template<typename CoordinateType>
Vec<CoordinateType>&  Vec<CoordinateType>::operator-=(const Vec<CoordinateType>& v)
{
    POP_DbgAssert( this->size()==v.size());
    std::transform(this->begin(),this->end(),v.begin(),this->begin(),std::minus<CoordinateType>());
    return *this;
}
template<typename CoordinateType>
Vec<CoordinateType>&  Vec<CoordinateType>::operator*=(const Vec<CoordinateType>& v)
{
    POP_DbgAssert( this->size()==v.size());
    std::transform(this->begin(),this->end(),v.begin(),this->begin(),std::multiplies<CoordinateType>());
    return *this;
}
template<typename CoordinateType>
Vec<CoordinateType>&  Vec<CoordinateType>::operator/=(const Vec<CoordinateType>& v)
{
    POP_DbgAssert( this->size()==v.size());
    std::transform(this->begin(),this->end(),v.begin(),this->begin(),std::divides<CoordinateType>());
    return *this;
}


template<typename CoordinateType>
Vec<CoordinateType>  Vec<CoordinateType>::operator+(const Vec<CoordinateType>& v)const
{
    Vec<CoordinateType> vout(*this);
    vout+=v;
    return vout;
}
template<typename CoordinateType>
Vec<CoordinateType>  Vec<CoordinateType>::operator-(const Vec<CoordinateType>& v)const
{
    Vec<CoordinateType> vout(*this);
    vout-=v;
    return vout;
}

template<typename CoordinateType>
Vec<CoordinateType>  Vec<CoordinateType>::operator*(const Vec<CoordinateType>& v)const
{
    Vec<CoordinateType> vout(*this);
    vout*=v;
    return vout;
}
template<typename CoordinateType>
Vec<CoordinateType>  Vec<CoordinateType>::operator/(const Vec<CoordinateType>& v)const
{
    Vec<CoordinateType> vout(*this);
    vout/=v;
    return vout;
}

template<typename CoordinateType>
Vec<CoordinateType>  Vec<CoordinateType>::operator-()
{
    Vec<CoordinateType> vout(this->getDomain());
    std::transform(this->begin(),this->end(),vout.begin(),std::negate<CoordinateType>());
    return vout;
}
template<typename CoordinateType>
Vec<CoordinateType>&  Vec<CoordinateType>::operator/=(CoordinateType v){
    for(unsigned int i=0;i<this->size();i++)
        this->operator ()(i)/=v;
    return *this;
}
template<typename CoordinateType>
Vec<CoordinateType>&  Vec<CoordinateType>::operator*=(CoordinateType v){
    for(unsigned int i=0;i<this->size();i++)
        this->operator ()(i)*=v;
    return *this;
}

template<typename CoordinateType>
Vec<CoordinateType>&  Vec<CoordinateType>::operator+=(CoordinateType v){
    for(unsigned int i=0;i<this->size();i++)
        this->operator ()(i)+=v;
    return *this;
}
template<typename CoordinateType>
Vec<CoordinateType>&  Vec<CoordinateType>::operator-=(CoordinateType v){
    for(unsigned int i=0;i<this->size();i++)
        this->operator ()(i)-=v;
    return *this;
}
template<typename CoordinateType>
Vec<CoordinateType>  Vec<CoordinateType>::operator+(CoordinateType value)const
{
    Vec<CoordinateType> vout(*this);
    vout+=value;
    return vout;
}
template<typename CoordinateType>
Vec<CoordinateType>  Vec<CoordinateType>::operator-(CoordinateType value)const
{
    Vec<CoordinateType> vout(*this);
    vout-=value;
    return vout;
}
template<typename CoordinateType>
Vec<CoordinateType>  Vec<CoordinateType>::operator*(CoordinateType value)const
{
    Vec<CoordinateType> vout(*this);
    vout*=value;
    return vout;
}
template<typename CoordinateType>
Vec<CoordinateType>  Vec<CoordinateType>::operator/(CoordinateType value)const
{
    Vec<CoordinateType> vout(*this);
    vout/=value;
    return vout;
}
template<typename T1>
Vec<T1>  operator+(T1 a,const Vec<T1>& v)
{
    Vec<T1> v1(v);
    v1+=a;
    return v1;
}
template<typename T1>
Vec<T1>  operator-(T1 a,const Vec<T1>& v)
{
    Vec<T1> v1(v.size(),a);
    v1-=v;
    return v1;
}
template<typename T1>
Vec<T1>  operator*(T1 a,const Vec<T1>& v)
{
    Vec<T1> v1(v);
    v1*=a;
    return v1;
}
template<typename T1>
Vec<T1>  operator/(T1 a,const Vec<T1>& v)
{
    Vec<T1> v1(v.size(),a);
    v1/=v;
    return v1;
}
/*!
* \ingroup Vec
* \brief absolute value for each coordinate
* \param v1  VecN
* \return output VecN
*
*/
template<typename T1>
pop::Vec<T1> absolute(const pop::Vec<T1>& v1){
    pop::Vec<T1> vout(v1.size());
    std::transform (v1.begin(), v1.end(), vout.begin(), (T1(*)(T1)) absolute );
    return vout;
}
/*!
* \ingroup Vec
* \brief Rounds x downward for each coordinate
* \param v1  VecN
* \return output VecN
*
*/
template<typename T1>
pop::Vec<T1> floor(const pop::Vec<T1>& v1){
    pop::Vec<T1> vout(v1.size());
    std::transform (v1.begin(), v1.end(), vout.begin(), (T1(*)(T1)) std::floor );
    return vout;
}

/*!
* \ingroup Vec
* \brief norm of the VecN \f$\vert u \vert^p=(\sum_i |u_i|^p)^{1/p}\f$
* \param v1  VecN
* \param p  p-norm
* \return norm
*
*/
template<typename T1>
F32 normValue(const pop::Vec<T1>& v1,int p=2){
    return v1.norm(p);
}
/*!
* \ingroup Vec
* \brief norm of the VecN \f$\vert u \vert^p=\sum_i |u_i|^p\f$
* \param v1  VecN
* \param p  p-norm
* \return norm
*
*/
template<typename T1>
F32 normPowerValue(const pop::Vec<T1>& v1,int p=2){
    return v1.normPower(p);
}
/*!
* \ingroup Vec
* \brief distance between two vectors \f$\vert u-v \vert^p\f$
* \param u  VecN
* \param v  VecN
* \param p  p-norm
* \return norm
*
*/
template<typename CoordinateType1>
F32 distance(const pop::Vec<CoordinateType1>& u, const pop::Vec<CoordinateType1>&  v,int p=2)
{
    return normValue(u-v,p);
}

/*!
* \ingroup Vec
* \brief  round functions return the integral value nearest to x rounding half-way cases away for each coordinate
* \param v1  VecN
* \return output VecN
*
*/
template<typename T1>
pop::Vec<T1> round(const pop::Vec<T1>& v1){
    pop::Vec<T1> vout(v1.size());
    std::transform (v1.begin(), v1.end(), vout.begin(), (T1(*)(T1)) round );
    return vout;
}
/*!
* \ingroup Vec
* \brief  maximum  of  VecN \a v1 by the VecN \a v2 \f$\min(v1,v2)=(\min(v1_0,v2_0),\min(v1_1,v2_1))\f$ for each coordinate
* \param v1 first VecN
* \param v2 second VecN
* \return output VecN
*
*
*/
template<typename T1>
pop::Vec<T1> maximum(const pop::Vec<T1>& v1,const pop::Vec<T1>& v2){
    POP_DbgAssert(v1.size()==v2.size());
    pop::Vec<T1> vout(v1.size());
    pop::FunctorF::FunctorMaxF2<T1,T1> op;
    std::transform (v1.begin(), v1.end(), v2.begin(),vout.begin(), op);
    return vout;
}

/*!
* \ingroup Vec
* \brief  minimum of  VecN \a u by the vector \a v \f$\max(v1,v2)=(\max(v1_0,v2_0),\max(v1_1,v2_1))\f$ for each coordinate
* \param v1 first vector
* \param v2 second vector
* \return output vector
*
*
*/
template<typename T1>
pop::Vec<T1> minimum(const pop::Vec<T1>& v1,const pop::Vec<T1>& v2){
    POP_DbgAssert(v1.size()==v2.size());
    pop::Vec<T1> vout(v1.size());
    pop::FunctorF::FunctorMinF2<T1,T1> op;
    std::transform (v1.begin(), v1.end(), v2.begin(),vout.begin(), op  );
    return vout;
}
/*!
* \ingroup Vec
* \brief  inner product of two vector  \f$<v1,v2>=\sum_i v1_i v2_i\f$
* \param v1 first vector
* \param v2 second vector
* \return output vector
*
*
*/

namespace Private {struct FunctoProductInner{template <class T1, class T2>F32 operator ()(const T1&  x1,const T2&  x2){return static_cast<F32>(x1)*static_cast<F32>(x2);}};}
template<typename T1>
F32  productInner(const pop::Vec<T1>& v1,const pop::Vec<T1>& v2)
{
    POP_DbgAssert( v1.size()==v2.size());

    return std::inner_product(v1.begin(),v1.end(),v2.begin(),T1(0), std::plus<T1>(),Private::FunctoProductInner());
}

/*!
* \ingroup Vec
* \param out output stream
* \param m input Vec
* \return output stream
*
*  stream insertion of the  Vec
*/
template<typename T1>
std::ostream& operator << (std::ostream& out, const pop::Vec<T1>& m){

    out<<m.size()<<';';
    for(unsigned int j=0;j<m.size();j++)
    {

        out<<m(j);
        out<<';';
        //if(j!=m.size()-1)out<<';';
    }
    return out;
}
/*!
* \ingroup Vec
* \param in input stream
* \param m ouput Vec
* \return input stream
*
*  stream extraction of the  Vec
*/
template<typename T1>
std::istream& operator >> (std::istream& in, pop::Vec<T1>& m){
    T1 x;
    m.clear();
    std::string mot;
    std::getline( in, mot, ';' );
    int size;
    pop::BasicUtility::String2Any(mot,size);
    m.resize(size);
    for(unsigned int j=0;j<m.size();j++)
    {
        std::getline( in, mot, ';' );
        pop::BasicUtility::String2Any(mot,x);
        m.operator ()(j)=x;

    }
    return in;
}
}
#endif // Vec_H
