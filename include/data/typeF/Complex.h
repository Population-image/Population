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
#ifndef COMPLEX_HPP
#define COMPLEX_HPP
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include<limits>
#include<cmath>

#include"data/utility/BasicUtility.h"
#include"data/typeF/TypeF.h"
#include"data/typeF/TypeTraitsF.h"
#include"data/typeF/RGB.h"

namespace pop
{
/*! \ingroup TypeF
* \defgroup Complex  Complex{F32}
* \brief template class for complex numbers in cartesian form
*/
template<class Type>
class POP_EXPORTS Complex
{
    /*!
        \class pop::Complex
        \brief Complex number a+ib
        \author Tariel Vincent
        \ingroup Complex
        \tparam Type Number type
        *
        *
        * A complex number is a number that can be put in the form a + bi, where a and b are  numbers and i is called the imaginary unit, where i^2 = −1.
        * We implement the classical arithmetics for this number

        * To facilite its utilisation, we use some typedef declarations to define the usual types to allow coding in C-style as these ones:
            - ComplexF32: each number is  representing by a float
            - ComplexF32: each number is  representing by a float

        \code
            ComplexF32 c1(1,-1);
            ComplexF32 c2(1,1);
             std::cout<<c1*c2<<std::endl;
        \endcode
        * \sa http://en.wikipedia.org/wiki/Complex_number
    */
private:
    Type _data[2];
public:
    /*! \var DIM
     * Space dimension equal to 2
     */
    enum{
        DIM=2
    };
    /*!
    \typedef E
    * index type to access elements
    */
    typedef int E;
    /*!
    \typedef F
    * type of the numbers
    */
    typedef Type F;


    /*!
    *
    * default constructor
    */
    Complex()
    {
        _data[0]=0;
        _data[1]=0;
    }
    /*!
    * \param c copied complex number
    *
    * copy constructor
    */
    template<typename U>
    Complex(const Complex<U> & c)
    {
        _data[0]=c.real();
        _data[1]=c.img();
    }
    /*!
    * \param real_value real part
    * \param img_value imaginary part
    *
    * constructor the complex number c = real_value + i* img_value
    */
    Complex(const  Type & real_value,const Type & img_value=0)
    {
        _data[0]=real_value;
        _data[1]=img_value;
    }
    /*!
    * \param i index
    * \return element
    *
    * Access to the element at the given index i=0=real, i=1=imaginary
    */
    Type & operator ()(int i){
        return _data[i];
    }
    const Type & operator ()(int i)const{
        return _data[i];
    }
    /*!
    * \return  real part
    *
    * access to the real part
    */
    Type & real()
    {
        return _data[0];
    }
    /*!
    * \return  imaginary part
    *
    * access to the imaginary part
    */
    Type & img()
    {
        return _data[1];
    }
    const Type & real()
    const
    {
        return _data[0];
    }
    const Type & img()
    const
    {
        return _data[1];
    }
    /*!
    * \return reference of this complex number
    *
    * conjugate c=a+ib becomes c=a-ib
    */

    Complex  conjugate()const
    {
        Complex c(*this);
        c.img()=-c.img();
        return c;
    }
    /*!
    * \param p other complex
    * \return this complex
    *
    * Basic assignement of this complex number by \a other
    */
    Complex &  operator =( const Complex &p)
    {
        this->real() = p.real();
        this->img() = p.img();
        return *this;
    }
    /*!
    * \param p other complex
    * \return this complex
    *
    * Adds the contents of \a other to this complex number.
    */

    inline Complex & operator +=(const Complex &p)
    {
        this->real() += p.real();
        this->img() += p.img();
        return *this;
    }
    /*!
    * \param value factor
    * \return this complex
    *
    * Adds all channels of this complex by the factor \sa value
    */
    inline Complex & operator +=(Type value)
    {
        this->real() += value;
        this->img() += value;
        return *this;
    }
    /*!
    * \param p other complex
    * \return this complex
    *
    * Substract this complex by the contents of \a other.
    */
    inline Complex & operator -=(const Complex &p)
    {
        this->real() -= p.real();
        this->img() -= p.img();
        return *this;
    }
    /*!
    * \param value factor
    * \return this complex
    *
    * Substract all channels of this complex by the factor \sa value
    */
    inline Complex & operator -=(Type value)
    {
        this->real() -= value;
        this->img() -= value;
        return *this;
    }
    /*!
    * \return this complex
    *
    * oposite all channels of this complex (the type of the channel must be signed)
    */
    inline Complex  operator -()const
    {
        Complex x;
        x._data[0]-=_data[0];
        x._data[1]-=_data[1];
        return *this;
    }
    /*!
    * \param p other complex
    * \return this complex
    *
    * Multiply this complex by the contents of \a other.
    */
    inline Complex & operator *=(const Complex &p)
    {
        Type temp= productInner(this->real(),p.real())-productInner(this->img(),p.img());
        this->img() = productInner(this->real(),p.img())+productInner(this->img(),p.real());
        this->real() = temp;
        return *this;
    }
    /*!
    * \param value factor
    * \return this complex
    *
    * Multiply all channels of this complex by the factor \sa value
    */
    inline Complex & operator *=(Type value)
    {
        this->real() *= value;
        this->img()*= value;
        return *this;
    }
    /*!
    * \param p other complex
    * \return this complex
    *
    * Divide this complex by the contents of \a other.
    */
    inline Complex & operator /=(const Complex &p)
    {
        Type mod1= std::sqrt(productInner(this->real(),this->real())+productInner(this->img(),this->img()));
        Type angle1 = std::acos(this->real()/mod1);
        if(this->img()<0)angle1=-angle1;

        Type mod2= std::sqrt(productInner(p.real(),p.real())+productInner(p.img(),p.img()));
        Type angle2 = std::acos(p.real()/mod2);
        if(p.img()<0)angle2=-angle2;


        Type sumangle = angle1 + angle2;
        Type divmod   = mod1/mod2;
        this->real() = divmod * std::cos(sumangle);
        this->img() = divmod * std::sin(sumangle);
        return *this;
    }
    /*!
    * \param value factor
    * \return this complex
    *
    * Divide all channels of this complex by the factor \sa value
    */
    inline Complex & operator /=(Type value)
    {
        this->real() /= value;
        this->img()/= value;
        return *this;
    }
    /*!
    * \param x1 other complex
    * \return this complex
    *
    * Multiply this complex by the contents of \a other.
    */
    Complex  operator*(const Complex&  x1)const
    {
        Complex  x(*this);
        x*=x1;
        return x;
    }
    /*!
    * \param value other complex
    * \return this complex
    *
    * Multiply this complex by the contents of \a value.
    */
    Complex  operator*(Type  value)const
    {
        Complex  x(*this);
        x*=value;
        return x;
    }
    Complex  operator/(const Complex&  x1)const
    {
        Complex  x(*this);
        x/=x1;
        return x;
    }
    Complex  operator/(Type  value)const
    {
        Complex  x(*this);
        x/=value;
        return x;
    }
    /*!
    * \param x1 other complex
    * \return this complex
    *
    * Add this complex by the contents of \a other.
    */
    Complex  operator+(const Complex&  x1)const
    {
        Complex  x(*this);
        x+=x1;
        return x;
    }
    /*!
    * \param value scalar value
    * \return this complex
    *
    * Add this complex by the contents of \a value.
    */
    Complex  operator+(Type  value)const
    {
        Complex  x(*this);
        x+=value;
        return x;
    }
    /*!
    * \param x1 other complex
    * \return this complex
    *
    * Substract this complex by the contents of \a other.
    */
    Complex  operator-(const Complex&  x1)const
    {
        Complex  x(*this);
        x-=x1;
        return x;
    }
    /*!
    * \param value scalar value
    * \return this complex
    *
    * Substract this complex by the contents of \a value.
    */
    Complex  operator-(Type  value)const
    {
        Complex  x(*this);
        x-=value;
        return x;
    }

    /*!
    * \param p other complex
    * \return boolean
    *
    * return true for each channel of this complex is equal to the channel of the other complex, false otherwise
    */
    bool operator ==(const Complex &p )const
    {
        if(this->real() == p.real()&&  this->img() == p.img()) return true;
        else return false;
    }
    /*!
    * \param x2 other complex
    * \return boolean
    *
    * return true for at least one channel of this complex is different to the channel of the other complex, false otherwise
    */
    bool  operator!=(const Complex&  x2)const
    {
        if(this->real()!=x2.real() ||  this->img()!=x2.img())
            return true;
        else
            return false;
    }
    /*!
    * \param x other complex
    * \return boolean
    *
    * return true for a luminance of this complex is superior to the luminance of the other complex, false otherwise
    */
    bool operator >(const Complex&x)const
    {
        if(this->real()>x.real() )return true;
        else if(this->real()==x.real()&&this->img()>x.img() )return true;
        else return false;
    }

    /*!
    * \param x other complex
    * \return boolean
    *
    */
    bool operator <(const Complex&x)const
    {
        if(this->real()!=x.real() )return true;
        else if(this->real()==x.real()&&this->img()<x.img() )return true;
        else return false;
    }

    /*!
    * \param x other complex
    * \return boolean
    *
    */
    bool operator >=(const Complex&x)const
    {
        if(this->real()>x.real() )return true;
        else if(this->real()==x.real()&&this->img()>=x.img() )return true;
        else return false;
    }
    /*!
    * \param x other Complex
    * \return boolean
    *
    */
    bool operator <=(const Complex&x)const
    {
        if(this->real()!=x.real() )return true;
        else if(this->real()==x.real()&&this->img()<=x.img() )return true;
        else return false;
    }
    /*!
    * \param angle_radian angle in radian
    *
    * set the complex number from the angle c=cos (angle_radian)+ i*sin (angle_radian)
    */
    void fromAngle(F32 angle_radian)
    {
        this->real() = std::cos (angle_radian);
        this->img() =  std::sin (angle_radian);
    }
    /*!
    * \return angle
    *
    * return the angle defined by the complex number
    */
    F32 toAngle()
    {
        if(this->real()>0&&this->img()>=0)
            return std::atan(this->img()/this->img());
        else if(this->real()<0&&this->img()>=0)
            return std::atan(this->img()/this->img())+PI;
        else if(this->real()>0&&this->img()<=0)
            return std::atan(this->img()/this->img())-PI;
        else if(this->img()>=0)
            return PI/2;
        else
            return -PI/2;
    }


    /*!
    * \param p norm  (2=euclidean)
    * \return the euclidean norm of the vector
    *
    *  return \f$ (|v(0)|^p+|v(1)|^p)^(1/p)\f$
    */
    F32 norm(F32 p=2)const{
        if(p==0)
            return maximum(normValue(this->real(),0),normValue(this->img(),0));
        else if(p==1)
             return normValue(this->real(),0)+normValue(this->img(),0);
        else if(p==2)
            return std::sqrt(normPowerValue(this->real(),2)+normPowerValue(this->img(),2));
        else
            return std::pow(normPowerValue(this->real(),p)+normPowerValue(this->img(),p),1./p);
    }
    F32 normPower(F32 p=2)const{
        if(p==0)
            return maximum(normValue(this->real(),0),normValue(this->img(),0));
        else if(p==1)
             return normValue(this->real(),0)+normValue(this->img(),0);
        else if(p==2)
            return normPowerValue(this->real(),2)+normPowerValue(this->img(),2);
        else
            return normPowerValue(this->real(),p)+normPowerValue(this->img(),p);
    }
#ifdef HAVE_SWIG
    void setValue(int index, Type value){
        _data[index]=value;
    }

    Type getValue(int index)const{
        return _data[index];
    }
#endif

};


template <class T1>
Complex<T1>  operator*(T1  x2, const Complex<T1>&  x1)
{
    Complex<T1>  x(x1);
    x*=x2;
    return x;
}

template <class T1>
Complex<T1>  operator/(T1  x2, const Complex<T1>&  x1)
{
    Complex<T1>  x(x2);
    x/=x1;
    return x;
}

template <class T1>
Complex<T1>  operator+(T1  x2,const Complex<T1>&  x1)
{
    Complex<T1>  x(x1);
    x+=x2;
    return x;
}

template <class T1>
Complex<T1>  operator-(T1  x2,const Complex<T1>&  x1)
{
    Complex<T1>  x(x2);
    x-=x1;
    return x;
}


template<typename Type>
struct isVectoriel<Complex<Type> >{
    enum { value =true};
};
template<typename TypeIn,typename TypeOut>
struct FunctionTypeTraitsSubstituteF<Complex<TypeIn>,TypeOut>
{
    typedef Complex<typename FunctionTypeTraitsSubstituteF<TypeIn,TypeOut>::Result> Result;
};
typedef Complex<F32> ComplexF32;


template< typename R, typename T>
struct ArithmeticsSaturation<Complex<R>,Complex<T> >
{
    static Complex<R> Range(const Complex<T>& p)
    {
        return Complex<R>(
                    ArithmeticsSaturation<R,T>::Range(p.real()),
                    ArithmeticsSaturation<R,T>::Range(p.img())
                    );
    }
};
template<typename R,typename Scalar>
struct ArithmeticsSaturation<Scalar,Complex<R> >
{
    static Scalar  Range(const Complex<R>& p)
    {
        return ArithmeticsSaturation<Scalar,R>::Range(p.norm());
    }
};
template< typename R, typename Scalar>
struct ArithmeticsSaturation<Complex<R>,Scalar >
{
    static Complex<R> Range(Scalar p)
    {

        return Complex<R>(
                    ArithmeticsSaturation<R,Scalar>::Range(p),
                    ArithmeticsSaturation<R,Scalar>::Range(p)
                    );
    }
};

template<typename Scalar>
struct ArithmeticsSaturation<RGB<Scalar>,Complex<F32> >
{
    static RGB<Scalar>  Range(const Complex<F32>& p)
    {
        return RGB<Scalar>(ArithmeticsSaturation<Scalar,F32>::Range(p.norm()));
    }
};
//template<typename Scalar>
//struct ArithmeticsSaturation<RGBA<Scalar>,Complex<F32> >
//{
//    static RGBA<Scalar>  Range(const Complex<F32>& p)
//    {
//        return RGBA<Scalar>(ArithmeticsSaturation<Scalar,F32>::Range(p.norm()));
//    }
//};
template<typename Scalar>
struct ArithmeticsSaturation<Complex<F32> , RGB<Scalar> >
{
    static Complex<F32>  Range(const RGB<Scalar>& p)
    {
        return Complex<F32>(p.norm());
    }
};
/*!
* \ingroup Complex
* \brief minimum of the two numbers
* \param x1 first complex number
* \param x2 second complex number
*
*
*/
template <class T1>
pop::Complex<T1>  minimum(const pop::Complex<T1>&  x1,const pop::Complex<T1>&  x2)
{
    return pop::Complex<T1>(minimum(x1.real(),x2.real()),minimum(x1.img(),x2.img()));
}
/*!
* \ingroup Complex
* \brief maximum of the two numbers
* \param x1 first complex number
* \param x2 second complex number
*
*
*/
template <class T1>
pop::Complex<T1>  maximum(const pop::Complex<T1>&  x1,const pop::Complex<T1>&  x2)
{
    return pop::Complex<T1>(maximum(x1.real(),x2.real()),maximum(x1.img(),x2.img()));
}
/*!
* \ingroup Complex
* \brief sqrt of the number  http://en.wikipedia.org/wiki/Complex_number#Square_root
* \param x1  complex number
*
*
*/
template <class T1>
pop::Complex<T1>  squareRoot(const pop::Complex<T1>&  x1)
{
    return pop::Complex<T1>(squareRoot( (x1.real() + squareRoot(x1.real()*x1.real() +x1.img()*x1.img()))/2),pop::sgn(x1.img())*squareRoot( (x1.real() + squareRoot(-x1.real()*x1.real() +x1.img()*x1.img()))/2));
}
/*!
* \ingroup Complex
* \brief inner product <x1=(a+ib)|x2=(c+id)>=a*c+b*d
* \param x1  complex number
* \param x2  complex number
*
*
*/
template <class T1>
F32  productInner(const pop::Complex<T1>&  x1,const pop::Complex<T1>&  x2)
{
    return x1.real()*x2.real()+  x1.img()*x2.img();
}
/*!
* \ingroup Complex
* \brief norm of the number a^2+b^2  with x1 = a+ib
* \param x1  complex number
* \param p p-norm
*
*
*/
template <class T1>
F32  normValue(const pop::Complex<T1>&  x1,F32 p=2)
{
    return x1.norm(p);
}
/*!
* \ingroup Complex
* \brief norm of the number a^2+b^2  with x1 = a+ib
* \param x1  complex number
* \param p p-norm
*
*
*/
template <class T1>
F32  normPowerValue(const pop::Complex<T1>&  x1,F32 p=2)
{
    return x1.normPower(p);
}
/*!
* \ingroup Complex
* \brief distance between two complex numbers
* \param x1  complex number
* \param x2  complex number
* \param p p-norm
*
*
*/
template <class T1>
F32  distance(const pop::Complex<T1>&  x1,const pop::Complex<T1>&  x2,F32 p=2)
{
    return normValue(x1-x2,p);
}

/*!
* \ingroup Complex
* \brief Returns the absolute value of each channel
* \param x1  complex number
*
*
*/
template <class T1>
pop::Complex<T1>  absolute(const pop::Complex<T1>&  x1)
{
    return pop::Complex<T1>(absolute(x1.real()),absolute(x1.img()));
}
/*! stream insertion operator
 * \ingroup Complex
* \param out  output stream
* \param x1  complex number
*
*
*/
template <class T1>
std::ostream& operator << (std::ostream& out, const pop::Complex<T1> x1)
{
    out<<x1.real()<<"<C>"<<x1.img()<<"<C>";
    return out;
}
/*! stream extraction operator
 * \ingroup Complex
* \param in  input stream
* \param x1  complex number
*
*
*/
template <class T1>
std::istream & operator >> (std::istream& in, pop::Complex<T1> & x1)
{
    std::string str;
    str = pop::BasicUtility::getline( in, "<C>" );
    pop::BasicUtility::String2Any(str,x1.real());
    str = pop::BasicUtility::getline( in, "<C>" );
    pop::BasicUtility::String2Any(str,x1.img());
    return in;
}
template<typename T>
struct NumericLimits<pop::Complex<T> >
{
    static const bool is_specialized = true;

    static pop::Complex<T> minimumRange() throw()
    { return pop::Complex<T>(pop::NumericLimits<T>::minimumRange());}
    static pop::Complex<T> maximumRange() throw()
    { return pop::Complex<T>(NumericLimits<T>::maximumRange()); }
    static const int digits10 = NumericLimits<T>::digits10;
    static const bool is_integer = NumericLimits<T>::is_integer;
};

template<typename T1>
pop::Complex<T1> round(const pop::Complex<T1>& v){
    return pop::Complex<T1>(round(v.real()),round(v.img()));
}
}



#endif // COMPLEX_HPP

