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
#ifndef RGB_HPP
#define RGB_HPP
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include<limits>
#include<cmath>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include"PopulationConfig.h"
#include"data/utility/BasicUtility.h"
#include"data/typeF/TypeF.h"
#include"data/typeF/TypeTraitsF.h"

#if Pop_OS==2
//In #include<windows.h>, we have the macro RGB
#undef RGB
#endif
namespace pop
{
/*! \ingroup Data
* \defgroup TypeF TypeF
* \brief type for scalar/vector elements in matrix or vector coordinate 
*/

/*! \ingroup TypeF
* \defgroup RGB  RGB{UI8,F32}
* \brief template class for red green blue color
*/
template<class Type>
class POP_EXPORTS RGB
{

    /*!
	* \class pop::RGB
	* \brief Red Green Blue channel
	* \ingroup RGB
    * \author Tariel Vincent
	* \tparam channel type
	*
    * This is an additive RGB model in which red, green, and blue light are added together in various ways to reproduce a broad array of colors \n
	* To facilitate its utilisation, we use some typedef declarations to define the usual types to allow coding in C-style as these ones:
    *        - RGBUI8: each channel is one byte representing an integer between 0 and 255, classical type for pixel/voxel of a colour pop::MatN (image) 
    *        - RGBF32: each channel is a float.
	*
    * \code
    * RGBUI8 c1(100,200,30);
    * c1.r()=10;
    * c1=c1+20;
    * std::cout<<c1<<std::endl;
    * std::cout<<c1.lumi()<<std::endl;
    * \endcode
    * \sa http://en.wikipedia.org/wiki/RGB_color_model
    */

public:

    /*! \var DIM
     * Space dimension equal to 3
     */
    enum{
        DIM=3
    };
    /*!
    \typedef E
    * index type to access elements
    */
    typedef int E;
    /*!
    \typedef F
    * type of the channel
    */
    typedef Type F;

    /*!
    *
    * default constructor
    */
    RGB()
    //        :_data{0,0,0} extended initializer for c++0x
    {
        _data[0]=0;
        _data[1]=0;
        _data[2]=0;
    }

    /*!
    * \param r_value red RGB
    * \param g_value green RGB
    * \param b_value blue RGB
    *
    * constructor the RGB object with the RGB values
    */
    RGB( Type  r_value, Type  g_value, Type  b_value)
    //   :_data{r,g,b} extended initializer for c++0x
    {
        _data[0]=r_value;
        _data[1]=g_value;
        _data[2]=b_value;
    }
    /*!
    * \param value  scalar value
    *
    * constructor the RGB object with the RGB values equal to \a value
    */
    RGB( Type value)
    //        :_data{value,value,value}
    {
        _data[0]=value;
        _data[1]=value;
        _data[2]=value;
    }

    /*!
    * \param c copied RGB
    *
    * copy constructor
    */
    template<typename G>
    RGB(const RGB<G> & c)
    //        :_data{c.r(),c.g(),c.b()}
    {
        _data[0]=c.r();
        _data[1]=c.g();
        _data[2]=c.b();
    }
    /*!
    * \param i index
    * \return element
    *
    * Access to the element at the given index i=0=red, i=1=green, i=2=blue
    */
    Type & operator ()(int i){
        return _data[i];
    }
    const Type & operator ()(int i)const{
        return _data[i];
    }
//    Type & operator [](int i){
//        return _data[i];
//    }
//    const Type & operator [](int i)const{
//        return _data[i];
//    }

    /*!
    * \return red element
    *
    * Access to the red element
    */
    Type & r()
    {
        return _data[0];
    }
    /*!
    * \return green element
    *
    * Access to the green element
    */
    Type & g()
    {
        return _data[1];
    }
    /*!
    * \return blue element
    *
    * Access to the blue element
    */
    Type & b()
    {
        return _data[2];
    }
    const Type & r()
    const
    {
        return _data[0];
    }
    const Type & g()
    const
    {
        return _data[1];
    }
    const Type & b()
    const
    {
        return _data[2];
    }

    /*!
    * \param p other RGB
    * \return this RGB
    *
    * Basic assignment of this RGB by \a other
    */
    template <class U1>
    RGB &  operator =( const RGB<U1> &p)
    {
        _data[0]=p.r();
        _data[1]=p.g();
        _data[2]=p.b();
        return *this;
    }

    /*!
    * \param p other RGB
    * \return this RGB
    *
    * Adds the contents of \a other to this RGB.
    */
    template<typename Type1>
    inline RGB & operator +=(const RGB<Type1> &p)
    {
        typedef typename pop::ArithmeticsTrait<Type,Type1>::Result UpType;
        _data[0]=ArithmeticsSaturation<F,UpType >::Range(static_cast<UpType>(_data[0])+static_cast<UpType>(p.r()));
        _data[1]=ArithmeticsSaturation<F,UpType >::Range(static_cast<UpType>(_data[1])+static_cast<UpType>(p.g()));
        _data[2]=ArithmeticsSaturation<F,UpType >::Range(static_cast<UpType>(_data[2])+static_cast<UpType>(p.b()));
        return *this;
    }
    /*!
    * \param value factor
    * \return this RGB
    *
    * Adds all channels of this RGB by the factor \sa value
    */
    template<typename Scalar>
    inline RGB & operator +=(Scalar value)
    {
        typedef typename pop::ArithmeticsTrait<Type,Scalar>::Result UpType;
        _data[0]=ArithmeticsSaturation<F,UpType >::Range(static_cast<UpType>(_data[0])+static_cast<UpType>(value));
        _data[1]=ArithmeticsSaturation<F,UpType >::Range(static_cast<UpType>(_data[1])+static_cast<UpType>(value));
        _data[2]=ArithmeticsSaturation<F,UpType >::Range(static_cast<UpType>(_data[2])+static_cast<UpType>(value));

        return *this;
    }

    /*!
    * \param p other RGB
    * \return this RGB
    *
    * subtract this RGB by the contents of \a other.
    */
    template<typename Type1>
    inline RGB & operator -=(const RGB<Type1> &p)
    {
        typedef typename pop::ArithmeticsTrait<Type,Type1>::Result UpType;
        _data[0]=ArithmeticsSaturation<F,UpType >::Range(static_cast<UpType>(_data[0])-static_cast<UpType>(p.r()));
        _data[1]=ArithmeticsSaturation<F,UpType >::Range(static_cast<UpType>(_data[1])-static_cast<UpType>(p.g()));
        _data[2]=ArithmeticsSaturation<F,UpType >::Range(static_cast<UpType>(_data[2])-static_cast<UpType>(p.b()));
        return *this;
    }
    /*!
    * \param value factor
    * \return this RGB
    *
    * subtract all channels of this RGB by the factor \sa value
    */
    template<typename Scalar>
    inline RGB & operator -=(Scalar value)
    {
        typedef typename pop::ArithmeticsTrait<Type,Scalar>::Result UpType;
        _data[0]=ArithmeticsSaturation<F,UpType >::Range(static_cast<UpType>(_data[0])-static_cast<UpType>(value));
        _data[1]=ArithmeticsSaturation<F,UpType >::Range(static_cast<UpType>(_data[1])-static_cast<UpType>(value));
        _data[2]=ArithmeticsSaturation<F,UpType >::Range(static_cast<UpType>(_data[2])-static_cast<UpType>(value));
        return *this;
    }
    /*!
    * \return this RGB
    *
    * opposite all channels of this RGB (the type of the channel must be signed)
    */
    inline RGB & operator -()
    {
        _data[0]-=_data[0];
        _data[1]-=_data[1];
        _data[2]-=_data[2];
        return *this;
    }
    /*!
    * \param p other RGB
    * \return this RGB
    *
    * Multiply this RGB by the contents of \a other.
    */
    template<typename Type1>
    inline RGB & operator *=(const RGB<Type1> &p)
    {
        typedef typename pop::ArithmeticsTrait<Type,Type1>::Result UpType;
        _data[0]=ArithmeticsSaturation<F,UpType >::Range(static_cast<UpType>(_data[0])*static_cast<UpType>(p.r()));
        _data[1]=ArithmeticsSaturation<F,UpType >::Range(static_cast<UpType>(_data[1])*static_cast<UpType>(p.g()));
        _data[2]=ArithmeticsSaturation<F,UpType >::Range(static_cast<UpType>(_data[2])*static_cast<UpType>(p.b()));
        return *this;
    }
    /*!
    * \param value factor
    * \return this RGB
    *
    * Multiply all channels of this RGB by the factor \sa value
    */
    template<typename Scalar>
    inline RGB & operator *=(Scalar value)
    {
        typedef typename pop::ArithmeticsTrait<Type,Scalar>::Result UpType;
        _data[0]=ArithmeticsSaturation<F,UpType >::Range(static_cast<UpType>(_data[0])*static_cast<UpType>(value));
        _data[1]=ArithmeticsSaturation<F,UpType >::Range(static_cast<UpType>(_data[1])*static_cast<UpType>(value));
        _data[2]=ArithmeticsSaturation<F,UpType >::Range(static_cast<UpType>(_data[2])*static_cast<UpType>(value));

        return *this;
    }
    /*!

    * \param p other RGB
    * \return this RGB
    *
    * Divide this RGB by the contents of \a other.
    */
    template<typename Type1>
    inline RGB & operator /=(const RGB<Type1> &p)
    {
        typedef typename pop::ArithmeticsTrait<Type,Type1>::Result UpType;
        _data[0]=ArithmeticsSaturation<F,UpType >::Range(static_cast<UpType>(_data[0])/static_cast<UpType>(p.r()));
        _data[1]=ArithmeticsSaturation<F,UpType >::Range(static_cast<UpType>(_data[1])/static_cast<UpType>(p.g()));
        _data[2]=ArithmeticsSaturation<F,UpType >::Range(static_cast<UpType>(_data[2])/static_cast<UpType>(p.b()));
        return *this;
    }
    /*!
    * \param value factor
    * \return this RGB
    *
    * Divide all channels of this RGB by the factor \sa value
    */
    template<typename Scalar>
    inline RGB & operator /=(Scalar value)
    {
        typedef typename pop::ArithmeticsTrait<Type,Scalar>::Result UpType;
        _data[0]=ArithmeticsSaturation<F,UpType >::Range(static_cast<UpType>(_data[0])/static_cast<UpType>(value));
        _data[1]=ArithmeticsSaturation<F,UpType >::Range(static_cast<UpType>(_data[1])/static_cast<UpType>(value));
        _data[2]=ArithmeticsSaturation<F,UpType >::Range(static_cast<UpType>(_data[2])/static_cast<UpType>(value));
        return *this;
    }
    /*!
    * \param p other RGB
    * \return boolean
    *
    * return true for each channel of this RGB is equal to the channel of the other RGB, false otherwise
    */
    template<typename Type1>
    bool operator ==(const RGB<Type1> &p )const
    {
        if(this->r() == p.r()&&  this->g() == p.g()&&  this->b() == p.b()) return true;
        else return false;
    }
    /*!
    * \param p other RGB
    * \return boolean
    *
    * return true for at least one channel of this RGB is different to the channel of the other RGB, false otherwise
    */
    template<typename Type1>
    bool  operator!=(const RGB<Type1>&  p)const{
        if(this->r() == p.r()&&  this->g() == p.g()&&  this->b() == p.b()) return false;
        else return true;
    }
    /*!

    * \param x other RGB
    * \return boolean
    *
    * return true for a luminance of this RGB is superior to the luminance of the other RGB, false otherwise
    */
    template<typename Type1>
    bool operator >(const RGB<Type1>&x)const
    {
        if(this->lumi()>x.lumi())return true;
        else return false;
    }
    template<typename Scalar>
    bool operator >(const Scalar&x)const
    {
        if(this->lumi()>x)return true;
        else return false;
    }
    /*!
    * \param x other RGB
    * \return boolean
    *
    * return true for a luminance of this RGB is inferior to the luminance of the other RGB, false otherwise
    */
    template<typename Type1>
    bool operator <(const RGB<Type1>&x)const
    {
        if(this->lumi()<x.lumi())return true;
        else return false;
    }
    template<typename Scalar>
    bool operator <(const Scalar&x)const
    {
        if(this->lumi()<x)return true;
        else return false;
    }
    /*!
    * \param x other RGB
    * \return boolean
    *
    * return true for a luminance of this RGB is superior to the luminance of the other RGB, false otherwise
    */
    template<typename Type1>
    bool operator >=(const RGB<Type1>&x)const
    {
        if(this->lumi()>=x.lumi())return true;
        else return false;
    }
    template<typename Scalar>
    bool operator >=(const Scalar&x)const
    {
        if(this->lumi()>=x)return true;
        else return false;
    }
    /*!
    * \param x other RGB
    * \return boolean
    *
    * return true for a luminance of this RGB is inferior to the luminance of the other RGB, false otherwise
    */
    template<typename Type1>
    bool operator <=(const RGB<Type1>&x)const
    {
        if(this->lumi()<=x.lumi())return true;
        else return false;
    }
    template<typename Scalar>
    bool operator <=(const Scalar&x)const
    {
        if(this->lumi()<=x)return true;
        else return false;
    }
    /*!
    * \return luminance
    *
    * return the luminance of this RGB  (0.299*this->r() + 0.587*this->g() + 0.114*this->b())
    */
    F32 lumi()const{
        return 0.299*this->r() + 0.587*this->g() + 0.114*this->b()+0.000001;
    }

    /*!
    * \param y luma
    * \param u chrominance
    * \param v chrominance
    *
    * Convert  Y'UV model defines a RGB space in terms of one luma (Y') and two chrominance (UV) to RGB model for this RGB object
    * \sa http://en.wikipedia.org/wiki/YUV
    */
    template<typename Type1>
    void fromYUV(Type1 y,Type1 u,Type1 v){
        _data[0] = pop::ArithmeticsSaturation<Type,F32 >::Range(1.0*y+  0*u     +  1.13983*v);
        _data[1] = pop::ArithmeticsSaturation<Type,F32 >::Range(1.0*y+ -0.39465*u -0.58060*v);
        _data[2] = pop::ArithmeticsSaturation<Type,F32 >::Range(1.0*y+  2.03211*u+  0*v);
    }
    /*!
    * \param y luma
    * \param u chrominance
    * \param v chrominance
    *
    * Get Y'UV model defines a RGB space in terms of one luma (Y') and two chrominance (UV) to RGB model for this RGB object
    * \sa http://en.wikipedia.org/wiki/YUV
    */
    template<typename Type1>
    void toYUV(Type1& y,Type1& u,Type1& v)const{
        y = ArithmeticsSaturation<Type,F32 >::Range(0.299*_data[0]  + 0.587 *_data[1]+  0.114*_data[2]+0.0000001);//to avoid the float
        u = ArithmeticsSaturation<Type,F32 >::Range(-0.14713*_data[0]  + -0.28886*_data[1]+  0.436*_data[2]);
        v=  ArithmeticsSaturation<Type,F32 >::Range(0.615*_data[0]  + -0.51499*_data[1]+ -0.10001*_data[2]);
    }

    /*!
    * \param y  luma
    * \param u chrominance
    * \param v chrominance
    *
    * Convert  YUV (Y' stands for the luma component  and U and V are the chrominance ( components) model defines a RGB
    * \sa http://en.wikipedia.org/wiki/YUV
    */

    void fromYUV(F32 y,F32 u,F32 v){
        _data[0] = pop::ArithmeticsSaturation<Type,F32 >::Range(1.0*y+  0*u     +  1.13983*v);
        _data[1] = pop::ArithmeticsSaturation<Type,F32 >::Range(1.0*y+ -0.39465*u -0.58060*v);
        _data[2] = pop::ArithmeticsSaturation<Type,F32 >::Range(1.0*y+  2.03211*u+  0*v);
    }
    /*!
    * \param h luma
    * \param s chrominance
    * \param l chrominance
    *
    * Get Y'UV model defines a RGB space in terms of one luma (Y') and two chrominance (UV) to RGB model for this RGB object
    * \sa http://en.wikipedia.org/wiki/YUV
    *
    */
    void toHSL(F32& h,F32& s,F32& l)const{
        F32 Max_value= pop::NumericLimits<Type>::maximumRange();
        F32 var_R, var_G, var_B , var_min, var_max, var_delta;

        var_R = 1.*this->r()/Max_value;
        var_G = 1.*this->g()/Max_value;
        var_B = 1.*this->b()/Max_value;

        var_min = std::min(var_R, std::min(var_G, var_B));
        var_max = std::max(var_R, std::max(var_G, var_B));
        var_delta = var_max - var_min;

        //calcul de la luminence
        l = ( var_max + var_min )/2.;


        //correspond a un niveaux de gris
        if(var_delta == 0){
            h = 0;
            s = 0;
        }
        else{

            //calcul de la saturation
            if(l < 0.5)
            {
                s = var_delta / ( var_max + var_min );
            }
            else
            {
                s = var_delta / (2 - var_max - var_min);
            }

            F32 del_R = (((var_max - var_R)/6.) + (var_delta/2.)) / var_delta;
            F32 del_G = (((var_max - var_G)/6.) + (var_delta/2.)) / var_delta;
            F32 del_B = (((var_max - var_B)/6.) + (var_delta/2.)) / var_delta;


            //calcul de la teinte
            if(var_R == var_max)
            {
                h = del_B - del_G;
            }

            else if (var_G == var_max)
            {
                h = ((1 /3.) + del_R - del_B);
            }

            else if (var_B == var_max)
            {
                h = ((2 /3.) + del_G - del_R);

            }

            if (h < 0)
            {
                h += Max_value;
            }

            if(h > 1)
            {
                h -= Max_value;
            }
        }
    }


    /*!
    * \param file input file
    *
    * load the RGB from the input file
    */
    void load(const char * file)
    {
        std::ifstream fs(file);
        if (!fs.fail())
            fs>>*this;
        else
            std::cerr<<"In RGB::load, cannot open this file"+std::string(file);
        fs.close();
    }
    /*!
    * \param file input file
    *
    * save the RGB to the input file
    */
    void save(const char * file)const
    {
        std::ofstream fs(file);
        fs<<*this;
        fs.close();
    }

    /*!
    * \param rgb_value  RGB
    * \return output RGB
    *
    * addition of the RGB c with this  RGB
    */
    template<typename Type1>
    RGB  operator+(const RGB<Type1>&  rgb_value)const
    {
        RGB  x(*this);
        x+=rgb_value;
        return x;
    }

    /*!
    * \param v factor
    * \return output RGB
    *
    * addition of the factor \a value to all channels of the RGB \a c1
    */
    RGB  operator+(Type v)const
    {
        RGB  x(*this);
        x+=v;
        return x;
    }


    /*!
    * \param rgb_value first rgb valaue
    * \return output RGB
    *
    * Subtraction of this RGB \a c1 by the RGB \a c
    */
    template<typename Type1>
    RGB  operator-(const RGB<Type1> &  rgb_value)const
    {
        RGB  x(*this);
        x-=rgb_value;
        return x;
    }
    /*!
    * \param v factor
    * \return output RGB
    *
    * Subtraction of all channels of this RGB by the factor \a value
    */
    RGB  operator-(Type v)const
    {
        RGB  x(*this);
        x-=v;
        return x;
    }
    /*!
    * \param c  RGB
    * \return output RGB
    *
    * multiply all channels of this RGB by the RGB c
    */
    template<typename Type1>
    RGB  operator*(const RGB<Type1>&  c)const{
        RGB  x(*this);
        x*=c;
        return x;
    }
    /*!
    * \param v factor
    * \return output RGB
    *
    * multiply  all channels of this RGB by the factor \a value
    */
    template<typename G>
    RGB  operator*(G v  )const{
        RGB  x(*this);
        x*=v;
        return x;
    }

    /*!
    * \param c first RGB
    * \return output RGB
    *
    * divide all channels of this RGB by the RGB \a c
    */
    template<typename Type1>
    RGB  operator/(const RGB<Type1>&  c)const{
        RGB  x(*this);
        x/=c;
        return x;
    }
    /*!
    * \param v factor
    * \return output RGB
    *
    * divide  all channels of this RGB by the factor \a value
    */
    RGB  operator/(Type v)const{
        RGB  x(*this);
        x/=v;
        return x;
    }



    /*!
	\param p p-norm (-1=lumi, 2=euclidean, 0=infinite,...) 
    \return norm of the color
    *
    * return the norm as the luminance of the RGB colour by default
    */
    F32  norm(int p=-1)const
    {
        if(p==-1)
			return lumi();
		else if(p==0)
            return maximum(maximum(absolute(static_cast<F32>(this->_data[0])),absolute(static_cast<F32>(this->_data[1]))),absolute(static_cast<F32>(this->_data[2])));
        else if(p==1)
            return absolute(static_cast<F32>(this->_data[0]))+absolute(static_cast<F32>(this->_data[1]))+absolute(static_cast<F32>(this->_data[2]));

        else if(p==2)
            return std::sqrt(static_cast<F32>(this->_data[0]*this->_data[0]+this->_data[1]*this->_data[1]+this->_data[2]*this->_data[2]));
        else{
            F32 value = std::pow(absolute(static_cast<F32>(this->_data[0])),static_cast<F32>(p))+std::pow(absolute(static_cast<F32>(this->_data[1])),static_cast<F32>(p))+std::pow(absolute(static_cast<F32>(this->_data[2])),static_cast<F32>(p));
            return std::pow(value,1.0/p);
			}
    }
	F32 normPower(int p=-1)const{
        if(p==-1)
            return lumi();
        else if(p==0)
            return maximum(maximum(absolute(static_cast<F32>(this->_data[0])),absolute(static_cast<F32>(this->_data[1]))),absolute(static_cast<F32>(this->_data[2])));
        else if(p==1)
            return absolute(static_cast<F32>(this->_data[0]))+absolute(static_cast<F32>(this->_data[1]))+absolute(static_cast<F32>(this->_data[2]));

        else if(p==2)
            return static_cast<F32>(this->_data[0]*this->_data[0]+this->_data[1]*this->_data[1]+this->_data[2]*this->_data[2]);
        else{
            F32 value = std::pow(absolute(static_cast<F32>(this->_data[0])),static_cast<F32>(p))+std::pow(absolute(static_cast<F32>(this->_data[1])),static_cast<F32>(p))+std::pow(absolute(static_cast<F32>(this->_data[2])),static_cast<F32>(p));
            return value;
            }
    }
#ifdef HAVE_SWIG
    void setValue(int index, Type value){
        _data[index]=value;
    }
    Type getValue(int index)const{
        return _data[index];
    }
#endif
    static RGB<Type> randomRGB(){
        //        srand(time(0));
        int r = rand()%256;
        int g = rand()%256;
        int b = rand()%256;

        return RGB<Type>(r,g,b);
    }
private:
    Type _data[DIM];
};

/*!
* \param c1  RGB
* \param v factor
* \return output RGB
*
* addition of all channels the factor \a value to all channels of the RGB \a c1
*/
template <class T1>
RGB<T1>  operator+(T1 v,const RGB<T1>&  c1){
    RGB<T1>  x(c1);
    x+=v;
    return x;
}
/*!
* \param c1  RGB
* \param v factor
* \return output RGB
*
* subtraction of the factor \a value  by all channels of the RGB \a c1
*/
template<typename T1>
RGB<T1>  operator-(T1 v,const RGB<T1>&  c1)
{
    RGB<T1>  x(v);
    x-=c1;
    return x;
}

/*!
* \param v factor
* \param c1  RGB
* \return output RGB
*
* multiply  all channels of the RGB \a c1 by the factor \a value
*/
template <class T1>
RGB<T1>  operator*(T1 v, const RGB<T1>&  c1){
    RGB<T1>  x(c1);
    x*=v;
    return x;
}
/*!
* \param c1  RGB
* \param v factor
* \return output RGB
*
* divide the factor \a value by all channels of the RGB \a c1
*/
template <class T1>
RGB<T1>  operator/(T1 v, const RGB<T1>&  c1){
    RGB<T1>  x(v);
    x/=c1;
    return x;
}




typedef RGB<UI8> RGBUI8;
typedef RGB<F32> RGBF32;
template<typename Type>
struct isVectoriel<RGB<Type> >{
    enum { value =true};
};
template<typename TypeIn,typename TypeOut>
struct FunctionTypeTraitsSubstituteF<RGB<TypeIn>,TypeOut>
{
    typedef RGB<typename FunctionTypeTraitsSubstituteF<TypeIn,TypeOut>::Result > Result;
};

template< typename R, typename T>
struct ArithmeticsSaturation<RGB<R>,RGB<T> >
{
    static RGB<R> Range(const RGB<T>& p)
    {
        return RGB<R>(
                    ArithmeticsSaturation<R,T>::Range(p.r()),
                    ArithmeticsSaturation<R,T>::Range(p.g()),
                    ArithmeticsSaturation<R,T>::Range(p.b())
                    );
    }
};
template< typename R, typename Scalar>
struct ArithmeticsSaturation<RGB<R>,Scalar >
{
    static RGB<R> Range(Scalar p)
    {

        return RGB<R>(
                    ArithmeticsSaturation<R,Scalar>::Range(p),
                    ArithmeticsSaturation<R,Scalar>::Range(p),
                    ArithmeticsSaturation<R,Scalar>::Range(p)
                    );
    }
};
template<typename Scalar,typename T>
struct ArithmeticsSaturation<Scalar,RGB<T> >
{
    static Scalar  Range(const RGB<T>& p)
    {
        return ArithmeticsSaturation<Scalar,T>::Range(p.lumi());
    }
};

/*! 
* \ingroup RGB
* \brief return the RGB with the minimum luminance
* \param x1 first RGB number
* \param x2 second RGB number
*
*
*/
template <class T1>
pop::RGB<T1>  minimum(const pop::RGB<T1>&  x1,const pop::RGB<T1>&  x2)
{
    if(x1.lumi()<x2.lumi())return x1;
    else return x2;
}
/*! 
* \ingroup RGB
* \brief return the RGB with the maximum luminance
* \param x1 first RGB number
* \param x2 second RGB number
*
*
*/
template <class T1>
pop::RGB<T1>  maximum(const pop::RGB<T1>&  x1,const pop::RGB<T1>&  x2)
{
    if(x1.lumi()>x2.lumi())return x1;
    else return x2;
}
/*!
* \ingroup RGB
* \brief scalar product 
* \param x1 first RGB
* \param x2 second RGB
* \return scalar value
*
* scalar product of the two RGB \f$ u\cdot v=\sum_{i=0}^{n-1} u_i v_i = u_0 v_0 + u_1 v_1 + \cdots + u_{n-1} v_{n-1}.\f$
*/
template <class T1>
inline F32  productInner(const pop::RGB<T1>&  x1,const pop::RGB<T1>&  x2)
{
    return productInner(x1(0),x2(0))+productInner(x1(1),x2(1))+productInner(x1(2),x2(2));
}

/*! 
* \ingroup RGB
* \brief  return the luminance as default norm
* \param p p-norm (-1=luminance, 2=euclidean)
* \param x1  RGB number
*
*
*/
template <class T1>
F32  normValue(const pop::RGB<T1>&  x1,int p=-1)
{
    return x1.norm(p);
}
/*! 
* \ingroup RGB
* \brief  return the luminance as default norm
* \param p p-norm (-1=luminance, 2=euclidean)
* \param x1  RGB number
*
*
*/
template <class T1>
F32  normPowerValue(const pop::RGB<T1>&  x1,int p=-1)
{
    return x1.normPower(p);
}

/*!
* \ingroup RGB
* \brief  return the luminance as default norm
* \param p p-norm (-1=luminance, 2=euclidean)
* \param x1  RGB number
* \param x2  RGB number
*
*
*/
template <class T1>
F32  distance(const pop::RGB<T1>&  x1,const pop::RGB<T1>&  x2,int p=-1)
{
    return norm(x1-x2,p);
}
template <class T1>
pop::RGB<T1>  absolute(const pop::RGB<T1>&  x1)
{
    return pop::RGB<T1>(absolute(x1.r()),absolute(x1.g()),absolute(x1.b()));
}

/*! 
* \ingroup RGB
* \brief  return round of each channel
* \param v RGB number
*
*
*/
template<typename T1>
pop::RGB<T1> round(const pop::RGB<T1>& v){
    return pop::RGB<T1>(round(v.r()),round(v.g()),round(v.b()) );
}
template<typename T1>
pop::RGB<T1> squareRoot(const pop::RGB<T1>& v){
    return pop::RGB<T1>(squareRoot(v.r()),squareRoot(v.g()),squareRoot(v.b()) );
}
/*! stream insertion operator
* \ingroup RGB
* \param out  output stream
* \param h  RGB number
*
*
*/
template <class T1>
std::ostream& operator << (std::ostream& out, const pop::RGB<T1>& h)
{
    out<<h.r()<<"<C>"<<h.g()<<"<C>"<<h.b()<<"<C>";
    return out;
}
template <>
inline std::ostream& operator << (std::ostream& out, const pop::RGB<UI8>& h)
{
    out<<(int)h.r()<<"<C>"<<(int)h.g()<<"<C>"<<(int)h.b()<<"<C>";
    return out;
}
/*! stream extraction operator
* \ingroup RGB
* \param in  input stream
* \param h  RGB number
*
*
*/

template <class T1>
std::istream & operator >> (std::istream& in, pop::RGB<T1> & h)
{
    std::string str;
    str = pop::BasicUtility::getline( in, "<C>" );
    T1 v;
    pop::BasicUtility::String2Any(str,v );
    h.r() =v;
    str =  pop::BasicUtility::getline( in, "<C>" );
    pop::BasicUtility::String2Any(str,v);
    h.g()=v;
    str =  pop::BasicUtility::getline( in, "<C>" );
    pop::BasicUtility::String2Any(str,v);
    h.b()=v;
    return in;
}


template<typename T>
struct NumericLimits<pop::RGB<T> >
{
    static const bool is_specialized = true;

    static pop::RGB<T> minimumRange() throw()
    { return pop::RGB<T>(pop::NumericLimits<T>::minimumRange());}
    static pop::RGB<T> maximumRange() throw()
    { return pop::RGB<T>(NumericLimits<T>::maximumRange()); }
    static const int digits10;
    static const bool is_integer;
};
template<typename T>
const int NumericLimits<pop::RGB<T> >::digits10 = NumericLimits<T>::digits10;
template<typename T>
const bool NumericLimits<pop::RGB<T> >::is_integer = NumericLimits<T>::is_integer;



}
//namespace pop
//{
///// @cond DEV
//template<class Type>
//class POP_EXPORTS RGBA
//{
//    /*!
//        \class pop::RGBA
//		\ingroup RGB
//        \brief RED GREEN BLUE Alpha color channel
//        \author Tariel Vincent
//        \tparam Type type of each RGBA channel

//         A RGBA is defined by 3 channels red, green, and blue in RGBA model.
//         This is an additive RGBA model in which red, green, and blue light are added together in various ways to reproduce a broad array of colors.\n

//        To facilitate its utilisation, we use some typedef declarations to define the usual types to allow coding in C-style as these ones:
//            - RGBAUI8: each channel is one byte representing an integer between 0 and 255, this type is the classical one for pixel/voxel RGBA
//            - RGBAF32: each channel is a float.

//        \code
//        RGBAUI8 c1(100,200,30,100);
//        c1.r()=10;
//        c1=c1+20;
//        std::cout<<c1<<std::endl;
//        std::cout<<c1.lumi()<<std::endl;
//        \endcode

//         \sa http://en.wikipedia.org/wiki/RGB_color_model
//    */

//public:

//    /*! \var DIM
//     * Space dimension equal to 4
//     */
//    enum{
//        DIM=4
//    };
//    /*!
//    \typedef E
//    * index type to access elements
//    */
//    typedef int E;
//    /*!
//    \typedef F
//    * type of the channel
//    */
//    typedef Type F;

//    /*!
//    *
//    * default constructor
//    */
//    RGBA()
//    //        :_data{0,0,0} extended initializer for c++0x
//    {
//        _data[0]=0;
//        _data[1]=0;
//        _data[2]=0;
//        _data[3]=255;
//    }

//    /*!
//    * \param r_value red RGBA
//    * \param g_value green RGBA
//    * \param b_value blue RGBA
//    * \param a_value  RGBA
//    *
//    * constructor the RGBA object with the RGBA values
//    */
//    RGBA(const Type & r_value,const Type & g_value,const Type & b_value,const Type & a_value)
//    //   :_data{r,g,b}extended initializer for c++0x
//    {
//        _data[0]=r_value;
//        _data[1]=g_value;
//        _data[2]=b_value;
//        _data[3]=a_value;
//    }
//    /*!
//    * \param value  scalar value
//    *
//    * constructor the RGBA object with the RGBA values equal to \a value
//    */
//    RGBA(const Type & value)
//    //        :_data{value,value,value}
//    {
//        _data[0]=value;
//        _data[1]=value;
//        _data[2]=value;
//        _data[3]=value;
//    }

//    /*!
//    * \param c copied RGBA
//    *
//    * copy constructor
//    */
//    template<typename G>
//    RGBA(const RGBA<G> & c)
//    //        :_data{c.r(),c.g(),c.b()}
//    {
//        _data[0]=c.r();
//        _data[1]=c.g();
//        _data[2]=c.b();
//        _data[3]=c.a();
//    }
//    /*!
//    * \param i index
//    * \return element
//    *
//    * Access to the element at the given index i=0=red, i=1=green, i=2=blue
//    */
//    Type & operator ()(int i){
//        return _data[i];
//    }
//    const Type & operator ()(int i)const{
//        return _data[i];
//    }
////    Type & operator [](int i){
////        return _data[i];
////    }
////    const Type & operator [](int i)const{
////        return _data[i];
////    }

//    /*!
//    * \return red element
//    *
//    * Access to the red element
//    */
//    Type & r()
//    {
//        return _data[0];
//    }
//    /*!
//    * \return green element
//    *
//    * Access to the green element
//    */
//    Type & g()
//    {
//        return _data[1];
//    }
//    /*!
//    * \return blue element
//    *
//    * Access to the blue element
//    */
//    Type & b()
//    {
//        return _data[2];
//    }
//    /*!
//    * \return alpha element
//    *
//    * Access to the alpha element
//    */
//    Type & a()
//    {
//        return _data[3];
//    }
//    const Type & r()
//    const
//    {
//        return _data[0];
//    }
//    const Type & g()
//    const
//    {
//        return _data[1];
//    }
//    const Type & b()
//    const
//    {
//        return _data[2];
//    }
//    const Type & a()
//    const
//    {
//        return _data[3];
//    }
//    /*!
//    * \param p other RGBA
//    * \return this RGBA
//    *
//    * Basic assignment of this RGBA by \a other
//    */
//    template <class U1>
//    RGBA &  operator =( const RGBA<U1> &p)
//    {
//        _data[0]=p.r();
//        _data[1]=p.g();
//        _data[2]=p.b();
//                _data[3]=p.a();
//        return *this;
//    }

//    /*!
//    * \param p other RGBA
//    * \return this RGBA
//    *
//    * Adds the contents of \a other to this RGBA.
//    */
//    template<typename Type1>
//    inline RGBA & operator +=(const RGBA<Type1> &p)
//    {
//        typedef typename pop::ArithmeticsTrait<Type,Type1>::Result UpType;
//        _data[0]=ArithmeticsSaturation<F,UpType >::Range(static_cast<UpType>(_data[0])+static_cast<UpType>(p.r()));
//        _data[1]=ArithmeticsSaturation<F,UpType >::Range(static_cast<UpType>(_data[1])+static_cast<UpType>(p.g()));
//        _data[2]=ArithmeticsSaturation<F,UpType >::Range(static_cast<UpType>(_data[2])+static_cast<UpType>(p.b()));
//        _data[3]=ArithmeticsSaturation<F,UpType >::Range(static_cast<UpType>(_data[3])+static_cast<UpType>(p.a()));
//        return *this;
//    }
//    /*!
//    * \param value factor
//    * \return this RGBA
//    *
//    * Adds all channels of this RGBA by the factor \sa value
//    */
//    template<typename Scalar>
//    inline RGBA & operator +=(Scalar value)
//    {
//        typedef typename pop::ArithmeticsTrait<Type,Scalar>::Result UpType;
//        _data[0]=ArithmeticsSaturation<F,UpType >::Range(static_cast<UpType>(_data[0])+static_cast<UpType>(value));
//        _data[1]=ArithmeticsSaturation<F,UpType >::Range(static_cast<UpType>(_data[1])+static_cast<UpType>(value));
//        _data[2]=ArithmeticsSaturation<F,UpType >::Range(static_cast<UpType>(_data[2])+static_cast<UpType>(value));
//        _data[3]=ArithmeticsSaturation<F,UpType >::Range(static_cast<UpType>(_data[3])+static_cast<UpType>(value));

//        return *this;
//    }

//    /*!
//    * \param p other RGBA
//    * \return this RGBA
//    *
//    * Subtract this RGBA by the contents of \a other.
//    */
//    template<typename Type1>
//    inline RGBA & operator -=(const RGBA<Type1> &p)
//    {
//        typedef typename pop::ArithmeticsTrait<Type,Type1>::Result UpType;
//        _data[0]=ArithmeticsSaturation<F,UpType >::Range(static_cast<UpType>(_data[0])-static_cast<UpType>(p.r()));
//        _data[1]=ArithmeticsSaturation<F,UpType >::Range(static_cast<UpType>(_data[1])-static_cast<UpType>(p.g()));
//        _data[2]=ArithmeticsSaturation<F,UpType >::Range(static_cast<UpType>(_data[2])-static_cast<UpType>(p.b()));
//         _data[3]=ArithmeticsSaturation<F,UpType >::Range(static_cast<UpType>(_data[3])-static_cast<UpType>(p.a()));
//        return *this;
//    }
//    /*!
//    * \param value factor
//    * \return this RGBA
//    *
//    * Subtract all channels of this RGBA by the factor \sa value
//    */
//    template<typename Scalar>
//    inline RGBA & operator -=(Scalar value)
//    {
//        typedef typename pop::ArithmeticsTrait<Type,Scalar>::Result UpType;
//        _data[0]=ArithmeticsSaturation<F,UpType >::Range(static_cast<UpType>(_data[0])-static_cast<UpType>(value));
//        _data[1]=ArithmeticsSaturation<F,UpType >::Range(static_cast<UpType>(_data[1])-static_cast<UpType>(value));
//        _data[2]=ArithmeticsSaturation<F,UpType >::Range(static_cast<UpType>(_data[2])-static_cast<UpType>(value));
//        _data[3]=ArithmeticsSaturation<F,UpType >::Range(static_cast<UpType>(_data[3])-static_cast<UpType>(value));

//        return *this;
//    }
//    /*!
//    * \return this RGBA
//    *
//    * oposite all channels of this RGBA (the type of the channel must be signed)
//    */
//    inline RGBA & operator -()
//    {
//        _data[0]-=_data[0];
//        _data[1]-=_data[1];
//        _data[2]-=_data[2];
//        _data[3]-=_data[3];
//        return *this;
//    }
//    /*!
//    * \param p other RGBA
//    * \return this RGBA
//    *
//    * Multiply this RGBA by the contents of \a other.
//    */
//    template<typename Type1>
//    inline RGBA & operator *=(const RGBA<Type1> &p)
//    {
//        typedef typename pop::ArithmeticsTrait<Type,Type1>::Result UpType;
//        _data[0]=ArithmeticsSaturation<F,UpType >::Range(static_cast<UpType>(_data[0])*static_cast<UpType>(p.r()));
//        _data[1]=ArithmeticsSaturation<F,UpType >::Range(static_cast<UpType>(_data[1])*static_cast<UpType>(p.g()));
//        _data[2]=ArithmeticsSaturation<F,UpType >::Range(static_cast<UpType>(_data[2])*static_cast<UpType>(p.b()));
//        _data[3]=ArithmeticsSaturation<F,UpType >::Range(static_cast<UpType>(_data[3])*static_cast<UpType>(p.a()));

//        return *this;
//    }
//    /*!
//    * \param value factor
//    * \return this RGBA
//    *
//    * Multiply all channels of this RGBA by the factor \sa value
//    */
//    template<typename Scalar>
//    inline RGBA & operator *=(Scalar value)
//    {
//        typedef typename pop::ArithmeticsTrait<Type,Scalar>::Result UpType;
//        _data[0]=ArithmeticsSaturation<F,UpType >::Range(static_cast<UpType>(_data[0])*static_cast<UpType>(value));
//        _data[1]=ArithmeticsSaturation<F,UpType >::Range(static_cast<UpType>(_data[1])*static_cast<UpType>(value));
//        _data[2]=ArithmeticsSaturation<F,UpType >::Range(static_cast<UpType>(_data[2])*static_cast<UpType>(value));
//        _data[3]=ArithmeticsSaturation<F,UpType >::Range(static_cast<UpType>(_data[3])*static_cast<UpType>(value));

//        return *this;
//    }
//    /*!

//    * \param p other RGBA
//    * \return this RGBA
//    *
//    * Divide this RGBA by the contents of \a other.
//    */
//    template<typename Type1>
//    inline RGBA & operator /=(const RGBA<Type1> &p)
//    {
//        typedef typename pop::ArithmeticsTrait<Type,Type1>::Result UpType;
//        _data[0]=ArithmeticsSaturation<F,UpType >::Range(static_cast<UpType>(_data[0])/static_cast<UpType>(p.r()));
//        _data[1]=ArithmeticsSaturation<F,UpType >::Range(static_cast<UpType>(_data[1])/static_cast<UpType>(p.g()));
//        _data[2]=ArithmeticsSaturation<F,UpType >::Range(static_cast<UpType>(_data[2])/static_cast<UpType>(p.b()));
//        _data[3]=ArithmeticsSaturation<F,UpType >::Range(static_cast<UpType>(_data[3])/static_cast<UpType>(p.b()));

//        return *this;
//    }
//    /*!
//    * \param value factor
//    * \return this RGBA
//    *
//    * Divide all channels of this RGBA by the factor \sa value
//    */
//    template<typename Scalar>
//    inline RGBA & operator /=(Scalar value)
//    {
//        typedef typename pop::ArithmeticsTrait<Type,Scalar>::Result UpType;
//        _data[0]=ArithmeticsSaturation<F,UpType >::Range(static_cast<UpType>(_data[0])/static_cast<UpType>(value));
//        _data[1]=ArithmeticsSaturation<F,UpType >::Range(static_cast<UpType>(_data[1])/static_cast<UpType>(value));
//        _data[2]=ArithmeticsSaturation<F,UpType >::Range(static_cast<UpType>(_data[2])/static_cast<UpType>(value));
//        _data[3]=ArithmeticsSaturation<F,UpType >::Range(static_cast<UpType>(_data[3])/static_cast<UpType>(value));

//        return *this;
//    }
//    /*!
//    * \param p other RGBA
//    * \return boolean
//    *
//    * return true for each channel of this RGBA is equal to the channel of the other RGBA, false otherwise
//    */
//    template<typename Type1>
//    bool operator ==(const RGBA<Type1> &p )const
//    {
//        if(this->r() == p.r()&&  this->g() == p.g()&&  this->b() == p.b()&& this->a() == p.a()) return true;
//        else return false;
//    }
//    /*!
//    * \param p other RGBA
//    * \return boolean
//    *
//    * return true for at least one channel of this RGBA is different to the channel of the other RGBA, false otherwise
//    */
//    template<typename Type1>
//    bool  operator!=(const RGBA<Type1>&  p)const{
//        if(this->r() == p.r()&&  this->g() == p.g()&&  this->b() == p.b()&& this->a() == p.a()) return false;
//        else return true;
//    }
//    /*!

//    * \param x other RGBA
//    * \return boolean
//    *
//    * return true for a luminance of this RGBA is superior to the luminance of the other RGBA, false otherwise
//    */
//    template<typename Type1>
//    bool operator >(const RGBA<Type1>&x)const
//    {
//        if(this->lumi()>x.lumi())return true;
//        else return false;
//    }
//    /*!
//    * \param x other RGBA
//    * \return boolean
//    *
//    * return true for a luminance of this RGBA is inferior to the luminance of the other RGBA, false otherwise
//    */
//    template<typename Type1>
//    bool operator <(const RGBA<Type1>&x)const
//    {
//        if(this->lumi()<x.lumi())return true;
//        else return false;
//    }

//    /*!
//    * \param x other RGBA
//    * \return boolean
//    *
//    * return true for a luminance of this RGBA is superior to the luminance of the other RGBA, false otherwise
//    */
//    template<typename Type1>
//    bool operator >=(const RGBA<Type1>&x)const
//    {
//        if(this->lumi()>=x.lumi())return true;
//        else return false;
//    }
//    /*!
//    * \param x other RGBA
//    * \return boolean
//    *
//    * return true for a luminance of this RGBA is inferior to the luminance of the other RGBA, false otherwise
//    */
//    template<typename Type1>
//    bool operator <=(const RGBA<Type1>&x)const
//    {
//        if(this->lumi()<=x.lumi())return true;
//        else return false;
//    }

//    /*!
//    * \return luminance
//    *
//    * return the luminance of this RGBA  (0.299*this->r() + 0.587*this->g() + 0.114*this->b())
//    */
//    F32 lumi()const{
//        return 0.299*this->r() + 0.587*this->g() + 0.114*this->b()+0.000001;
//    }

//    /*!
//    * \param y luma
//    * \param u chrominance
//    * \param v chrominance
//    *
//    * Convert  Y'UV model defines a RGBA space in terms of one luma (Y') and two chrominance (UV) to RGBA model for this RGBA object
//    \sa http://en.wikipedia.org/wiki/YUV
//    */
//    template<typename Type1>
//    void fromYUV(Type1 y,Type1 u,Type1 v){
//        _data[0] = pop::ArithmeticsSaturation<Type,F32 >::Range(1.0*y+  0*u     +  1.13983*v);
//        _data[1] = pop::ArithmeticsSaturation<Type,F32 >::Range(1.0*y+ -0.39465*u -0.58060*v);
//        _data[2] = pop::ArithmeticsSaturation<Type,F32 >::Range(1.0*y+  2.03211*u+  0*v);
//    }
//    /*!
//    * \param y luma
//    * \param u chrominance
//    * \param v chrominance
//    *
//    * Get Y'UV model defines a RGBA space in terms of one luma (Y') and two chrominance (UV) to RGBA model for this RGBA object
//    \sa http://en.wikipedia.org/wiki/YUV
//    */
//    template<typename Type1>
//    void toYUV(Type1& y,Type1& u,Type1& v)const{
//        y = ArithmeticsSaturation<Type,F32 >::Range(0.299*_data[0]  + 0.587 *_data[1]+  0.114*_data[2]+0.0000001);//to avoid the float
//        u = ArithmeticsSaturation<Type,F32 >::Range(-0.14713*_data[0]  + -0.28886*_data[1]+  0.436*_data[2]);
//        v=  ArithmeticsSaturation<Type,F32 >::Range(0.615*_data[0]  + -0.51499*_data[1]+ -0.10001*_data[2]);
//    }
//    /*!
//    * \param file input file
//    *
//    * load the RGBA from the input file
//    */
//    void load(const char * file)
//    {
//        std::ifstream fs(file);
//        if (!fs.fail())
//            fs>>*this;
//        else
//            std::cerr<<"In RGBA::load, cannot open this file"+std::string(file);
//        fs.close();
//    }
//    /*!
//    * \param file input file
//    *
//    * save the RGBA to the input file
//    */
//    void save(const char * file)const
//    {
//        std::ofstream fs(file);
//        fs<<*this;
//        fs.close();
//    }

//    /*!
//    * \param r  RGBA
//    * \return output RGBA
//    *
//    * addition of the RGBA c with this  RGBA
//    */
//    template<typename Type1>
//    RGBA  operator+(const RGBA<Type1>&  r)const
//    {
//        RGBA  x(*this);
//        x+=r;
//        return x;
//    }

//    /*!
//    * \param v factor
//    * \return output RGBA
//    *
//    * addition of the factor \a value to all channels of the RGBA \a c1
//    */
//    RGBA  operator+(Type v)const
//    {
//        RGBA  x(*this);
//        x+=v;
//        return x;
//    }


//    /*!
//    * \param r first RGBA
//    * \return output RGBA
//    *
//    * Subtraction of this RGBA \a c1 by the RGBA \a c
//    */
//    template<typename Type1>
//    RGBA  operator-(const RGBA<Type1> &  r)const
//    {
//        RGBA  x(*this);
//        x-=r;
//        return x;
//    }
//    /*!
//    * \param v factor
//    * \return output RGBA
//    *
//    * Subtraction of all channels of this RGBA by the factor \a value
//    */
//    RGBA  operator-(Type v)const
//    {
//        RGBA  x(*this);
//        x-=v;
//        return x;
//    }
//    /*!
//    * \param c  RGBA
//    * \return output RGBA
//    *
//    * multiply all channels of this RGBA by the RGBA c
//    */
//    template<typename Type1>
//    RGBA  operator*(const RGBA<Type1>&  c)const{
//        RGBA  x(*this);
//        x*=c;
//        return x;
//    }
//    /*!
//    * \param v factor
//    * \return output RGBA
//    *
//    * multiply  all channels of this RGBA by the factor \a value
//    */
//    template<typename G>
//    RGBA  operator*(G v  )const{
//        RGBA  x(*this);
//        x*=v;
//        return x;
//    }

//    /*!
//    * \param c first RGBA
//    * \return output RGBA
//    *
//    * divide all channels of this RGBA by the RGBA \a c
//    */
//    template<typename Type1>
//    RGBA  operator/(const RGBA<Type1>&  c)const{
//        RGBA  x(*this);
//        x/=c;
//        return x;
//    }
//    /*!
//    * \param v factor
//    * \return output RGBA
//    *
//    * divide  all channels of this RGBA by the factor \a value
//    */
//    RGBA  operator/(Type v)const{
//        RGBA  x(*this);
//        x/=v;
//        return x;
//    }



//    /*!
//    \return norm of the color
//    *
//    * return the norm as the luminance of the RGBA RGBA
//    */
//    F32  norm()
//    {
//        return lumi();
//    }


//#ifdef HAVE_SWIG
//    void setValue(int index, Type value){
//        _data[index]=value;
//    }
//    Type getValue(int index)const{
//        return _data[index];
//    }
//#endif
//    static RGBA<Type> randomRGBA(){
//        //        srand(time(0));
//        int r = rand()%256;
//        int g = rand()%256;
//        int b = rand()%256;
//        int a = rand()%256;
//        return RGBA<Type>(r,g,b,a);
//    }
//private:
//    Type _data[DIM];
//};
///*!
//@}
//*/
///*!
//* \param c1  RGBA
//* \param v factor
//* \return output RGBA
//*
//* addition of all channels the factor \a value to all channels of the RGBA \a c1
//*/
//template <class T1>
//RGBA<T1>  operator+(T1 v,const RGBA<T1>&  c1){
//    RGBA<T1>  x(c1);
//    x+=v;
//    return x;
//}
///*!
//* \param c1  RGBA
//* \param v factor
//* \return output RGBA
//*
//* Subtraction of the factor \a value  by all channels of the RGBA \a c1
//*/
//template<typename T1>
//RGBA<T1>  operator-(T1 v,const RGBA<T1>&  c1)
//{
//    RGBA<T1>  x(v);
//    x-=c1;
//    return x;
//}

///*!
//* \param v factor
//* \param c1  RGBA
//* \return output RGBA
//*
//* multiply  all channels of the RGBA \a c1 by the factor \a value
//*/
//template <class T1>
//RGBA<T1>  operator*(T1 v, const RGBA<T1>&  c1){
//    RGBA<T1>  x(c1);
//    x*=v;
//    return x;
//}
///*!
//* \param c1  RGBA
//* \param v factor
//* \return output RGBA
//*
//* divide the factor \a value by all channels of the RGBA \a c1
//*/
//template <class T1>
//RGBA<T1>  operator/(T1 v, const RGBA<T1>&  c1){
//    RGBA<T1>  x(v);
//    x/=c1;
//    return x;
//}




//typedef RGBA<UI8> RGBAUI8;
//typedef RGBA<F32> RGBAF32;
//template<typename Type>
//struct isVectoriel<RGBA<Type> >{
//    enum { value =true};
//};
//template<typename TypeIn,typename TypeOut>
//struct FunctionTypeTraitsSubstituteF<RGBA<TypeIn>,TypeOut>
//{
//    typedef RGBA<typename FunctionTypeTraitsSubstituteF<TypeIn,TypeOut>::Result > Result;
//};



//template< typename R, typename T>
//struct ArithmeticsSaturation<RGBA<R>,RGB<T> >
//{
//    static RGBA<R> Range(const RGB<T>& p)
//    {
//        return RGBA<R>(
//                    ArithmeticsSaturation<R,T>::Range(p.r()),
//                    ArithmeticsSaturation<R,T>::Range(p.g()),
//                    ArithmeticsSaturation<R,T>::Range(p.b()),
//                    ArithmeticsSaturation<R,T>::Range(255)
//                    );
//    }
//};
//template< typename R, typename T>
//struct ArithmeticsSaturation<RGB<R>,RGBA<T> >
//{
//    static RGB<R> Range(const RGBA<T>& p)
//    {
//        return RGB<R>(
//                    ArithmeticsSaturation<R,T>::Range(p.r()),
//                    ArithmeticsSaturation<R,T>::Range(p.g()),
//                    ArithmeticsSaturation<R,T>::Range(p.b())
//                    );
//    }
//};
//template< typename R, typename T>
//struct ArithmeticsSaturation<RGBA<R>,RGBA<T> >
//{
//    static RGBA<R> Range(const RGBA<T>& p)
//    {
//        return RGBA<R>(
//                    ArithmeticsSaturation<R,T>::Range(p.r()),
//                    ArithmeticsSaturation<R,T>::Range(p.g()),
//                    ArithmeticsSaturation<R,T>::Range(p.b()),
//                    ArithmeticsSaturation<R,T>::Range(p.a())
//                    );
//    }
//};
//template< typename R, typename Scalar>
//struct ArithmeticsSaturation<RGBA<R>,Scalar >
//{
//    static RGBA<R> Range(Scalar p)
//    {

//        return RGBA<R>(
//                    ArithmeticsSaturation<R,Scalar>::Range(p),
//                    ArithmeticsSaturation<R,Scalar>::Range(p),
//                    ArithmeticsSaturation<R,Scalar>::Range(p),
//                    ArithmeticsSaturation<R,Scalar>::Range(p)
//                    );
//    }
//};


//template<typename Scalar,typename T>
//struct ArithmeticsSaturation<Scalar,RGBA<T> >
//{
//    static Scalar  Range(const RGBA<T>& p)
//    {
//        return ArithmeticsSaturation<Scalar,T>::Range(p.lumi());
//    }
//};
///*!
//* \param x1 first RGBA
//* \param x2 second RGBA
//* \return scalar value
//*
//* scalar product of the two RGBA \f$ u\cdot v=\sum_{i=0}^{n-1} u_i v_i = u_0 v_0 + u_1 v_1 + \cdots + u_{n-1} v_{n-1}.\f$
//*/
//template <class T1>
//inline T1   productInner(const pop::RGBA<T1>&  x1,const pop::RGBA<T1>&  x2)
//{
//    T1 value=0;
//    for(pop::UI32 i = 0;i<4;i++)
//    {
//        value+=productInner(x1(i),x2(i));
//    }
//    return value;
//}
///*! return the RGBA with the mininum luminance
//* \param x1 first RGBA number
//* \param x2 second RGBA number
//*
//*
//*/
//template <class T1>
//pop::RGBA<T1>  minimum(const pop::RGBA<T1>&  x1,const pop::RGBA<T1>&  x2)
//{
//    if(x1.lumi()<x2.lumi())return x1;
//    else return x2;
//}
///*! return the RGBA with the maximum luminance
//* \param x1 first RGBA number
//* \param x2 second RGBA number
//*
//*
//*/
//template <class T1>
//pop::RGBA<T1>  maximum(const pop::RGBA<T1>&  x1,const pop::RGBA<T1>&  x2)
//{
//    if(x1.lumi()>x2.lumi())return x1;
//    else return x2;
//}
///*! sqrt of the each RGBA channel
//* \param x1  RGBA number
//*
//*
//*/
//template <class T1>
//pop::RGBA<T1>  squareRoot(const pop::RGBA<T1>&  x1)
//{
//    return pop::RGBA<T1>(sqrt(x1.r()),sqrt(x1.g()),sqrt(x1.b()),sqrt(x1.a()));
//}
///*! return the luminance
//* \param x1  RGBA number
//*
//*
//*/
//template <class T1>
//F32  normValue(const pop::RGBA<T1>&  x1)
//{
//    return x1.lumi();
//}

///*! stream insertion operator
//* \param out  output stream
//* \param h  RGBA number
//*
//*
//*/
//template <class T1>
//std::ostream& operator << (std::ostream& out, const pop::RGBA<T1>& h)
//{
//    out<<h.r()<<"<C>"<<h.g()<<"<C>"<<h.b()<<"<C>"<<h.a()<<"<C>";
//    return out;
//}
///*! stream extraction operator
//* \param in  input stream
//* \param h  RGBA number
//*
//*
//*/
//template <class T1>
//std::istream & operator >> (std::istream& in, pop::RGBA<T1> & h)
//{
//    std::string str;
//    str = pop::BasicUtility::getline( in, "<C>" );
//    T1 v;
//    pop::BasicUtility::String2Any(str,v );
//    h.r() =v;
//    str =  pop::BasicUtility::getline( in, "<C>" );
//    pop::BasicUtility::String2Any(str,v);
//    h.g()=v;
//    str =  pop::BasicUtility::getline( in, "<C>" );
//    pop::BasicUtility::String2Any(str,v);
//    h.b()=v;
//    str =  pop::BasicUtility::getline( in, "<C>" );
//    pop::BasicUtility::String2Any(str,v);
//    h.a()=v;
//    return in;
//}


//template<typename T>
//struct NumericLimits<pop::RGBA<T> >
//{
//    static const bool is_specialized = true;

//    static pop::RGBA<T> minimumRange() throw()
//    { return pop::RGBA<T>(pop::NumericLimits<T>::minimumRange());}
//    static pop::RGBA<T> maximumRange() throw()
//    { return pop::RGBA<T>(NumericLimits<T>::maximumRange()); }
//    static const int digits10;
//    static const bool is_integer;
//};
//template<typename T>
//const int NumericLimits<pop::RGBA<T> >::digits10 = NumericLimits<T>::digits10;
//template<typename T>
//const bool NumericLimits<pop::RGBA<T> >::is_integer = NumericLimits<T>::is_integer;

//template<typename T1>
//pop::RGBA<T1> round(const pop::RGBA<T1>& v){
//    return pop::RGBA<T1>(round(v.r()),round(v.g()),round(v.b()),round(v.a()) );
//}
///// @endcond
//}
#endif // RGBA_HPP

