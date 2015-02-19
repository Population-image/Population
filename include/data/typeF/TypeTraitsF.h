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

#ifndef TYPETRAITSF_HPP
#define TYPETRAITSF_HPP
#include<limits>
#include<string>
#include"data/typeF/TypeF.h"
#include"PopulationConfig.h"
#include"data/utility/BasicUtility.h"


namespace pop
{
POP_EXPORTS UI8 maximum(UI8 v1,UI8 v2);
POP_EXPORTS UI16 maximum(UI16 v1,UI16 v2);
POP_EXPORTS UI32 maximum(UI32 v1,UI32 v2);
POP_EXPORTS I8 maximum(I8 v1,I8 v2);
POP_EXPORTS I16 maximum(I16 v1,I16 v2);
POP_EXPORTS I32 maximum(I32 v1,I32 v2);
POP_EXPORTS F32 maximum(F32 v1,F32 v2);
POP_EXPORTS F64 maximum(F64 v1,F64 v2);

POP_EXPORTS UI8 minimum(UI8 v1,UI8 v2);
POP_EXPORTS UI16 minimum(UI16 v1,UI16 v2);
POP_EXPORTS UI32 minimum(UI32 v1,UI32 v2);
POP_EXPORTS I8 minimum(I8 v1,I8 v2);
POP_EXPORTS I16 minimum(I16 v1,I16 v2);
POP_EXPORTS I32 minimum(I32 v1,I32 v2);
POP_EXPORTS F32 minimum(F32 v1,F32 v2);
POP_EXPORTS F64 minimum(F64 v1,F64 v2);

POP_EXPORTS F32 absolute(UI8 v1);
POP_EXPORTS F32 absolute(UI16 v1);
POP_EXPORTS F32 absolute(UI32 v1);
POP_EXPORTS F32 absolute(I8 v1);
POP_EXPORTS F32 absolute(I16 v1);
POP_EXPORTS F32 absolute(I32 v1);
POP_EXPORTS F32 absolute(F32 v1);
POP_EXPORTS F64 absolute(F64 v1);

POP_EXPORTS F32 normValue(UI8 v1,int =2);
POP_EXPORTS F32 normValue(UI16 v1,int =2);
POP_EXPORTS F32 normValue(UI32 v1,int =2);
POP_EXPORTS F32 normValue(I8 v1,int =2);
POP_EXPORTS F32 normValue(I16 v1,int =2);
POP_EXPORTS F32 normValue(I32 v1,int =2);
POP_EXPORTS F32 normValue(F32 v1,int =2);
POP_EXPORTS F64 normValue(F64 v1,int =2);

POP_EXPORTS F32 normPowerValue(UI8 v1,int =2);
POP_EXPORTS F32 normPowerValue(UI16 v1,int =2);
POP_EXPORTS F32 normPowerValue(UI32 v1,int =2);
POP_EXPORTS F32 normPowerValue(I8 v1,int =2);
POP_EXPORTS F32 normPowerValue(I16 v1,int =2);
POP_EXPORTS F32 normPowerValue(I32 v1,int =2);
POP_EXPORTS F32 normPowerValue(F32 v1,int =2);
POP_EXPORTS F64 normPowerValue(F64 v1,int =2);

POP_EXPORTS F32 distance(UI8 v1,UI8 v2, int p =2);
POP_EXPORTS F32 distance(UI16 v1,UI16 v2, int p =2);
POP_EXPORTS F32 distance(UI32 v1,UI32 v2, int p =2);
POP_EXPORTS F32 distance(I8 v1,I8 v2, int p =2);
POP_EXPORTS F32 distance(I16 v1,I16 v2, int p =2);
POP_EXPORTS F32 distance(I32 v1,I32 v2, int p =2);
POP_EXPORTS F32 distance(F32 v1,F32 v2, int p =2);
POP_EXPORTS F64 distance(F64 v1,F64 v2, int p =2);

POP_EXPORTS F32 productInner(UI8 v1,UI8 v2);
POP_EXPORTS F32 productInner(UI16 v1,UI16 v2);
POP_EXPORTS F32 productInner(UI32 v1,UI32 v2);
POP_EXPORTS F32 productInner(I8 v1,I8 v2);
POP_EXPORTS F32 productInner(I16 v1,I16 v2);
POP_EXPORTS F32 productInner(I32 v1,I32 v2);
POP_EXPORTS F32 productInner(F32 v1,F32 v2);
POP_EXPORTS F64 productInner(F64 v1,F64 v2);

POP_EXPORTS F32 round(F32 v1);
POP_EXPORTS F64 round(F64 v1);

POP_EXPORTS F32 squareRoot(F32 v1);
POP_EXPORTS F64 squareRoot(F64 v1);

template<typename T>
class NumericLimits
{

private:
    template<int DIM>
    T _minimumRange(Int2Type<DIM>);
    static inline T _minimumRange(Int2Type<true>){
        return (std::numeric_limits<T>::min)();
    }
    static inline T _minimumRange(Int2Type<false>){
        return -(std::numeric_limits<T>::max)();
    }
public:
    static const bool is_specialized = true;

    static T minimumRange() throw()
    { return _minimumRange(Int2Type<std::numeric_limits<T>::is_integer>() );}
    static T maximumRange() throw()
    { return (std::numeric_limits<T>::max)(); }
    static const int digits10 = std::numeric_limits<T>::digits10;
    static const bool is_integer = std::numeric_limits<T>::is_integer;
};


////////////////////////////////////////////////////////////////////////////////
// ArithmeticsTrait
// Definition of the cast rules for an arithmetic operation
// Let suppose this code
// UI8 c1=230;
// UI8 c2 = 30;
// UI8 c3 = c1 + C2;
// here c1 and c2 are cast in type short I32
////////////////////////////////////////////////////////////////////////////////


template<typename F>
struct isVectoriel{
    enum { value =false};
};
template<typename Type>
struct TypeTraitsTypeScalar;

template<typename T1, typename T2>
struct ArithmeticsTrait;




namespace Private{
template<typename PixelType,int isVect>
struct TypeTraitsTypeScalarTest;

template<typename PixelType>
struct TypeTraitsTypeScalarTest<PixelType,true>{
    typedef typename PixelType::F Result;
};
template<typename PixelType>
struct TypeTraitsTypeScalarTest<PixelType,false>{
    typedef  PixelType Result;
};

template<typename Type1,typename PixelType2,int isVect1,int isVect2>
struct ArithmeticsTraitTest;

template<typename PixelType1,typename PixelType2>
struct ArithmeticsTraitTest<PixelType1,PixelType2,false,false>
{
    typedef PixelType1 Result;
};
template<typename PixelType1,typename PixelType2>
struct ArithmeticsTraitTest<PixelType1,PixelType2,true,false>
{
    typedef typename TypeTraitsTypeScalar<PixelType1 >::Result PixelType1Scalar;
    typedef typename FunctionTypeTraitsSubstituteF<PixelType1,typename ArithmeticsTrait<PixelType1Scalar,PixelType2>::Result>::Result Result;
};
template<typename PixelType1,typename PixelType2>
struct ArithmeticsTraitTest<PixelType1,PixelType2,false, true>
{
    typedef typename TypeTraitsTypeScalar<PixelType2>::Result PixelType2Scalar;
    typedef typename FunctionTypeTraitsSubstituteF<PixelType2,typename ArithmeticsTrait<PixelType1,PixelType2Scalar>::Result>::Result Result;
};
template<typename PixelType1,typename PixelType2>
struct ArithmeticsTraitTest<PixelType1,PixelType2,true, true>
{
    typedef typename TypeTraitsTypeScalar<PixelType1 >::Result PixelType1Scalar;
    typedef typename TypeTraitsTypeScalar<PixelType2  >::Result PixelType2Scalar;
    typedef typename FunctionTypeTraitsSubstituteF<PixelType1,typename ArithmeticsTrait<PixelType1Scalar,PixelType2Scalar>::Result>::Result Result;
};

}
template<typename Type>
struct TypeTraitsTypeScalar
{
    typedef typename Private::TypeTraitsTypeScalarTest<Type,isVectoriel<Type>::value >::Result Result;
};
template<typename T1, typename T2>
struct ArithmeticsTrait
{
    typedef typename Private::ArithmeticsTraitTest<T1,T2,isVectoriel<T1>::value,isVectoriel<T2>::value >::Result Result;
};



//F1=UI8
template<>
struct ArithmeticsTrait<UI8,UI8>
{
    typedef short Result;
};
template<>
struct ArithmeticsTrait<UI8,UI16>
{
    typedef I32 Result;
};
template<>
struct ArithmeticsTrait<UI8,I16>
{
    typedef I32 Result;
};

template<>
struct ArithmeticsTrait<UI8,UI32>
{
    typedef UI32 Result;
};
template<>
struct ArithmeticsTrait<UI8,I32>
{
    typedef I32 Result;
};
template<>
struct ArithmeticsTrait<UI8,F32>
{
    typedef F32 Result;
};
template<>
struct ArithmeticsTrait<UI8,F64>
{
    typedef F64 Result;
};


//F1=UI16
template<>
struct ArithmeticsTrait<UI16,UI8>
{
    typedef I32 Result;
};

template<>
struct ArithmeticsTrait<UI16,UI16>
{
    typedef I32 Result;
};
template<>
struct ArithmeticsTrait<UI16,I16>
{
    typedef I32 Result;
};
template<>
struct ArithmeticsTrait<UI16,I32>
{
    typedef I32 Result;
};

template<>
struct ArithmeticsTrait<UI16,F32>
{
    typedef F32 Result;
};

//F1=I16
template<>
struct ArithmeticsTrait<I16,UI8>
{
    typedef I32 Result;
};

template<>
struct ArithmeticsTrait<I16,UI16>
{
    typedef I32 Result;
};
template<>
struct ArithmeticsTrait<I16,I16>
{
    typedef I32 Result;
};
template<>
struct ArithmeticsTrait<I16,I32>
{
    typedef I32 Result;
};

template<>
struct ArithmeticsTrait<I16,F32>
{
    typedef F32 Result;
};



//F1=I32
template<>
struct ArithmeticsTrait<I32,UI8>
{
    typedef I32 Result;
};
template<>
struct ArithmeticsTrait<I32,UI16>
{
    typedef I32 Result;
};
template<>
struct ArithmeticsTrait<I32,I16>
{
    typedef I32 Result;
};
template<>
struct ArithmeticsTrait<I32,I32>
{
    typedef I32 Result;
};

template<>
struct ArithmeticsTrait<I32,F32>
{
    typedef F32 Result;
};
//F1=F32
template<>
struct ArithmeticsTrait<F32,UI8>
{
    typedef F32 Result;
};
template<>
struct ArithmeticsTrait<F32,UI16>
{
    typedef F32 Result;
};
template<>
struct ArithmeticsTrait<F32,I16>
{
    typedef F32 Result;
};

template<>
struct ArithmeticsTrait<F32,I32>
{
    typedef F32 Result;
};
template<>
struct ArithmeticsTrait<F32,F32>
{
    typedef F32 Result;
};


////////////////////////////////////////////////////////////////////////////////
// Bound Policy
////////////////////////////////////////////////////////////////////////////////


//By default, this method return the value
template< typename R, typename T>
struct ArithmeticsSaturation
{
    static R Range(T p)
    {
        if(p>=NumericLimits<R>::maximumRange())return NumericLimits<R>::maximumRange();
        else if(p<NumericLimits<R>::minimumRange())return NumericLimits<R>::minimumRange();
        else return static_cast<R>(p);
    }
};
template<>
struct ArithmeticsSaturation<UI8,F32>
{
    static UI8 Range(F32 p)
    {
        if(p+0.5>=NumericLimits<UI8>::maximumRange())return NumericLimits<UI8>::maximumRange();
        else if(p+0.5<NumericLimits<UI8>::minimumRange())return NumericLimits<UI8>::minimumRange();
        else return static_cast<UI8>(p+0.5);
    }
};
template<>
struct ArithmeticsSaturation<UI8,F64>
{
    static UI8 Range(F64 p)
    {
        if(p+0.5>=NumericLimits<UI8>::maximumRange())return NumericLimits<UI8>::maximumRange();
        else if(p+0.5<NumericLimits<UI8>::minimumRange())return NumericLimits<UI8>::minimumRange();
        else return static_cast<UI8>(p+0.5);
    }
};

template<>
struct ArithmeticsSaturation< F32, F32>
{
    static F32 Range(F32 p)
    {
        return p;
    }
};


template< typename R, typename T>
struct ArithmeticsPeriodic
{
    static R Range(T p)
    {
        return (p+NumericLimits<R>::maximumRange())%NumericLimits<R>::maximumRange();
    }
};



}

#endif // TYPETRAITSF_HPP
