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
////////////////////////////////////////////////////////////////////////////////
// These header files  come from the Loki Library
// Copyright (c) 2001 by Andrei Alexandrescu
// This code accompanies the book:
// Alexandrescu, Andrei. "Modern C++ Design: Generic Programming and Design
//     Patterns Applied". Copyright (c) 2001. Addison-Wesley.
// Permission to use, copy, modify, distribute and sell this software for any
//     purpose is hereby granted without fee, provided that the above copyright
//     notice appear in all copies and that both that copyright notice and this
//     permission notice appear in supporting documentation.
// The author or Addison-Welsey Longman make no representations about the
//     suitability of this software for any purpose. It is provided "as is"
//     without express or implied warranty.
////////////////////////////////////////////////////////////////////////////////



#include"limits"

#include"cmath"
#include"data/GP/Type2Id.h"
#include"data/typeF/TypeF.h"
#include"PopulationConfig.h"
#include"data/GP/TypeManip.h"


namespace pop
{

template<typename T>
class NumericLimits
{

private:
    template<int DIM>
    T _minimumRange(Loki::Int2Type<DIM>);
    static inline T _minimumRange(Loki::Int2Type<true>){
        return std::numeric_limits<T>::min();
    }
    static inline T _minimumRange(Loki::Int2Type<false>){
        return -std::numeric_limits<T>::max();
    }
public:
    static const bool is_specialized = true;

    static T minimumRange() throw()
    { return _minimumRange(Loki::Int2Type<std::numeric_limits<T>::is_integer>() );}
    static T maximumRange() throw()
    { return std::numeric_limits<T>::max(); }
    static const int digits10 = std::numeric_limits<T>::digits10;
    static const bool is_integer = std::numeric_limits<T>::is_integer;
};
////////////////////////////////////////////////////////////////
// The identifiant is define by using a enum
// Note: If you add your own identifiant, add it at the end of the list
////////////////////////////////////////////////////////////////




template<>
struct POP_EXPORTS Type2Id<UI8>
{
    Type2Id();
    std::vector<std::string> id;

};


template<>
struct POP_EXPORTS Type2Id<UI16>
{
    Type2Id();
    std::vector<std::string> id;
};

template<>
struct POP_EXPORTS Type2Id<I32>
{
    Type2Id();
    std::vector<std::string> id;
};


template<>
struct POP_EXPORTS Type2Id<F64>
{
    Type2Id();
    std::vector<std::string> id;
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
template<typename Type,int isVect>
struct TypeTraitsTypeScalarTest;

template<typename Type>
struct TypeTraitsTypeScalarTest<Type,true>{
    typedef typename Type::F Result;
};
template<typename Type>
struct TypeTraitsTypeScalarTest<Type,false>{
    typedef  Type Result;
};

template<typename Type1,typename Type2,int isVect1,int isVect2>
struct ArithmeticsTraitTest;

template<typename Type1,typename Type2>
struct ArithmeticsTraitTest<Type1,Type2,false,false>
{
        typedef Type1 Result;
};
template<typename Type1,typename Type2>
struct ArithmeticsTraitTest<Type1,Type2,true,false>
{
    typedef typename TypeTraitsTypeScalar<Type1 >::Result Type1Scalar;
    typedef typename FunctionTypeTraitsSubstituteF<Type1,typename ArithmeticsTrait<Type1Scalar,Type2>::Result>::Result Result;
};
template<typename Type1,typename Type2>
struct ArithmeticsTraitTest<Type1,Type2,false, true>
{
    typedef typename TypeTraitsTypeScalar<Type2>::Result Type2Scalar;
    typedef typename FunctionTypeTraitsSubstituteF<Type2,typename ArithmeticsTrait<Type1,Type2Scalar>::Result>::Result Result;
};
template<typename Type1,typename Type2>
struct ArithmeticsTraitTest<Type1,Type2,true, true>
{
    typedef typename TypeTraitsTypeScalar<Type1 >::Result Type1Scalar;
    typedef typename TypeTraitsTypeScalar<Type2  >::Result Type2Scalar;
    typedef typename FunctionTypeTraitsSubstituteF<Type1,typename ArithmeticsTrait<Type1Scalar,Type2Scalar>::Result>::Result Result;
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
struct ArithmeticsTrait<UI16,F64>
{
    typedef F64 Result;
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
struct ArithmeticsTrait<I16,F64>
{
    typedef F64 Result;
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
struct ArithmeticsTrait<I32,F64>
{
    typedef F64 Result;
};
//F1=F64
template<>
struct ArithmeticsTrait<F64,UI8>
{
    typedef F64 Result;
};
template<>
struct ArithmeticsTrait<F64,UI16>
{
    typedef F64 Result;
};
template<>
struct ArithmeticsTrait<F64,I16>
{
    typedef F64 Result;
};

template<>
struct ArithmeticsTrait<F64,I32>
{
    typedef F64 Result;
};
template<>
struct ArithmeticsTrait<F64,F64>
{
    typedef F64 Result;
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
struct ArithmeticsSaturation< F64, F64>
{
    static F64 Range(F64 p)
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
