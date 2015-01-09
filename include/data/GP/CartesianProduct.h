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


#ifndef CARTESIANPRODUCT_H
#define CARTESIANPRODUCT_H
#include"modules/GP/NullType.h"
#include"modules/GP/Typelist.h"
#include"modules/GP/TypelistMacros.h"
namespace pop
{
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

////////////////////////////////////////////////////////////////////////////////
// class template isTypeList
// Result is  true if the type is TypeList, false otherwise
////////////////////////////////////////////////////////////////////////////////

template<class T>
struct isTypeList
{
    enum {Result = false};
};
template <class Head2, class Tail2>
        struct isTypeList<Loki::Typelist<Head2, Tail2> >
{
    enum {Result = true};
};

namespace Private{
    template <bool flag1,bool flag2,class T1,class T2>
            struct AppendCase;

    template <bool flag2,class T1,class T2>
            struct AppendCase<true,flag2,T1,T2>
    {
        typedef typename Loki::TL::Append<T1,T2>::Result Result;
    };
    template <class T1,class T2>
            struct AppendCase<false,true,T1,T2>
    {
        typedef typename Loki::TL::Append<T2,T1>::Result Result;
    };
    template <class T1,class T2>
            struct AppendCase<false,false,T1,T2>
    {
        typedef LOKI_TYPELIST_2(T1,T2) Result;
    };
}
////////////////////////////////////////////////////////////////////////////////
// class template Append
// Appends a type or a typelist to another
// Invocation (T1 is either a type or a typelist and T2 is either a type or a typelist):
// Append<T1, T2>::Result
// returns a typelist that is the concatenation of T1 followed by T2 and NullType-terminated
////////////////////////////////////////////////////////////////////////////////
//T1 can be a TypeList or other and also for T2
template <class T1,class T2>
        struct Append
{
    typedef typename Private::AppendCase<isTypeList<T1>::Result,isTypeList<T2>::Result,T1,T2>::Result Result;
};
////////////////////////////////////////////////////////////////////////////////
// class template ConcaType2Type
// a Cartesian product  is the direct product of T1 and T2.
// The number of elements in T, Card(T), is equal to the product card(T1) by card(T2)
// Invocations : TList1 and TList2 are either a typelist or a single Type
///////////////////////////////////////////////////////////////////////////////

namespace Private
{
    template <class TList1,class TList2, class TLIST1MEMORY>
            struct ProductCartesianMemory;
    template <class TLIST1>
            struct ProductCartesianMemory<TLIST1,Loki::NullType,TLIST1>
    {
        typedef Loki::NullType Result;
    };
    template <class Head2, class Tail2,class TLIST1MEMORY>
            struct ProductCartesianMemory<Loki::NullType,Loki::Typelist<Head2, Tail2>,TLIST1MEMORY>
    {
        typedef typename ProductCartesianMemory<TLIST1MEMORY,Tail2,TLIST1MEMORY>::Result Result;
    };
    template <class Head1, class Tail1,class Head2, class Tail2,class TLIST1MEMORY>
            struct ProductCartesianMemory<Loki::Typelist<Head1, Tail1>,Loki::Typelist<Head2, Tail2>,TLIST1MEMORY>
    {
    private:
        typedef typename ProductCartesianMemory<Tail1,Loki::Typelist<Head2, Tail2>,TLIST1MEMORY>::Result typerecu;
        typedef typename Append<Head1,Head2>::Result typehere;
    public:
        typedef Loki::Typelist<typehere,typerecu  > Result;
    };
}
template <class TList1,class TList2>
        struct ProductCartesian
{
    typedef typename Private::ProductCartesianMemory< TList1, TList2,TList1>::Result Result;
};


template<typename TListTList,int index,typename Type>
struct FilterKeepTlistTlist;

template <int index,typename Type>
struct FilterKeepTlistTlist<Loki::NullType,index,Type>
{
    typedef Loki::NullType Result;
};
template <class Head, class Tail,int index,typename Type>
struct FilterKeepTlistTlist<Loki::Typelist<Head, Tail>,index,Type>
{
    typedef Loki::Typelist<Head,typename FilterKeepTlistTlist<Tail,index,Type>::Result > Type1;
    typedef typename FilterKeepTlistTlist<Tail,index,Type>::Result  Type2;
    enum{sametype=Loki::IsSameType<typename Loki::TL::TypeAt<Head,index>::Result,Type>::value};
    typedef typename Loki::Select<sametype ,  Type1 , Type2  >::Result  Result;
};
template<typename TListTList,int index,typename Type>
struct FilterRemoveTlistTlist;

template <int index,typename Type>
struct FilterRemoveTlistTlist<Loki::NullType,index,Type>
{
    typedef Loki::NullType Result;
};
template <class Head, class Tail,int index,typename Type>
struct FilterRemoveTlistTlist<Loki::Typelist<Head, Tail>,index,Type>
{
    typedef Loki::Typelist<Head,typename FilterRemoveTlistTlist<Tail,index,Type>::Result > Type1;
    typedef typename FilterRemoveTlistTlist<Tail,index,Type>::Result  Type2;
    enum{sametype=Loki::IsSameType<typename Loki::TL::TypeAt<Head,index>::Result,Type>::value};
    typedef typename Loki::Select<sametype ,  Type2,Type1  >::Result  Result;
};
}
#endif // CARTESIANPRODUCT_H
