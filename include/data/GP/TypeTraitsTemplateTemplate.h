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
#ifndef TYPETRAITSTEMPLATETEMPLATE_H
#define TYPETRAITSTEMPLATETEMPLATE_H
namespace pop
{
#include"data/GP/NullType.h"
#include"data/GP/Typelist.h"
#include"data/GP/TypelistMacros.h"






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
// Type traits class  SubstituteTemplateParameter
// To avoid the template template idiom, we use a classical template classs instantied
// with blank parameter that we subsitute with those contained in TList
// Restriction: until 4 parameter whatever the type or non-type
////////////////////////////////////////////////////////////////////////////////
template<typename TList,typename T>
struct SubstituteTemplateParameter;

//For one template parameter
template<typename T1, typename IN1, template <typename>  class T>
struct SubstituteTemplateParameter<LOKI_TYPELIST_1(T1), T<IN1> >
{
    typedef T<T1> Result;
};

template<typename T1, int IN1, template <int>  class T>
struct SubstituteTemplateParameter<LOKI_TYPELIST_1(T1), T<IN1> >
{
    typedef T<T1::value> Result;
};

//For two template parameters
template<typename T1, typename T2, typename IN1, typename IN2, template <typename,typename>  class T>
struct SubstituteTemplateParameter<LOKI_TYPELIST_2(T1,T2), T<IN1,IN2> >
{
    typedef T<T1,T2> Result;
};


template<typename T1, typename T2, int IN1,typename IN2, template <int,typename>  class T>
struct SubstituteTemplateParameter<LOKI_TYPELIST_2(T1,T2), T<IN1,IN2> >
{
    typedef T<T1::value,T2> Result;
};
template<typename T1, typename T2, typename IN1, int IN2, template <typename,int>  class T>
struct SubstituteTemplateParameter<LOKI_TYPELIST_2(T1,T2), T<IN1,IN2> >
{
    typedef T<T1,T2::value> Result;
};


template<typename T1,typename T2  , int IN1,int IN2, template <int,int>  class T>
struct SubstituteTemplateParameter<LOKI_TYPELIST_2(T1,T2), T<IN1,IN2> >
{
    typedef T<T1::value,T2::value> Result;
};


//For three template parameters
template<typename T1,typename T2,typename T3  , typename IN1,typename IN2,typename IN3, template <typename,typename,typename>  class T>
struct SubstituteTemplateParameter<LOKI_TYPELIST_3(T1,T2,T3), T<IN1,IN2,IN3> >
{
    typedef T<T1,T2,T3> Result;
};
template<typename T1,typename T2,typename T3  , int IN1,typename IN2,typename IN3, template <int,typename,typename>  class T>
struct SubstituteTemplateParameter<LOKI_TYPELIST_3(T1,T2,T3), T<IN1,IN2,IN3> >
{
    typedef T<T1::value,T2,T3> Result;
};
template<typename T1,typename T2,typename T3  , typename IN1,int IN2,typename IN3, template <typename,int,typename>  class T>
struct SubstituteTemplateParameter<LOKI_TYPELIST_3(T1,T2,T3), T<IN1,IN2,IN3> >
{
    typedef T<T1,T2::value,T3> Result;
};
template<typename T1,typename T2,typename T3  , typename IN1, typename IN2,int IN3, template <typename,typename,int>  class T>
struct SubstituteTemplateParameter<LOKI_TYPELIST_3(T1,T2,T3), T<IN1,IN2,IN3> >
{
    typedef T<T1,T2,T3::value> Result;
};

template<typename T1,typename T2,typename T3  , int IN1,int IN2,typename IN3, template <int,int,typename>  class T>
struct SubstituteTemplateParameter<LOKI_TYPELIST_3(T1,T2,T3), T<IN1,IN2,IN3> >
{
    typedef T<T1::value,T2::value,T3> Result;
};
template<typename T1,typename T2,typename T3  , int IN1, typename IN2,int IN3, template <int,typename,int>  class T>
struct SubstituteTemplateParameter<LOKI_TYPELIST_3(T1,T2,T3), T<IN1,IN2,IN3> >
{
    typedef T<T1::value,T2,T3::value> Result;
};
template<typename T1,typename T2,typename T3  , typename IN1, int IN2,int IN3, template <typename,int,int>  class T>
struct SubstituteTemplateParameter<LOKI_TYPELIST_3(T1,T2,T3), T<IN1,IN2,IN3> >
{
    typedef T<T1,T2::value,T3::value> Result;
};
template<typename T1,typename T2,typename T3  , int IN1, int IN2,int IN3, template <int,int,int>  class T>
struct SubstituteTemplateParameter<LOKI_TYPELIST_3(T1,T2,T3), T<IN1,IN2,IN3> >
{
    typedef T<T1::value,T2::value,T3::value> Result;
};



//For four template parameters
template<typename T1,typename T2,typename T3,typename T4  , typename IN1,typename IN2,typename IN3,typename IN4, template <typename,typename,typename,typename>  class T>
struct SubstituteTemplateParameter<LOKI_TYPELIST_4(T1,T2,T3,T4), T<IN1,IN2,IN3,IN4> >
{
    typedef T<T1,T2,T3,T4> Result;
};
template<typename T1,typename T2,typename T3,typename T4  , int IN1,typename IN2,typename IN3,typename IN4, template <int,typename,typename,typename>  class T>
struct SubstituteTemplateParameter<LOKI_TYPELIST_4(T1,T2,T3,T4), T<IN1,IN2,IN3,IN4> >
{
    typedef T<T1::value,T2,T3,T4> Result;
};
template<typename T1,typename T2,typename T3,typename T4  , typename IN1,int IN2,typename IN3,typename IN4, template <typename,int,typename,typename>  class T>
struct SubstituteTemplateParameter<LOKI_TYPELIST_4(T1,T2,T3,T4), T<IN1,IN2,IN3,IN4> >
{
    typedef T<T1,T2::value,T3,T4> Result;
};
template<typename T1,typename T2,typename T3,typename T4  , typename IN1,typename IN2,int IN3,typename IN4, template <typename,typename,int,typename>  class T>
struct SubstituteTemplateParameter<LOKI_TYPELIST_4(T1,T2,T3,T4), T<IN1,IN2,IN3,IN4> >
{
    typedef T<T1,T2,T3::value,T4> Result;
};
template<typename T1,typename T2,typename T3,typename T4  , typename IN1,typename IN2,typename IN3,int IN4, template <typename,typename,typename,int>  class T>
struct SubstituteTemplateParameter<LOKI_TYPELIST_4(T1,T2,T3,T4), T<IN1,IN2,IN3,IN4> >
{
    typedef T<T1,T2,T3,T4::value> Result;
};

template<typename T1,typename T2,typename T3,typename T4  , int IN1,int IN2,typename IN3,typename IN4, template <int,int,typename,typename>  class T>
struct SubstituteTemplateParameter<LOKI_TYPELIST_4(T1,T2,T3,T4), T<IN1,IN2,IN3,IN4> >
{
    typedef T<T1::value,T2::value,T3,T4> Result;
};
template<typename T1,typename T2,typename T3,typename T4  , int IN1, typename IN2,int IN3,typename IN4, template <int,typename,int,typename>  class T>
struct SubstituteTemplateParameter<LOKI_TYPELIST_4(T1,T2,T3,T4), T<IN1,IN2,IN3,IN4> >
{
    typedef T<T1::value,T2,T3::value,T4> Result;
};
template<typename T1,typename T2,typename T3,typename T4  , int IN1, typename IN2, typename IN3,int IN4, template <int,typename,typename,int>  class T>
struct SubstituteTemplateParameter<LOKI_TYPELIST_4(T1,T2,T3,T4), T<IN1,IN2,IN3,IN4> >
{
    typedef T<T1::value,T2,T3,T4::value> Result;
};

template<typename T1,typename T2,typename T3,typename T4  , typename IN1, int IN2, int IN3,typename IN4, template <typename,int,int,typename>  class T>
struct SubstituteTemplateParameter<LOKI_TYPELIST_4(T1,T2,T3,T4), T<IN1,IN2,IN3,IN4> >
{
    typedef T<T1,T2::value,T3::value,T4> Result;
};
template<typename T1,typename T2,typename T3,typename T4  , typename IN1, int IN2, typename  IN3,int IN4, template <typename,int,typename,int>  class T>
struct SubstituteTemplateParameter<LOKI_TYPELIST_4(T1,T2,T3,T4), T<IN1,IN2,IN3,IN4> >
{
    typedef T<T1,T2::value,T3,T4::value> Result;
};
template<typename T1,typename T2,typename T3,typename T4  , typename IN1, typename IN2, int  IN3,int IN4, template <typename,typename,int,int>  class T>
struct SubstituteTemplateParameter<LOKI_TYPELIST_4(T1,T2,T3,T4), T<IN1,IN2,IN3,IN4> >
{
    typedef T<T1,T2,T3::value,T4::value> Result;
};
template<typename T1,typename T2,typename T3,typename T4  , typename IN1, int IN2, int  IN3,int IN4, template <typename,int,int,int>  class T>
struct SubstituteTemplateParameter<LOKI_TYPELIST_4(T1,T2,T3,T4), T<IN1,IN2,IN3,IN4> >
{
    typedef T<T1,T2::value,T3::value,T4::value> Result;
};

template<typename T1,typename T2,typename T3,typename T4  , int IN1, typename IN2, int  IN3,int IN4, template <int,typename,int,int>  class T>
struct SubstituteTemplateParameter<LOKI_TYPELIST_4(T1,T2,T3,T4), T<IN1,IN2,IN3,IN4> >
{
    typedef T<T1::value,T2,T3::value,T4::value> Result;
};

template<typename T1,typename T2,typename T3,typename T4  , int IN1, int IN2,  typename  IN3,int IN4, template <int,int,typename,int>  class T>
struct SubstituteTemplateParameter<LOKI_TYPELIST_4(T1,T2,T3,T4), T<IN1,IN2,IN3,IN4> >
{
    typedef T<T1::value,T2::value,T3,T4::value> Result;
};

template<typename T1,typename T2,typename T3,typename T4  , int IN1, int IN2, int  IN3,typename IN4, template <int,int,int,typename>  class T>
struct SubstituteTemplateParameter<LOKI_TYPELIST_4(T1,T2,T3,T4), T<IN1,IN2,IN3,IN4> >
{
    typedef T<T1::value,T2::value,T3::value,T4> Result;
};
template<typename T1,typename T2,typename T3,typename T4  , int IN1, int IN2, int  IN3,int IN4, template <int,int,int,int>  class T>
struct SubstituteTemplateParameter<LOKI_TYPELIST_4(T1,T2,T3,T4), T<IN1,IN2,IN3,IN4> >
{
    typedef T<T1::value,T2::value,T3::value,T4::value> Result;
};

//For five template parameters, only types
template<typename T1,typename T2,typename T3,typename T4,typename T5  , typename IN1,typename IN2,typename IN3,typename IN4,typename IN5, template <typename,typename,typename,typename,typename>  class T>
struct SubstituteTemplateParameter<LOKI_TYPELIST_5(T1,T2,T3,T4,T5), T<IN1,IN2,IN3,IN4,IN5> >
{
    typedef T<T1,T2,T3,T4,T5> Result;
};


template<typename TList>
struct MultipleInheritanceFunctionnality;


template<typename T1>
struct MultipleInheritanceFunctionnality<LOKI_TYPELIST_1(T1)>:T1
{
    typedef LOKI_TYPELIST_1(T1) TList;
    MultipleInheritanceFunctionnality()
    { }
    template<typename T2>
    MultipleInheritanceFunctionnality(MultipleInheritanceFunctionnality<LOKI_TYPELIST_2(T1,T2)>& a)
        :T1(a)
    {}

};
template<typename T1,typename T2>
struct MultipleInheritanceFunctionnality<LOKI_TYPELIST_2(T1,T2)>:T1,T2
{
    typedef LOKI_TYPELIST_2(T1,T2) TList;
    MultipleInheritanceFunctionnality()
    {}
    MultipleInheritanceFunctionnality(MultipleInheritanceFunctionnality<LOKI_TYPELIST_1(T1)>& a)
        :T1(a) {}
    template<typename T3>
    MultipleInheritanceFunctionnality(MultipleInheritanceFunctionnality<LOKI_TYPELIST_3(T1,T2,T3)>& a)
        :T1(a),T2(a){}
};
template<typename T1,typename T2,typename T3>
struct MultipleInheritanceFunctionnality<LOKI_TYPELIST_3(T1,T2,T3)>:T1,T2,T3
{
    typedef LOKI_TYPELIST_3(T1,T2,T3) TList;
    MultipleInheritanceFunctionnality()
    {}
    MultipleInheritanceFunctionnality(MultipleInheritanceFunctionnality<LOKI_TYPELIST_1(T1)>& a)
        :T1(a) {}
    MultipleInheritanceFunctionnality(MultipleInheritanceFunctionnality<LOKI_TYPELIST_2(T1,T2)>& a)
        :T1(a),T2(a) {}

    template<typename T4>
    MultipleInheritanceFunctionnality(MultipleInheritanceFunctionnality<LOKI_TYPELIST_4(T1,T2,T3,T4)>& a)
        :T1(a),T2(a),T3(a){}
};
template<typename T1,typename T2,typename T3,typename T4>
struct MultipleInheritanceFunctionnality<LOKI_TYPELIST_4(T1,T2,T3,T4)>:T1,T2,T3,T4
{
        typedef LOKI_TYPELIST_4(T1,T2,T3,T4) TList;
};
namespace Private {
template<int iter,int DIM>
struct Loop
{

    template<typename Charactristics,typename Functor>
    static void serialize( Charactristics &c,Functor& f){
        typedef typename Charactristics::TList Tlist;
        typedef typename Loki::TL::TypeAt<Tlist,iter>::Result Charactristic;
        Charactristic * c_cast = dynamic_cast<Charactristic*>(&c);
        f(* c_cast);
        Loop <iter+1,DIM>::serialize(c,f);
    }

    template<typename Charactristics,typename Functor, typename Arg1>
    static void serialize( Charactristics &c,Functor& f, Arg1& a1){
        typedef typename Charactristics::TList Tlist;
        typedef typename Loki::TL::TypeAt<Tlist,iter>::Result Charactristic;
        Charactristic * c_cast = dynamic_cast<Charactristic*>(&c);
        f(* c_cast,a1);
        Loop <iter+1,DIM>::serialize(c,f,a1);
    }
    template<typename Charactristics,typename Functor, typename Arg1,typename Arg2>
    static void serialize( Charactristics &c,Functor& f, Arg1& a1,Arg2& a2){
        typedef typename Charactristics::TList Tlist;
        typedef typename Loki::TL::TypeAt<Tlist,iter>::Result Charactristic;
        Charactristic * c_cast = dynamic_cast<Charactristic*>(&c);
        f(* c_cast,a1,a2);
        Loop <iter+1,DIM>::serialize(c,f,a1,a2);
    }
};
template<int DIM>
struct Loop <DIM,DIM>
{
    template<typename Charactristics,typename Functor>
    static void serialize( Charactristics&, Functor& ){}
    template<typename Charactristics,typename Functor,typename Arg1>
    static void serialize( Charactristics&, Functor& , Arg1&){}
    template<typename Charactristics,typename Functor,typename Arg1,typename Arg2>
    static void serialize( Charactristics&, Functor& , Arg1&, Arg2&){}
};
}
template<typename Charactristics,typename Functor,typename Arg1,typename Arg2>
void serializeMultipleInheritanceFunctionnality(Charactristics &c,Functor& f, Arg1& a1,Arg2& a2){
    Private::Loop<0,Loki::TL::Length<typename Charactristics::TList>::value  > loop;
    loop.serialize(c,f,a1,a2);
}

}


#endif // TYPETRAITSTEMPLATETEMPLATE_H
