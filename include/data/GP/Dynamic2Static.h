/******************************************************************************\
|*                   Population library for C++ X.X.X                         *|
|*----------------------------------------------------------------------------*|
The Population License is similar to the MIT license in adding this clause:
for any writing public or private that has resulted from the use of the
software population, the reference of this book "Population library, 2012,
Vincent Tariel" shall be included in it.

So, the terms of the Population License are:

Copyright © 2012, Tariel Vincent

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

#ifndef DYNAMIC2STATIC_H
#define DYNAMIC2STATIC_H


/******************************************************************************\
|*                   Population library for C++ X.X.X                         *|
|*----------------------------------------------------------------------------*|
The Population License is similar to the MIT license in adding this clause:
for any writing public or private that has resulted from the use of the
software population, the reference of this book "Population library, 2012,
Vincent Tariel" shall be included in it.

So, the terms of the Population License are:

Copyright © 2012, Tariel Vincent

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
#include <stdlib.h>
#include<string>
#include<iostream>
#include"data/GP/NullType.h"
#include"data/GP/Typelist.h"
#include"data/GP/TypelistMacros.h"
#include"data/GP/TypeTraitsTemplateTemplate.h"
#include"data/utility/Exception.h"
namespace pop
{

////////////////////////////////////////////////////////////////////////////////
// recursive template Dynamic2Static
// call the functor after the dynamic cast of the first parameter until 5 parameters
////////////////////////////////////////////////////////////////////////////////
template<typename T>
struct Dynamic2Static;
template <>
struct Dynamic2Static<Loki::NullType>
{
    template<typename P1,typename WhatEverT>
    static bool TestTypeInTList( P1 * , Loki::Type2Type<WhatEverT>)
    {
        return false;
    }

    template<typename P1,typename WhatEverT>
    static bool TestTypeInTList( P1 & , Loki::Type2Type<WhatEverT>)throw(pexception)
    {
        return false;
    }

    template<typename Functor,typename P1,typename WhatEverT>
    static void Switch( Functor &,P1 * , Loki::Type2Type<WhatEverT>)throw(pexception)
    {
        throw(pexception("PB: No type founded in the hierachy class during the Dynamic2Static call"));
    }
    template<typename Functor,typename P1,typename WhatEverT>
    static void Switch( Functor &,P1 &, Loki::Type2Type<WhatEverT>)throw(pexception)
    {
        throw(pexception("PB: No type founded in the hierachy class during the Dynamic2Static call"));
    }

    template<typename Functor,typename P1,typename P2,typename WhatEverT>
    static void Switch( Functor &,P1 * , P2&, Loki::Type2Type<WhatEverT>)throw(pexception)
    {
        throw(pexception("PB: No type founded in the hierachy class during the Dynamic2Static call"));
    }
    template<typename Functor,typename P1,typename P2, typename WhatEverT>
    static void Switch( Functor &,P1 &,P2 & , Loki::Type2Type<WhatEverT>)throw(pexception)
    {
        throw(pexception("PB: No type founded in the hierachy class during the Dynamic2Static call"));
    }
    template<typename Functor,typename P1,typename P2,typename P3,typename WhatEverT>
    static void Switch( Functor &,P1 * , P2&, P3 &, Loki::Type2Type<WhatEverT>)throw(pexception)
    {
        throw(pexception("PB: No type founded in the hierachy class during the Dynamic2Static call"));
    }
    template<typename Functor,typename P1,typename P2,typename P3, typename WhatEverT>
    static void Switch( Functor &,P1 &,P2 & , P3 &,Loki::Type2Type<WhatEverT>)throw(pexception)
    {
        throw(pexception("PB: No type founded in the hierachy class during the Dynamic2Static call"));
    }
    template<typename Functor,typename P1,typename P2,typename P3,typename P4,typename WhatEverT>
    static void Switch( Functor &,P1 * , P2&, P3 &,P4 &,Loki::Type2Type<WhatEverT>)throw(pexception)
    {
        throw(pexception("PB: No type founded in the hierachy class during the Dynamic2Static call"));
    }
    template<typename Functor,typename P1,typename P2,typename P3, typename P4,typename WhatEverT>
    static void Switch( Functor &,P1 &,P2 & , P3 &,P4 &,Loki::Type2Type<WhatEverT>)throw(pexception)
    {
        throw(pexception("PB: No type founded in the hierachy class during the Dynamic2Static call"));
    }
    template<typename Functor,typename P1,typename P2,typename P3,typename P4,typename P5,typename WhatEverT>
    static void Switch( Functor &,P1 * , P2&, P3 &,P4 &,P5&,Loki::Type2Type<WhatEverT>)throw(pexception)
    {
        throw(pexception("PB: No type founded in the hierachy class during the Dynamic2Static call"));
    }
    template<typename Functor,typename P1,typename P2,typename P3, typename P4,typename P5,typename WhatEverT>
    static void Switch( Functor &,P1 &,P2 & , P3 &,P4 &,P5&,Loki::Type2Type<WhatEverT>)throw(pexception)
    {
        throw(pexception("PB: No type founded in the hierachy class during the Dynamic2Static call"));
    }

    template<typename Functor,typename P1,typename P2,typename P3,typename P4,typename P5,typename P6,typename WhatEverT>
    static void Switch( Functor &,P1 * , P2&, P3 &,P4 &,P5&,P6&,Loki::Type2Type<WhatEverT>)throw(pexception)
    {
        throw(pexception("PB: No type founded in the hierachy class during the Dynamic2Static call"));
    }
    template<typename Functor,typename P1,typename P2,typename P3, typename P4,typename P5,typename P6,typename WhatEverT>
    static void Switch( Functor &,P1 &,P2 & , P3 &,P4 &,P5&,P6&,Loki::Type2Type<WhatEverT>)throw(pexception)
    {
        throw(pexception("PB: No type founded in the hierachy class during the Dynamic2Static call"));
    }

    template<typename Functor,typename P1,typename P2,typename P3,typename P4,typename P5,typename P6,typename P7,typename WhatEverT>
    static void Switch( Functor &,P1 * , P2&, P3 &,P4 &,P5&,P6&,P7&,Loki::Type2Type<WhatEverT>)throw(pexception)
    {
        throw(pexception("PB: No type founded in the hierachy class during the Dynamic2Static call"));
    }
    template<typename Functor,typename P1,typename P2,typename P3, typename P4,typename P5,typename P6,typename P7,typename WhatEverT>
    static void Switch( Functor &,P1 &,P2 & , P3 &,P4 &,P5&,P6&,P7&,Loki::Type2Type<WhatEverT>)throw(pexception)
    {
        throw(pexception("PB: No type founded in the hierachy class during the Dynamic2Static call"));
    }

};
template <class Head, class Tail>
struct Dynamic2Static<Loki::Typelist< Head, Tail> >
{
    template<typename P1,typename WhatEverT>
    static bool TestTypeInTList( P1 * p, Loki::Type2Type<WhatEverT> t)
    {
        typedef typename  SubstituteTemplateParameter<Head, WhatEverT>::Result T;
        if( dynamic_cast<T *>(p) )
            return true;
        else
           return Dynamic2Static<Tail>::TestTypeInTList(p,t);
    }

    template<typename P1,typename WhatEverT>
    static bool TestTypeInTList( P1 & p, Loki::Type2Type<WhatEverT> t)
    {
        typedef typename  SubstituteTemplateParameter<Head, WhatEverT>::Result T;
        if( dynamic_cast<T *>(&p) )
            return true;
        else
           return Dynamic2Static<Tail>::TestTypeInTList(p,t);
    }

    template<typename Functor,typename P1,typename WhatEverT>
    static void Switch( Functor &f,P1 *  p1, Loki::Type2Type<WhatEverT> t)throw(pexception)
    {
        typedef typename  SubstituteTemplateParameter<Head, WhatEverT>::Result T;
        if( T * d=dynamic_cast<T *>(p1) )
            f(d);
        else
            Dynamic2Static<Tail>::Switch(f,p1,t);
    }
    template<typename Functor,typename P1,typename WhatEverT>
    static void Switch( Functor &f,P1 & p1, Loki::Type2Type<WhatEverT> t)throw(pexception)
    {
        typedef typename  SubstituteTemplateParameter<Head, WhatEverT>::Result T;
        if( T * d=dynamic_cast<T *>(&p1) )
            f(*d);
        else
            Dynamic2Static<Tail>::Switch(f,p1,t);
    }

    template<typename Functor,typename P1,typename P2,typename WhatEverT>
    static void Switch( Functor &f,P1 *  p1,P2 & p2, Loki::Type2Type<WhatEverT> t)throw(pexception)
    {
        typedef typename  SubstituteTemplateParameter<Head, WhatEverT>::Result T;
        if( T * d=dynamic_cast<T *>(p1) )
            f(d,p2);
        else
            Dynamic2Static<Tail>::Switch(f,p1,p2,t);
    }
    template<typename Functor,typename P1,typename P2,typename WhatEverT>
    static void Switch( Functor &f,P1 & p1,P2 & p2, Loki::Type2Type<WhatEverT> t)throw(pexception)
    {
        typedef typename  SubstituteTemplateParameter<Head, WhatEverT>::Result T;
        if( T * d=dynamic_cast<T *>(&p1) )
            f(*d,p2);
        else
            Dynamic2Static<Tail>::Switch(f,p1,p2,t);
    }

    template<typename Functor,typename P1,typename P2,typename P3,typename WhatEverT>
    static void Switch( Functor &f,P1 *  p1,P2 & p2,P3 & p3, Loki::Type2Type<WhatEverT> t)throw(pexception)
    {
        typedef typename  SubstituteTemplateParameter<Head, WhatEverT>::Result T;
        if( T * d=dynamic_cast<T *>(p1) )
            f(d,p2,p3);
        else
            Dynamic2Static<Tail>::Switch(f,p1,p2,p3,t);
    }

    template<typename Functor,typename P1,typename P2,typename P3,typename WhatEverT>
    static void Switch( Functor &f,P1 & p1,P2 & p2,P3 & p3,  Loki::Type2Type<WhatEverT> t)throw(pexception)
    {
        typedef typename  SubstituteTemplateParameter<Head, WhatEverT>::Result T;
        if( T * d=dynamic_cast<T *>(&p1) )
            f(*d,p2,p3);
        else
            Dynamic2Static<Tail>::Switch(f,p1,p2,p3,t);
    }

    template<typename Functor,typename P1,typename P2,typename P3,typename P4,typename WhatEverT>
    static void Switch( Functor &f,P1 *  p1,P2 & p2,P3 & p3,P4 & p4, Loki::Type2Type<WhatEverT> t)throw(pexception)
    {
        typedef typename  SubstituteTemplateParameter<Head, WhatEverT>::Result T;
        if( T * d=dynamic_cast<T *>(p1) )
            f(d,p2,p3,p4);
        else
            Dynamic2Static<Tail>::Switch(f,p1,p2,p3,p4,t);
    }

    template<typename Functor,typename P1,typename P2,typename P3,typename P4,typename WhatEverT>
    static void Switch( Functor &f,P1 & p1,P2 & p2,P3 & p3, P4 & p4, Loki::Type2Type<WhatEverT> t)throw(pexception)
   {
        typedef typename  SubstituteTemplateParameter<Head, WhatEverT>::Result T;
        if( T * d=dynamic_cast<T *>(&p1) )
            f(*d,p2,p3,p4);
        else
            Dynamic2Static<Tail>::Switch(f,p1,p2,p3,p4,t);
    }

    template<typename Functor,typename P1,typename P2,typename P3,typename P4,typename P5,typename WhatEverT>
    static void Switch( Functor &f,P1 *  p1,P2 & p2,P3 & p3,P4 & p4, P5 & p5, Loki::Type2Type<WhatEverT> t)throw(pexception)
    {
        typedef typename  SubstituteTemplateParameter<Head, WhatEverT>::Result T;
        if( T * d=dynamic_cast<T *>(p1) )
            f(d,p2,p3,p4,p5);
        else
            Dynamic2Static<Tail>::Switch(f,p1,p2,p3,p4,p5,t);
    }

    template<typename Functor,typename P1,typename P2,typename P3,typename P4,typename P5,typename WhatEverT>
    static void Switch( Functor &f,P1 & p1,P2 & p2,P3 & p3, P4 & p4,P5 & p5, Loki::Type2Type<WhatEverT> t)throw(pexception)
   {
        typedef typename  SubstituteTemplateParameter<Head, WhatEverT>::Result T;
        if( T * d=dynamic_cast<T *>(&p1) )
            f(*d,p2,p3,p4,p5);
        else
            Dynamic2Static<Tail>::Switch(f,p1,p2,p3,p4,p5,t);
    }

    template<typename Functor,typename P1,typename P2,typename P3,typename P4,typename P5,typename P6,typename WhatEverT>
    static void Switch( Functor &f,P1 *  p1,P2 & p2,P3 & p3,P4 & p4, P5 & p5, P6 & p6,Loki::Type2Type<WhatEverT> t)throw(pexception)
    {
        typedef typename  SubstituteTemplateParameter<Head, WhatEverT>::Result T;
        if( T * d=dynamic_cast<T *>(p1) )
            f(d,p2,p3,p4,p5,p6);
        else
            Dynamic2Static<Tail>::Switch(f,p1,p2,p3,p4,p5,p6,t);
    }

    template<typename Functor,typename P1,typename P2,typename P3,typename P4,typename P5,typename P6,typename WhatEverT>
    static void Switch( Functor &f,P1 & p1,P2 & p2,P3 & p3, P4 & p4,P5 & p5,P6 & p6, Loki::Type2Type<WhatEverT> t)throw(pexception)
   {
        typedef typename  SubstituteTemplateParameter<Head, WhatEverT>::Result T;
        if( T * d=dynamic_cast<T *>(&p1) )
            f(*d,p2,p3,p4,p5,p6);
        else
            Dynamic2Static<Tail>::Switch(f,p1,p2,p3,p4,p5,p6,t);
    }
    template<typename Functor,typename P1,typename P2,typename P3,typename P4,typename P5,typename P6,typename P7,typename WhatEverT>
    static void Switch( Functor &f,P1 *  p1,P2 & p2,P3 & p3,P4 & p4, P5 & p5, P6 & p6,P7 & p7,Loki::Type2Type<WhatEverT> t)throw(pexception)
    {
        typedef typename  SubstituteTemplateParameter<Head, WhatEverT>::Result T;
        if( T * d=dynamic_cast<T *>(p1) )
            f(d,p2,p3,p4,p5,p6,p7);
        else
            Dynamic2Static<Tail>::Switch(f,p1,p2,p3,p4,p5,p6,p7,t);
    }

    template<typename Functor,typename P1,typename P2,typename P3,typename P4,typename P5,typename P6,typename P7,typename WhatEverT>
    static void Switch( Functor &f,P1 & p1,P2 & p2,P3 & p3, P4 & p4,P5 & p5,P6 & p6,P7 & p7, Loki::Type2Type<WhatEverT> t)throw(pexception)
   {
        typedef typename  SubstituteTemplateParameter<Head, WhatEverT>::Result T;
        if( T * d=dynamic_cast<T *>(&p1) )
            f(*d,p2,p3,p4,p5,p6,p7);
        else
            Dynamic2Static<Tail>::Switch(f,p1,p2,p3,p4,p5,p6,p7,t);
    }
};
}
#endif // DYNAMIC2STATIC_H
