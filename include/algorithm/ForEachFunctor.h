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
#ifndef FUNCTIONPROCEDUREFUNCTORF_HPP
#define FUNCTIONPROCEDUREFUNCTORF_HPP
#include"data/vec/VecN.h"
#include"data/mat/MatNIteratorE.h"
namespace pop
{
template<typename Function_E_F,typename Generator_F,typename IteratorE>
void forEachFunctorGenerator(Function_E_F & g,Generator_F  func, IteratorE it){
    while(it.next()){
        g(it.x())=func();
    }
}
template<typename Function_E_F,typename Generator_F>
void forEachFunctorGenerator(Function_E_F & g,Generator_F  func){
    typename Function_E_F::IteratorEDomain it=g.getIteratorEDomain();
    forEachFunctorGenerator(g,func,it);
}
template<typename Function1_E_F,typename Function2_E_F,typename FunctorUnary_F_F,typename IteratorE>
void forEachFunctorUnaryF(const Function1_E_F & f,Function2_E_F & g,FunctorUnary_F_F & func, IteratorE  it){
    while(it.next()){
        g(it.x())=func(f(it.x()));
    }
}
template<typename Function1_E_F,typename Function2_E_F,typename FunctorUnary_F_F>
void forEachFunctorUnaryF(const Function1_E_F & f,Function2_E_F & g,FunctorUnary_F_F & func){
    typename Function1_E_F::IteratorEDomain it=f.getIteratorEDomain();
    forEachFunctorUnaryF(f,g,func,it);
}
template<typename Function1_E_F,typename Function2_E_F,typename Function3_E_F,typename FunctorBinaryFF,typename IteratorE>
void forEachFunctorBinaryFF(const Function1_E_F & f,const Function2_E_F & g,Function3_E_F & h,FunctorBinaryFF  func, IteratorE it){
    while(it.next()){
        h(it.x())=func(f(it.x()),g(it.x()));
    }
}
template<typename Function1_E_F,typename Function2_E_F,typename Function3_E_F,typename FunctorBinaryFF>
void forEachFunctorBinaryFF(const Function1_E_F & f,const Function2_E_F & g,Function3_E_F & h,FunctorBinaryFF  func){
    typename Function1_E_F::IteratorEDomain it=f.getIteratorEDomain();
    forEachFunctorBinaryFF(f,g,h,func,it);
}
template<typename Function1_E_F,typename Function2_E_F,typename FunctorBinaryFunctionE,typename IteratorE>
void forEachFunctorBinaryFunctionE(const Function1_E_F & f, Function2_E_F & h, FunctorBinaryFunctionE & func, IteratorE  it){
    while(it.next()){
        h(it.x())=func( f, it.x());
    }
}
template<typename Function1_E_F,typename Function2_E_F,typename FunctorBinaryFunctionE>
void forEachFunctorBinaryFunctionE(const Function1_E_F & f, Function2_E_F & h, FunctorBinaryFunctionE & func){
    typename Function1_E_F::IteratorEDomain it=f.getIteratorEDomain();
    forEachFunctorBinaryFunctionE(f,h,func,it);
}

template<typename Function_E_F,typename FunctorAccumulatorF,typename IteratorE>
typename FunctorAccumulatorF::ReturnType forEachFunctorAccumulator(const Function_E_F & f,  FunctorAccumulatorF & func, IteratorE & it){
    func.init();
    while(it.next()){
        func( f(it.x()));
    }
    return func.getValue();
}
template<typename Function1_E_F,typename Function2_E_F,typename FunctorAccumulatorF,typename IteratorEGlobal,typename IteratorELocal>
void forEachGlobalToLocal(const Function1_E_F & f, Function2_E_F &  h, FunctorAccumulatorF facc,IteratorELocal  it_local, IteratorEGlobal it_global){
    while(it_global.next()){
        it_local.init(it_global.x());
        h(it_global.x())=forEachFunctorAccumulator(f,facc,it_local);
    }
}
template<typename Function1_E_F,typename Function2_E_F,typename FunctorAccumulatorF,typename IteratorELocal>
void forEachGlobalToLocal(const Function1_E_F & f, Function2_E_F &  h, FunctorAccumulatorF facc,IteratorELocal it_local){
    typename Function1_E_F::IteratorEDomain it_global=f.getIteratorEDomain();
    forEachGlobalToLocal( f,  h,  facc, it_local,it_global);
}
}
#endif // FUNCTIONPROCEDUREFUNCTORF_HPP
