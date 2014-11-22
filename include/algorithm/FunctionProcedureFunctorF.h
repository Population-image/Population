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
#ifndef FUNCTIONPROCEDUREFUNCTORF_HPP
#define FUNCTIONPROCEDUREFUNCTORF_HPP
namespace pop
{
template<
        typename Generator,
        typename Iterator,
        typename Function2
        >
void FunctionProcedureGenerator(Generator & func, Iterator & it, Function2 & g)
{
    while(it.next())
    {
        g(it.x())=func();
    }
}

template<typename Function1,typename FunctorUnaryF1,typename Iterator,typename Function2>
void FunctionProcedureFunctorUnaryF(const Function1 & f,  FunctorUnaryF1 & func, Iterator & it, Function2 & h)throw(pexception)
{
    FunctionAssert(f,h,"FunctionProcedureFunctorUnaryF  ");
    while(it.next())
    {
        h(it.x())=func( f(it.x()));
    }
}

template<
        typename Function1,
        typename Function2,
        typename FunctorBinaryF2,
        typename Iterator,
        typename Function3
        >
void FunctionProcedureFunctorBinaryF2(const Function1 & f, const Function2 g, FunctorBinaryF2 &func, Iterator & it, Function3 & h)throw(pexception)
{
    FunctionAssert(f,h,"FunctionProcedureFunctorBinaryF2");
    FunctionAssert(g,h,"FunctionProcedureFunctorBinaryF2");

    while(it.next()){
        h(it.x())=func( f(it.x()), g(it.x()) );
    }
}
template<typename Function1,typename FunctorAccumulatorF,typename Iterator>
typename FunctorAccumulatorF::ReturnType FunctionProcedureFunctorAccumulatorF(const Function1 & f,  FunctorAccumulatorF & func, Iterator & it)
{
    func.init();
    while(it.next())
    {
        func( f(it.x()));
    }
    return func.getValue();
}

template<typename Function1,typename IteratorEGlobal,typename IteratorENeighborhood,typename FunctorAccumulatorF, typename Function2>
void FunctionProcedureLocal(const Function1 & f,  IteratorEGlobal & itg,IteratorENeighborhood & itn,FunctorAccumulatorF facc, Function2 & h)
{
    FunctionAssert(f,h,"FunctionProcedureFunctorBinaryFunctionE");
    while(itg.next()){
        itn.init(itg.x());
        h(itg.x())=FunctionProcedureFunctorAccumulatorF(f,facc,itn);
    }
}




}


#endif // FUNCTIONPROCEDUREFUNCTORF_HPP
