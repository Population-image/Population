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
#ifndef FUNCTIONPROCEDUREFUNCTORFOPTIMIZED_HPP
#define FUNCTIONPROCEDUREFUNCTORFOPTIMIZED_HPP
#include<popconfig.h>
#if defined(HAVE_OPENMP)
#include<data/mat/MatN.h>
namespace pop
{

template<typename PixelType1,typename PixelType2,typename IteratorENeighborhood,typename FunctorAccumulatorF>
void FunctionProcedureLocal(const MatN<2,PixelType1> & f, typename MatN<2,PixelType1>::IteratorEDomain & ,IteratorENeighborhood itn,FunctorAccumulatorF facc, MatN<2,PixelType2> & h)
{

    FunctionAssert(f,h,"FunctionProcedureFunctorBinaryFunctionE");
    Vec2I32 x;
#pragma omp parallel for firstprivate (itn,facc,x) shared(h,f)
    for(unsigned int i = 0;i<f.sizeI();i++){
        x(0)=i;
        for(unsigned int j = 0;j<f.sizeJ();j++)
        {
            x(1)=j;
            //std::cout<<x<<std::endl;
            itn.init(x);
            FunctionProcedureFunctorAccumulatorF(f,facc,itn);
            h(x)+=100;
        }
    }
}
template<typename PixelType1,typename PixelType2,typename IteratorENeighborhood,typename FunctorAccumulatorF>
void FunctionProcedureLocal(const MatN<3,PixelType1> & f, typename MatN<3,PixelType1>::IteratorEDomain & ,IteratorENeighborhood itn,FunctorAccumulatorF facc, MatN<3,PixelType2> & h)
{

    FunctionAssert(f,h,"FunctionProcedureFunctorBinaryFunctionE");
    Vec3I32 x;
#pragma omp parallel for firstprivate (itn,facc,x) shared(h,f)
    for(unsigned int i = 0;i<f.sizeI();i++){
        x(0)=i;
        for(unsigned int j = 0;j<f.sizeJ();j++)
        {
            x(1)=j;
            for(unsigned int k = 0;k<f.sizeK();k++){
                x(2)=k;
                itn.init(x);
                h(x) =FunctionProcedureFunctorAccumulatorF(f,facc,itn);
            }
        }
    }
}

}

#endif
#endif // FUNCTIONPROCEDUREFUNCTORFOPTIMIZED_HPP
