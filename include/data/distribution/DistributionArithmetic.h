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
#ifndef DISTRIBUTIONARITHMETIC_H
#define DISTRIBUTIONARITHMETIC_H
#include"data/distribution/Distribution.h"

namespace pop
{
/// @cond DEV
namespace Private {
class DistributionConcatenation
{
protected:
    Distribution * _fleft;
    Distribution * _fright;
public:
    DistributionConcatenation(const Distribution &f_left,const Distribution& f_right);
    DistributionConcatenation& operator=(const DistributionConcatenation&a);
    virtual ~DistributionConcatenation();
};
}

 class DistributionArithmeticAddition : public Distribution, public Private::DistributionConcatenation
 {
 public:
     DistributionArithmeticAddition(const Distribution &f_left,const Distribution& f_right);
    virtual DistributionArithmeticAddition * clone()const ;
    virtual F32 operator()(F32 value)const ;
    virtual F32 randomVariable()const;
 };
 class DistributionArithmeticSubtraction : public Distribution, public Private::DistributionConcatenation
 {
 public:
     DistributionArithmeticSubtraction(const Distribution &f_left,const Distribution& f_right);
    virtual DistributionArithmeticSubtraction * clone()const ;
    virtual F32 operator()(F32 value)const ;
    virtual F32 randomVariable()const;

 };
 class DistributionArithmeticMultiplication : public Distribution, public Private::DistributionConcatenation
 {
 public:
     DistributionArithmeticMultiplication(const Distribution &f_left,const Distribution& f_right);
    virtual DistributionArithmeticMultiplication * clone()const ;
    virtual F32 operator()(F32 value)const ;
    virtual F32 randomVariable()const;

 };
 class DistributionArithmeticDivision : public Distribution, public Private::DistributionConcatenation
 {
 public:
     DistributionArithmeticDivision(const Distribution &f_left,const Distribution& f_right);
    virtual DistributionArithmeticDivision * clone()const ;
    virtual F32 operator()(F32 value)const ;
    virtual F32 randomVariable()const;

 };
 class DistributionArithmeticComposition : public Distribution, public Private::DistributionConcatenation
 {
 public:
     DistributionArithmeticComposition(const Distribution &f_left,const Distribution& f_right);
    virtual DistributionArithmeticComposition * clone()const ;
    virtual F32 operator()(F32 value)const ;
    virtual F32 randomVariable()const;

 };
 class DistributionArithmeticMax : public Distribution, public Private::DistributionConcatenation
 {
 public:
     DistributionArithmeticMax(const Distribution &f_left,const Distribution& f_right);
    virtual DistributionArithmeticMax * clone()const ;
    virtual F32 operator()(F32 value)const ;
    virtual F32 randomVariable()const;

 };
 class DistributionArithmeticMin : public Distribution, public Private::DistributionConcatenation
 {

 public:
     DistributionArithmeticMin(const Distribution &f_left,const Distribution& f_right);
    virtual DistributionArithmeticMin * clone()const ;
    virtual F32 operator()(F32 value)const ;
    virtual F32 randomVariable()const;

 };
DistributionArithmeticAddition operator +(const Distribution &d1,const Distribution &d2);
DistributionArithmeticSubtraction operator -(const Distribution &d1,const Distribution &d2);
DistributionArithmeticMultiplication operator *(const Distribution &d1,const Distribution &d2);
DistributionArithmeticDivision operator /(const Distribution &d1,const Distribution &d2);

/*!
* \ingroup Distribution
* \param d1 input  Distribution
* \param d2 input  Distribution
* \return  distribution
*
*  h(x) = min(f(x),g(x))
* \code
        DistributionExpression d1("x^2");
        DistributionExpression d2("x^4");
        DistributionArithmeticMin dmin = maximum(d1,d2);//d1 is equal d1(x)=min(x^2,x^4)
        d1.display(-2,2);
* \endcode
*/
DistributionArithmeticMin minimum(const Distribution &d1,const Distribution &d2);
/*!
* \ingroup Distribution
* \param d1 other  Distribution
* \param d2 other  Distribution
* \return  distribution
*: public Distribution
*  h(x) = max(f(x),g(x))
* \code
        Distribution d1("x^2");
        Distribution d2("x^4");
        d1 = maximum(d1,d2);//d1 is equal d1(x)=max(x^2,x^4)
        d1.display(-2,2);
* \endcode
*/
DistributionArithmeticMax maximum(const Distribution &d1,const Distribution &d2);
DistributionArithmeticComposition f_rho_g(const Distribution &d1,const Distribution &d2);
/// @endcond
}
#endif // DISTRIBUTIONARITHMETIC_H
