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
#ifndef DISTRIBUTIONMULTIVARIATEARITHMETIC_H
#define DISTRIBUTIONMULTIVARIATEARITHMETIC_H
#include"data/distribution/DistributionMultiVariate.h"


namespace pop
{
/// @cond DEV

class DistributionMultiVariateArithmetic : public DistributionMultiVariate
{
    /*!
        \class pop::DistributionMultiVariateArithmetic
        \ingroup DistributionMultiVariate
        \brief Arithmetic DistributionMultiVariate interface to allow arithmtic operations for the DistributionMultiVariate class
        \author Tariel Vincent

      \sa DistributionMultiVariate::operator +() DistributionMultiVariate::operator -() DistributionMultiVariate::operator *() DistributionMultiVariate::operator /()
    */
private:
    DistributionMultiVariate  _fleft;
    DistributionMultiVariate  _fright;
public:
    void setDistributionMultiVariateLeft(const DistributionMultiVariate & f_left);
    void setDistributionMultiVariateRight(const DistributionMultiVariate &f_right);
    VecF32 randomVariable()const ;
    void setStep(F32 step)const;


    DistributionMultiVariate & getDistributionMultiVariateLeft();
    DistributionMultiVariate & getDistributionMultiVariateRight();
    const DistributionMultiVariate & getDistributionMultiVariateLeft()const;
    const DistributionMultiVariate & getDistributionMultiVariateRight()const;
    DistributionMultiVariateArithmetic();

};


 class DistributionMultiVariateArithmeticAddition : public DistributionMultiVariateArithmetic
 {

 public:
    DistributionMultiVariateArithmeticAddition();
    DistributionMultiVariateArithmeticAddition(const DistributionMultiVariateArithmeticAddition & dist);
    virtual DistributionMultiVariateArithmeticAddition * clone()const ;
    virtual F32 operator()(const VecF32&  value)const;
 };
 class DistributionMultiVariateArithmeticSubtraction : public DistributionMultiVariateArithmetic
 {

 public:
    DistributionMultiVariateArithmeticSubtraction();
    DistributionMultiVariateArithmeticSubtraction(const DistributionMultiVariateArithmeticSubtraction & dist);
    virtual DistributionMultiVariateArithmeticSubtraction * clone()const ;
    virtual F32 operator()(const VecF32&  value)const;

 };
 class DistributionMultiVariateArithmeticMultiplication : public DistributionMultiVariateArithmetic
 {

 public:
    DistributionMultiVariateArithmeticMultiplication();
    DistributionMultiVariateArithmeticMultiplication(const DistributionMultiVariateArithmeticMultiplication & dist);
    virtual DistributionMultiVariateArithmeticMultiplication * clone()const ;
    virtual F32 operator()(const VecF32&  value)const;

 };
 class DistributionMultiVariateArithmeticDivision : public DistributionMultiVariateArithmetic
 {

 public:
    DistributionMultiVariateArithmeticDivision();
    DistributionMultiVariateArithmeticDivision(const DistributionMultiVariateArithmeticDivision & dist);
    virtual DistributionMultiVariateArithmeticDivision * clone()const ;
    virtual F32 operator()(const VecF32&  value)const;

 };
 class DistributionMultiVariateArithmeticComposition : public DistributionMultiVariateArithmetic
 {

 public:
    DistributionMultiVariateArithmeticComposition();
    DistributionMultiVariateArithmeticComposition(const DistributionMultiVariateArithmeticComposition & dist);
    virtual DistributionMultiVariateArithmeticComposition * clone()const ;
    virtual F32 operator()(const VecF32&  value)const;

 };
 class DistributionMultiVariateArithmeticMax : public DistributionMultiVariateArithmetic
 {

 public:
    DistributionMultiVariateArithmeticMax();
    DistributionMultiVariateArithmeticMax(const DistributionMultiVariateArithmeticMax & dist);
    virtual DistributionMultiVariateArithmeticMax * clone()const ;
    virtual F32 operator()(const VecF32&  value)const;

 };
 class DistributionMultiVariateArithmeticMin : public DistributionMultiVariateArithmetic
 {

 public:
    DistributionMultiVariateArithmeticMin();
    DistributionMultiVariateArithmeticMin(const DistributionMultiVariateArithmeticMin & dist);
    virtual DistributionMultiVariateArithmeticMin * clone()const ;
    virtual F32 operator()(const VecF32&  value)const;

 };
 class DistributionMultiVariateSeparationProduct:public DistributionMultiVariateArithmetic
 {
     /*!
         \class pop::DistributionMerge
         \brief h(x,y)=f(x)*g(y)
         \author Tariel Vincent

     *
     *
     */
 public:
    DistributionMultiVariateSeparationProduct();
    DistributionMultiVariateSeparationProduct(const DistributionMultiVariateSeparationProduct & dist);
    virtual DistributionMultiVariateSeparationProduct * clone()const ;
    virtual F32 operator()(const VecF32&  value)const;
    VecF32 randomVariable()const ;
    virtual int getNbrVariable()const;
 };


 class DistributionMultiVariateCoupled:public DistributionMultiVariate
 {
     /*!
         \class pop::DistributionMerge
         \brief generate two coupled random variable \f$H(X={x,y})=F(X={x}) for x=y, 0 otherwise \f$
         \author Tariel Vincent

     *
     *
     */

private:
    int _nbr_variable_coupled;
    Distribution _single;

public:
    DistributionMultiVariateCoupled();
    DistributionMultiVariateCoupled(const DistributionMultiVariateCoupled &dist);

    void setNbrVariableCoupled(int nbr_variable_coupled);
    int getNbrVariableCoupled() const;


    void setSingleDistribution(const Distribution &distsingle);
    Distribution getSingleDistribution() const;

    virtual DistributionMultiVariateCoupled * clone()const ;
    virtual F32 operator()(const VecF32&  value)const;
    VecF32 randomVariable()const ;
    virtual int getNbrVariable()const;
 };
/// @endcond
}
#endif // DISTRIBUTIONMULTIVARIATEARITHMETIC_H
