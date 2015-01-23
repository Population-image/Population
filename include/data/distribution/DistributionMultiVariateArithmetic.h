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
///// @cond DEV
namespace Private {
class DistributionMultiVariateConcatenation
{
protected:
    DistributionMultiVariate * _fleft;
    DistributionMultiVariate * _fright;
public:
    DistributionMultiVariateConcatenation(const DistributionMultiVariate &f_left,const DistributionMultiVariate& f_right);
    DistributionMultiVariateConcatenation& operator=(const DistributionMultiVariateConcatenation&a);
    virtual ~DistributionMultiVariateConcatenation();
};
}


class DistributionMultiVariateArithmeticAddition : public DistributionMultiVariate, public Private::DistributionMultiVariateConcatenation
{
public:
    DistributionMultiVariateArithmeticAddition(const DistributionMultiVariate &f_left,const DistributionMultiVariate& f_right);
    virtual DistributionMultiVariateArithmeticAddition * clone()const ;
    virtual F32 operator()(const VecF32& value)const ;
    virtual VecF32 randomVariable()const;
    virtual unsigned int getNbrVariable()const;
};
class DistributionMultiVariateArithmeticSubtraction : public DistributionMultiVariate, public Private::DistributionMultiVariateConcatenation
{
public:
    DistributionMultiVariateArithmeticSubtraction(const DistributionMultiVariate &f_left,const DistributionMultiVariate& f_right);
    virtual DistributionMultiVariateArithmeticSubtraction * clone()const ;
    virtual F32 operator()(const VecF32& value)const ;
    virtual VecF32 randomVariable()const;
    virtual unsigned int getNbrVariable()const;

};
class DistributionMultiVariateArithmeticMultiplication : public DistributionMultiVariate, public Private::DistributionMultiVariateConcatenation
{
public:
    DistributionMultiVariateArithmeticMultiplication(const DistributionMultiVariate &f_left,const DistributionMultiVariate& f_right);
    virtual DistributionMultiVariateArithmeticMultiplication * clone()const ;
    virtual F32 operator()(const VecF32& value)const ;
    virtual VecF32 randomVariable()const;
    virtual unsigned int getNbrVariable()const;

};
class DistributionMultiVariateArithmeticDivision : public DistributionMultiVariate, public Private::DistributionMultiVariateConcatenation
{
public:
    DistributionMultiVariateArithmeticDivision(const DistributionMultiVariate &f_left,const DistributionMultiVariate& f_right);
    virtual DistributionMultiVariateArithmeticDivision * clone()const ;
    virtual F32 operator()(const VecF32& value)const ;
    virtual VecF32 randomVariable()const;
    virtual unsigned int getNbrVariable()const;

};

class DistributionMultiVariateArithmeticMax : public DistributionMultiVariate, public Private::DistributionMultiVariateConcatenation
{
public:
    DistributionMultiVariateArithmeticMax(const DistributionMultiVariate &f_left,const DistributionMultiVariate& f_right);
    virtual DistributionMultiVariateArithmeticMax * clone()const ;
    virtual F32 operator()(const VecF32& value)const ;
    virtual VecF32 randomVariable()const;
    virtual unsigned int getNbrVariable()const;

};
class DistributionMultiVariateArithmeticMin : public DistributionMultiVariate, public Private::DistributionMultiVariateConcatenation
{

public:
    DistributionMultiVariateArithmeticMin(const DistributionMultiVariate &f_left,const DistributionMultiVariate& f_right);
    virtual DistributionMultiVariateArithmeticMin * clone()const ;
    virtual F32 operator()(const VecF32& value)const ;
    virtual VecF32 randomVariable()const;
    virtual unsigned int getNbrVariable()const;

};
DistributionMultiVariateArithmeticAddition operator +(const DistributionMultiVariate &d1,const DistributionMultiVariate &d2);
DistributionMultiVariateArithmeticSubtraction operator -(const DistributionMultiVariate &d1,const DistributionMultiVariate &d2);
DistributionMultiVariateArithmeticMultiplication operator *(const DistributionMultiVariate &d1,const DistributionMultiVariate &d2);
DistributionMultiVariateArithmeticDivision operator /(const DistributionMultiVariate &d1,const DistributionMultiVariate &d2);
DistributionMultiVariateArithmeticMin minimum(const DistributionMultiVariate &d1,const DistributionMultiVariate &d2);
DistributionMultiVariateArithmeticMax maximum(const DistributionMultiVariate &d1,const DistributionMultiVariate &d2);
//class DistributionMultiVariateSeparationProduct:public DistributionMultiVariateArithmetic
//{
//    /*!
//         \class pop::DistributionMerge
//         \brief h(x,y)=f(x)*g(y)
//         \author Tariel Vincent
//     *
//     *
//     */
//public:
//    DistributionMultiVariateSeparationProduct();
//    DistributionMultiVariateSeparationProduct(const DistributionMultiVariateSeparationProduct & dist);
//    virtual DistributionMultiVariateSeparationProduct * clone()const ;
//    virtual F32 operator()(const VecF32&  value)const;
//    VecF32 randomVariable()const ;
//    virtual unsigned int getNbrVariable()const;
//};
// class DistributionMultiVariateCoupled:public DistributionMultiVariate
// {
//     /*!
//         \class pop::DistributionMerge
//         \brief generate two coupled random variable \f$H(X={x,y})=F(X={x}) for x=y, 0 otherwise \f$
//         \author Tariel Vincent
//     *
//     *
//     */
//private:
//    int _nbr_variable_coupled;
//    Distribution* _single;

//public:
//    DistributionMultiVariateCoupled();
//    DistributionMultiVariateCoupled(const DistributionMultiVariateCoupled &dist);

//    void setNbrVariableCoupled(int nbr_variable_coupled);
//    int getNbrVariableCoupled() const;


//    void setSingleDistribution(Distribution *distsingle);
//    Distribution * getSingleDistribution() const;

//    virtual DistributionMultiVariateCoupled * clone()const ;
//    virtual F32 operator()(const VecF32&  value)const;
//    VecF32 randomVariable()const ;
//    virtual int getNbrVariable()const;
// };
/// @endcond
}
#endif // DISTRIBUTIONMULTIVARIATEARITHMETIC_H
