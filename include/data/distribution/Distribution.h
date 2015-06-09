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
#ifndef CLIBRARYDISTRIBUTION_HPP
#define CLIBRARYDISTRIBUTION_HPP
#include <string>
#include <cstring>
#include <queue>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <stack>
#include <limits>

#include"3rdparty/MTRand.h"
#include"data/typeF/TypeTraitsF.h"
#include"data/vec/Vec.h"

///*! \ingroup Data
//* \defgroup DistributionFunction  Function (mathematics)
//* \brief Function (mathematics) defined with a symbolic link
//*/

namespace pop
{


///*! \ingroup DistributionFunction
//* \defgroup Distribution  Distribution
//* \brief mapping from the real number to the real number
//*/
class POP_EXPORTS Distribution
{
    /*!
        \class pop::Distribution
        \brief  abstract class for Distribution
        \author Tariel Vincent

* \section DefinitonDistribution Basic definition
*
* The class Distribution, not named function to avoid any confusions with the function concept,  is a mapping from the real number
* to the real number where the input element, x, completely determines the output element, y,  by this relation: y = f(x).
* The evaluation of the output element from an input element is a symbolic (implicit) relation for instance,
* for a function equal to \f$x \mapsto x^2\f$, the output element is evaluated by the multiplication of the input element by itself.
*
* \section ImplementationDistribution Code
*
* I implement a class pop::Distribution and many derived. Because I hide the polymorphism mechanis
* with an implementation pattern, you have to create objects (and not pointer) of the class pop::Distribution to operate
*  - basic arithmetic operations
*  - analysis/process with algorithm class pop::Statistics
*
* For instance, this code
      \code
        DistributionExpression f("x");//f(x)=x
        for(int i =0;i<4;i++)
            std::cout<<f(i)<<" ";
        std::cout<<std::endl;
        DistributionArithmeticMultiplication fmult = f*f;//now f(x)=x*x
        for(int i =0;i<4;i++)
            std::cout<<fmult(i)<<" ";
        std::cout<<std::endl;
        DistributionRegularStep fderivate = pop::Statistics::derivate(fmult,0,10);//now f(x)=2*x
        for(int i =0;i<4;i++)
            std::cout<<fderivate(i)<<" ";
        std::cout<<std::endl;
      \endcode
      produce this output: \n
    0 1 2 3 \n
    0 1 4 9 \n
    0.01 2.01 4.01 6.01 \n

    */
private:
    static unsigned long _init[4];
    static unsigned long _length;
    static MTRand_int32 _irand;
public:
    static MTRand_int32 &irand();

    /*!
    \fn virtual ~Distribution();
    *
    *  virtual destructor
    */
    virtual ~Distribution(){}

    /*!
    * \param value input value
    * \return y
    *
    *  Unary function to call y=f(value)
      \code
    DistributionExpression f("sqrt(x)");
    for(int i =0;i<4;i++)
        std::cout<<f(i)<<" ";
        \endcode
    */
    virtual F32 operator()(F32 value)const=0;
    /*!
    * \return X random variable
    *
    *  Generate random variable, X, following the probability distribution f.\n
    *  However,  some distributions do not have a procedure to generate some random variables.
    *  In this case, you have to use this algorithm  pop::Statistics::toProbabilityDistribution to generate the probability distribution.\n
    * For instance, in the following code, we simulate random variables followin an exponential probability distribution in two ways: from analytical or from expression.
    * Because DistributionExpression does not have a generator of random number,  we call the procedure pop::Statistics::toProbabilityDistribution .
    \code
        F32 lambda=1.f/10;
        std::string exp = "exp(-"+BasicUtility::Any2String(lambda)+"*x)";
        DistributionExpression expression(exp.c_str());
        DistributionRegularStep fexponentialfromexpression = pop::Statistics::toProbabilityDistribution(expression,0,100);//DistributionExpression does not have a generator of random number.

        DistributionPoisson  fexponentialfromanalytical(1./lambda);//DistributionExponential, has a genrator so you don't need to use  pop::Statistics::toProbabilityDistribution

        int nbr_sampling = 500000;
        F32 sum1=0,sum2=0;
        for(int i =0;i<nbr_sampling;i++){
            sum1+=fexponentialfromexpression.randomVariable();
            sum2+=fexponentialfromanalytical.randomVariable();
        }
        std::cout<<"Mean1: "<<    sum1/nbr_sampling<<std::endl;
        std::cout<<"Mean2: "<<    sum2/nbr_sampling<<std::endl;
        return 1;
    \endcode
    *
    */
    virtual F32 randomVariable()const=0;
    /*!
    * \return  VecNer distribution
    *
    *  clone pattern
    */
    virtual Distribution * clone()const=0 ;
    void display(F32 xmin=0,F32 xmax=255)const;
};


class POP_EXPORTS DistributionDisplay
{
public:
    /*!
    * \param d input Distribution
    * \param xmin xmin value
    * \param xmax xmax value
    * \param ymin ymin value
    * \param ymax ymax value
    * \param sizex resolution in x
    * \param sizey resolution in y
    * \return  windows as matrix
    *
    *  Simple multi graph display. With the arrow keys, you can translate the graph and with the + and - keys, you can (un)zoom
    *  For more extented graph display, you convert it in Matrix with this procedure pop::Statistics::toMatrix, then you save it. This matrix can be open by Matlab or gnuplot (plot 'dist.m' w l)
    \code
        DistributionExpression d1("1");
        DistributionExpression d2("x^2+1");
        DistributionArithmeticDivision div  = d1/d2;//d1 is equal to 1/(x^2-1)
        DistributionDisplay::display(div,-2,2);

        DistributionExpression f("x");//f(x)=x
        DistributionExpression f22("x^2");//f(x)=x
        DistributionDisplay::display(f,f22,0,1);

    \endcode
    */
    static void display(const Distribution & d,F32 xmin,F32 xmax,F32 ymin=NumericLimits<F32>::minimumRange(),F32 ymax=NumericLimits<F32>::maximumRange(),int sizewidth=600,int sizeheight=600);
    static void display(const Distribution & d1,const Distribution & d2,F32 xmin,F32 xmax,F32 ymin=NumericLimits<F32>::minimumRange(),F32 ymax=NumericLimits<F32>::maximumRange(),int sizewidth=600,int sizeheight=600);
    static void display(const Distribution & d1,const Distribution & d2,const Distribution & d3,F32 xmin,F32 xmax,F32 ymin=NumericLimits<F32>::minimumRange(),F32 ymax=NumericLimits<F32>::maximumRange(),int sizewidth=600,int sizeheight=600);
    static void display(pop::Vec<const Distribution*> d,F32 xmin,F32 xmax,F32 ymin=NumericLimits<F32>::minimumRange(),F32 ymax=NumericLimits<F32>::maximumRange(),int sizewidth=600,int sizeheight=600);
};



}

#endif // CLIBRARYDISTRIBUTION_HPP
