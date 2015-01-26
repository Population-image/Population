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


///*! \ingroup Data
//* \defgroup DistributionFunction  Function (mathematics)
//* \brief Function (mathematics) defined with a symbolic link
//*/

namespace pop
{


///*! \ingroup DistributionFunction
//* \defgroup Distribution  Distribution
//* \brief mapping from the real number to the real number
//*
//* \section DefinitonDistribution Basic definition
//*
//* The class Distribution, not named function to avoid any confusions with the function concept,  is a mapping from the real number
//* to the real number where the input element, x, completely determines the output element, y,  by this relation: y = f(x).
//* The evaluation of the output element from an input element is a symbolic (implicit) relation for instance,
//* for a function equal to \f$x \mapsto x^2\f$, the output element is evaluated by the multiplication of the input element by itself.
//*
//* \section ImplementationDistribution Code
//*
//* I implement a class pop::Distribution and many derived. Because I hide the polymorphism mechanis
//* with an implementation pattern, you have to create objects (and not pointer) of the class pop::Distribution to operate
//*  - basic arithmetic operations
//*  - analysis/process with algorithm class pop::Statistics
//*
//* For instance, this code
//      \code
//    Distribution f("x", "EXPRESSION");//f(x)=x
//    for(int i =0;i<4;i++)
//        std::cout<<f(i)<<" ";
//    std::cout<<std::endl;
//    f = f*f;//now f(x)=x*x
//    for(int i =0;i<4;i++)
//        std::cout<<f(i)<<" ";
//    std::cout<<std::endl;
//    f = pop::Statistics::derivate(f,0,10);//now f(x)=2*x
//    for(int i =0;i<4;i++)
//        std::cout<<f(i)<<" ";
//    std::cout<<std::endl;
//      \endcode
//      produce this output: \n
//    0 1 2 3 \n
//    0 1 4 9 \n
//    0.01 2.01 4.01 6.01 \n
//*/
class POP_EXPORTS Distribution
{
    /*!
        \class pop::Distribution
        \ingroup Distribution
        \brief  Function with a symbolic link
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
    Distribution f("x", "EXPRESSION");//f(x)=x
    for(int i =0;i<4;i++)
        std::cout<<f(i)<<" ";
    std::cout<<std::endl;
    f = f*f;//now f(x)=x*x
    for(int i =0;i<4;i++)
        std::cout<<f(i)<<" ";
    std::cout<<std::endl;
    f = pop::Statistics::derivate(f,0,10);//now f(x)=2*x
    for(int i =0;i<4;i++)
        std::cout<<f(i)<<" ";
    std::cout<<std::endl;
      \endcode
      produce this output: \n
    0 1 2 3 \n
    0 1 4 9 \n
    0.01 2.01 4.01 6.01 \n

    */
private:
   static unsigned long init[4];
    static unsigned long length;
public:
//    struct Test{
//        unsigned long operator ()(){
//            return 0;
//        }
//         unsigned long operator()(unsigned int N ) {
//            return 0;
//         }
//    };
//    static Test irand;
    static MTRand_int32 irand;
    //-------------------------------------
    //
    //! \name Constructor
    //@{
    //-------------------------------------
    /*!
    *
    * default constructor
    */
//    Distribution();


    /*!
    * \param d other distribution
    *
    * copy constructor
    * \code
    Distribution d(DistributionUniformReal(0,10));
    for(int i =0;i<30;i++)
        std::cout<<d.randomVariable()<<std::endl;
    return 1;
    * \endcode
    *
    */
//    Distribution(const Distribution & d);
    /*!
    * \param param discrete data in matrix (x-values=first column, y-valuess=second column)
    * \param type type of distribution with a Matrix parameter as argument
    *
    *  Contruct a regular step function as piecewise constant function having many regular discrete pieces.
    *  To define it, we use an input Mat2F32 such that the first column contained the X-values with a regular spacing between successive
    *  value and the second column Y-values. For instance, this code
    *
    * \code
    Mat2F32 m(100,2);
    F32 step = 0.01;
    for(int i =0;i<100;i++)
    {
        m(i,0)= step*i;
        m(i,1)= (step*i)*(step*i);
    }
    Distribution d(m);
    d.display();
    * \endcode
    * produces an approximation of the x^2 between 0 and 1.
    *
    * \sa DistributionRegularStep
    */
//    Distribution(const Mat2F32& param,std::string type="STEP");


    /*!
    * \param param  regular expression such that the variable is x
    * \param type  type of distribution with a string parameter as argument
    *
    * Contruct a function with a regular expression such that the variable is x. For instance, this code
    *
    *  \code
    *    Distribution f("x^2", "EXPRESSION");
    *    for(int i =0;i<4;i++)
    *        std::cout<<f(i)<<std::endl;
    * \endcode
        produces this output: 0 1 4. This class is based on the library Function Parser for C++ v4.5  http://warp.povusers.org/FunctionParser/. The syntax of the regular expression
    * is documented here http://warp.povusers.org/FunctionParser/fparser.html#literals\n
    * For developper, because the licence is under LGPL licence,  you cannot modify the code of the file fparser.hh, fpconfig.hh and fptypes.hh.
    * \sa DistributionExpression
    */
//    Distribution(const char* param,std::string type="EXPRESSION");



    /*!
    * \param param scalar parameter
    * \param type type of distribution with a scalar parameter as argument
    *
    * Contruct:
    * - type="POISSON" the poisson distribution with param=lambda is a discrete probability distribution that expresses the probability of a given number of events occurring
    * in a fixed interval of time and/or space if these events occur with a known average rate and independently of the time since the last event
    *
    * \f[ f(k; \lambda)=\frac{\lambda^k e^{-\lambda}}{k!}\f]
    *
    * - type="EXPONENTIAL" the exponential distribution with param=lambda is a continuous probability distributio that expresses the time between events in a Poisson process
    * such that its  probability  is
    *
    *        \f[         f(x;\lambda) = \left\{ \begin{array}{cc} \lambda e^{-\lambda x}, & x \ge 0, \\ 0, & x < 0. \end{array} \right. \f]
    * - type="DIRAC" the dirac distribution (usefull to generate a random number always equal to the same value in RandomGeometry algorithms)
    *
    *
    *
    *  \code
    *    Distribution f(10, "POISSON");
    *    for(int i =0;i<4;i++)
    *        std::cout<<f.randomVariable()<<std::endl;
    * \endcode
    * \sa DistributionPoisson DistributionExponential DistributionDirac
    * \sa http://en.wikipedia.org/wiki/Poisson_distribution http://en.wikipedia.org/wiki/Exponential_distribution http://en.wikipedia.org/wiki/Dirac_delta_function
    */
//    Distribution(F32 param,std::string type="POISSON");


    /*!
    * \param param1 first scalar parameter
    * \param param2 second scalar parameter
    * \param type type of distribution with two scalar parameters as arguments
    *
    * Contruct:
    * - type="UNIFORMREAL" the uniform distribution with a constant continuous probability in the range (param1,param2)
    * - type="UNIFORMINT" the uniform distribution with a constant discrete probability in the range (param1,param2)
    * - type="BINOMIAL"" the binomial distribution with param1=p and param2=n is the discrete probability distribution of the number of successes
    * in a sequence of n independent yes/no experiments, each of which yields success with probability p
    *  \f[  P(K = k) = {n\choose k}p^k(1-p)^{n-k}\f]
    * with\f$n\f$ the number of times and  \f$p\f$ the probability
    * - type="NORMAL" the normal (or Gaussian) distribution with param1=mean and param2=standard_deviation is a continuous probability distribution that has a bell-shaped probability density function, known as the Gaussian function or informally the bell curve
      *  P\f[ f(x;\mu,\sigma) = \frac{1}{\sigma\sqrt{2\pi}} e^{ -\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2 }\f]
      *  with\f$\mu\f$ the mean and  \f$\sigma\f$ the standard deviation
    *
    *
    *
    *  \code
    *    Distribution f(0,1, "UNIFORMREAL");
    *    for(int i =0;i<4;i++)
    *        std::cout<<f.randomVariable()<<std::endl;
    * \endcode
    * \sa  DistributionUniformReal, DistributionUniformInt, DistributionNormal, DistributionBinomial
    */
//    Distribution(F32 param1,F32 param2,std::string type="NORMAL");



    /*!
    \fn virtual ~Distribution();
    *
    *  virtual destructor
    */
    virtual ~Distribution(){}


    //@}
    //-------------------------------------
    //
    //! \name f(x)=y and generator of random variable
    //@{
    //-------------------------------------
    /*!
    * \param value input value
    * \return y
    *
    *  Unary function to call y=f(value)
      \code
    Distribution f("std::sqrt(x)", "EXPRESSION");
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
    F32 lambda=1./10;

     std::string exp = "std::exp(-"+BasicUtility::Any2String(lambda)+"*x)";
    Distribution fexponentialfromexpression(exp.c_str(), "EXPRESSION");
    fexponentialfromexpression = pop::Statistics::toProbabilityDistribution(fexponentialfromexpression,0,100);//DistributionExpression does not have a generator of random number.
    //So you must convert it in probability distribution before.

    Distribution  fexponentialfromanalytical(lambda,"EXPONENTIAL");//DistributionExponential, has a genrator so you don't need to use  pop::Statistics::toProbabilityDistribution

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


    //@}
    //-------------------------------------
    //
    //! \name In/out
    //@{
    //-------------------------------------

    /*!
    * \param xmin xmin value
    * \param xmax xmax value
    * \param ymin ymin value
    * \param ymax ymax value
    * \param sizewidth width resolution
    * \param sizeheight height resolution
    * \return  windows as matrix
    *
    *  Simple graph display. With the arrow keys, you can translate the graph and with the + and - keys, you can (un)zoom
   *  For more extented graph display, you convert it in Matrix with this procedure pop::Statistics::toMatrix, then you save it. This matrix can be open by Matlab or gnuplot (plot 'dist.m' w l)
    \code
    Distribution f("x*std::log(x)/std::log(2)", "EXPRESSION");
    // Display in the x-range [0,10] with this very simple graph editor
    f.display(0,10);//You have to close the windows to execute the rest of the code
    //Another solution, save it in a Matrix format and open it with a nice graph editor
    Mat2F32 m = pop::Statistics::toMatrix(f,0,10);
    m.save("dist.m");
    return 1;
    \endcode
    * \sa pop::Statistics::toMatrix
    */
//    Mat2RGBUI8 display(F32 xmin=-NumericLimits<F32>::maximumRange(),F32 xmax=NumericLimits<F32>::maximumRange(),F32 ymin=-NumericLimits<F32>::maximumRange(),F32 ymax=NumericLimits<F32>::maximumRange(),int sizewidth=800,int sizeheight=600);

    /*!
    * \param vd
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
    Distribution danalytical( 10,15,"NORMAL");
    VecF32 v;
    for(int i=0;i<1000000;i++){
        v.push_back(danalytical.randomVariable());
    }
    Distribution dstat = pop::Statistics::computedStaticticsFromRealRealizations(v);

    std::vector<Distribution> vd;
    vd.push_back(danalytical);
    vd.push_back(dstat);
    Distribution::multiDisplay(vd,-30,50);
    \endcode
    * \sa pop::Statistics::toMatrix
    */
//    static Mat2RGBUI8 multiDisplay( std::vector<Distribution> & vd,F32 xmin=-NumericLimits<F32>::maximumRange(),F32 xmax=NumericLimits<F32>::maximumRange(),F32 ymin=-NumericLimits<F32>::maximumRange(),F32 ymax=NumericLimits<F32>::maximumRange(),int sizex=800,int sizey=600);
//    static Mat2RGBUI8 multiDisplay( Distribution & d1,Distribution & d2,F32 xmin=-NumericLimits<F32>::maximumRange(),F32 xmax=NumericLimits<F32>::maximumRange(),F32 ymin=-NumericLimits<F32>::maximumRange(),F32 ymax=NumericLimits<F32>::maximumRange(),int sizex=800,int sizey=600);
//    static Mat2RGBUI8 multiDisplay( Distribution & d1,Distribution & d2,Distribution & d3,F32 xmin=-NumericLimits<F32>::maximumRange(),F32 xmax=NumericLimits<F32>::maximumRange(),F32 ymin=-NumericLimits<F32>::maximumRange(),F32 ymax=NumericLimits<F32>::maximumRange(),int sizex=800,int sizey=600);


    //@}
    //-------------------------------------
    //
    //! \name Arithmetic
    //@{
    //-------------------------------------
    /*!
    \param d other distribution
    *
    * Basic assignement of the other distribution to this distribution
    */
//    Distribution & operator =(const Distribution& d);



    /*!
    * \param g other  Distribution
    * \return   distribution
    *
    *  h(x) = (*this)(g(x))
    * \code
    Distribution d1("1/x");
    Distribution d2("x^2+1");

    d1 = d1.rho(d2);//d1 is equal to 1/(x^2+1)
    d1.display(-10,10);
    * \endcode
    */
//    Distribution rho(const Distribution &g)const;

    /*!
    * \param d input Distribution
    * \return   distribution
    *
    *  Addition operator :  h(x) = (*this)(x)+d(x)
    * \code
    Distribution d1("x^3");
    Distribution d2("-x");
    d1 = d1+d2;//d1 is equal to x^3-x
    d1.display(-2,2);
    * \endcode
    */
//    Distribution  operator +(const Distribution& d)const;

    /*!
    * \param d other Distribution
    * \return   distribution
    *
    *  Subtraction operator : h(x) = (*this)(x)-d(x)
    * \code
    Distribution d1("x^3");
    Distribution d2("x");
    d1 = d1-d2;//d1 is equal to x^3-x
    d1.display(-2,2);
    * \endcode
    */
//    Distribution  operator -(const Distribution& d)const;
    /*!
    * \param d other Distribution
    * \return   distribution
    *
    *  Multiplication operator : h(x) = (*this)(x)*d(x)
    * \code
    Distribution d1("x");
    Distribution d2("x^2-1");
    d1 = d1*d2;//d1 is equal to x*(x^2-1)
    d1.display(-2,2);
    * \endcode
    */
//    Distribution  operator *(const Distribution& d)const;

    /*!
    \fn Distribution  operator /(const Distribution& d)const;
    * \param d other Distribution
    * \return   distribution
    *
    *  Division operator : h(x) = (*this)(x)/d(x)
    * \code
    Distribution d1("1");
    Distribution d2("x^2+1");
    d1 = d1/d2;//d1 is equal to 1/(x^2-1)
    d1.display(-2,2);
    * \endcode
    */
//    Distribution  operator /(const Distribution& d)const;

    /*!
    * \return   distribution
    *
    *  Unary minus operator : h(x) = -(*this)(x)
    */
//    Distribution  operator -()const;
    //@}

    //-------------------------------------
    //
    //! \name others
    //@{
    //-------------------------------------

    /*!
    * \param step F32 step
    *
    *  You can ignore the implementation of this member in the derived class.
    * This member inform the object that the subinterval of the partition for calculation as a given step
    *
    * \sa IteratorDistributionSamplingRiemann DistributionDiscrete
    */
//    virtual void setStep(F32 step)const;


    /*!
    * \return min_value
    *
    *  f(x)=0 for x<min value by default min_value=-infinity
    *
    */
//    virtual F32 getXmin()const;

    /*!
    \fn virtual F32 getXmax()const;
    * \return max_value
    *
    *  f(x)=0 for x>max_value by default min_value=+infinity
    *
    */
//    virtual F32 getXmax()const;



    /*!
    * \return step of the distribution
    *
    *  partition interval used for approximation calculation (by default 0.01)
    *
    */
//    virtual F32 getStep()const;




    /*!
    * \return MTRand_int32
    *
    *  Get the generator of random number (mersene twister)
    */
//    MTRand_int32 & MTRand();


    /*!
    * \return  VecNer distribution
    *
    *  clone pattern
    */
    virtual Distribution * clone()const=0 ;
    //@}


//    virtual F32 randomVariable(F32 param)const ;
//    Distribution * ___getPointerImplementation();
//    const Distribution * ___getPointerImplementation()const ;
//    void ___setPointererImplementation(Distribution * d);
};


class DistributionDisplay
{
public:
    static void display(const Distribution & d,F32 xmin,F32 xmax,F32 ymin=-NumericLimits<F32>::maximumRange(),F32 ymax=NumericLimits<F32>::maximumRange(),int sizewidth=800,int sizeheight=600);
};





//POP_EXPORTS pop::Distribution maximum(const pop::Distribution & d1, const pop::Distribution & d2);


//POP_EXPORTS pop::Distribution minimum(const pop::Distribution & d1, const pop::Distribution & d2);

}

#endif // CLIBRARYDISTRIBUTION_HPP
