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

#ifndef CFUNCTIONGENERATEALEA_H_
#define CFUNCTIONGENERATEALEA_H_
#include <cstring>
#include <queue>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include<ctime>
#include"data/typeF/TypeF.h"
#include"PopulationConfig.h"
#include "data/distribution/Distribution.h"
namespace pop
{
/// @cond DEV
class POP_EXPORTS DistributionDiscrete
{
private:
    mutable F32 _step;
    /*!
        \class pop::DistributionDiscrete
        \ingroup Distribution
        \brief allowing operations on discrete distributions
        \author Tariel Vincent
      *
      *
      * In order to calculate for instance the inner product between a continuous distribution and the discrete distribution, you
      * have to transform numerically your discrete distribution to a pencil continuous distribution. Such that each discrete value is transformed as follows
      * \image html test.png
      *
      *
    */
public:
    DistributionDiscrete(F32 step=0.1);
    void setStep(F32 step)const;
    F32 getStep()const;
    bool isInStepIntervale(F32 value,F32 hitvalue) const;
};








class POP_EXPORTS DistributionSign:public Distribution, public DistributionDiscrete
{
    /*!
        \class pop::DistributionSign
        \ingroup Distribution
        \brief probability distribution returning randomly 1 or -1
        \author Tariel Vincent

      * This discrete distribution is only used to generate random variable following the probability distribution:
      *  P(X=-1)=O.5  and P(X=1)=O.5
      *
      * \sa DistributionUniformInt
      *

      *
    */
public:
    virtual DistributionSign * clone()const ;
    F32 operator()(F32 value)const ;
    F32 randomVariable()const ;

};

class POP_EXPORTS DistributionUniformReal:public Distribution
{
    /*!
     * \class pop::DistributionUniformReal
     * \ingroup Distribution
     * \brief probability distribution returning a real number in a given range
     * \author Tariel Vincent
     *
     * This distribution generate continuous random variable such that the probability distribution is uniform on the intervalle (xmin,xmax)
     *  P(X=x)=1/(xmax-xmin) for xmin<=x<=xmax, 0 otherwise
     * \code
        Distribution d1(-1,1,"UNIFORMREAL");
        F32 sum1 = 0;
        int time = 20000;
        Mat2F32 mcollect1(time,2);
        for(int i=0;i<time;i++){
            mcollect1(i,0)=i;
            mcollect1(i,1)=sum1;
            sum1+=d1.randomVariable();
        }
        DistributionRegularStep randomwalk1(mcollect1);
        randomwalk1.display();
     * \endcode
     * \sa DistributionUniformInt
    */
private:

    F32 _xmin;
    F32 _xmax;

public:

    /*!
    \fn DistributionUniformReal(F32 xmin, F32 xmax);
    *
    *   constructor the continuous uniform distribution in the intervale (xmin,xmax)
    */
    DistributionUniformReal(F32 xmin, F32 xmax);
    virtual DistributionUniformReal * clone()const ;
    F32 operator()(F32 value)const ;
    virtual F32 randomVariable()const ;
};
class POP_EXPORTS DistributionUniformInt:public Distribution, public DistributionDiscrete
{
    /*!
        \class pop::DistributionUniformInt
        \ingroup Distribution
        \brief probability distribution returning an integer number in a intervalle
        \author Tariel Vincent

      * This distribution generates discrete random variable such that the probability distribution is uniform on the intervalle (xmin,xmax)
      *  P(X=x)=1/(xmax-xmin) for xmin<=x<=xmax, 0 otherwise with X a discrete value
      *
      * \sa DistributionUniformReal
      *

      *
    */
private:
    int _xmin;
    int _xmax;
public:
    /*!
    \fn DistributionUniformInt(int xmin, int xmax);
    *
    *   constructor the discrete uniform distribution in the intervale (xmin,xmax)
    */
    DistributionUniformInt(int xmin, int xmax);
    F32 randomVariable()const ;
    DistributionUniformInt * clone()const ;
    F32 operator()(F32 value)const ;


};
class POP_EXPORTS DistributionNormal:public Distribution
{
    /*!
        \class pop::DistributionNormal
        \ingroup Distribution
        \brief normal(gaussian) distribution
        \author Tariel Vincent

      * The normal (or Gaussian) distribution is a continuous probability distribution that has a bell-shaped probability density function, known as the Gaussian function or informally the bell curve
      *  P\f[ f(x;\mu,\sigma) = \frac{1}{\sigma\sqrt{2\pi}} e^{ -\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2 }\f]
      *  with\f$\mu\f$ the mean and  \f$\sigma\f$ the standard deviation
      *
      *

      *
    */
private:
    F32 _mean;
    F32 _standard_deviation;
    DistributionUniformReal _real;
public:

    /*!
    \fn DistributionNormal(F32 mean, F32 standard_deviation);
    *
    *   constructor the continuous  normal distribution with the mean and the standard_deviation
    *
    */
    DistributionNormal(F32 mean, F32 standard_deviation);
    F32 randomVariable()const ;
    DistributionNormal * clone()const ;
    F32 operator()(F32 value)const ;

};
class POP_EXPORTS DistributionBinomial:public Distribution, public DistributionDiscrete
{    /*!
        \class pop::DistributionBinomial
        \ingroup Distribution
        \brief binomial distribution
        \author Tariel Vincent

      *The binomial distribution is the discrete probability distribution of the number of successes
      *in a sequence of n independent yes/no experiments, each of which yields success with probability p.
      *  \f[ <math> P(K = k) = {n\choose k}p^k(1-p)^{n-k}</math>\f]
      *  with\f$n\f$ the number of times and  \f$p\f$ the probability
      *
      *
      *

      *
    */
private:
    F32 _probability;
    int _number_times;
    DistributionUniformReal  distreal01;
public:

    DistributionBinomial(F32 probability, int number_times);
    F32 randomVariable()const ;
    DistributionBinomial * clone()const ;
    F32 operator()(F32 value)const ;
};

class POP_EXPORTS DistributionExponential:public Distribution
{

    /*!
            \class pop::DistributionExponential
            \ingroup Distribution
            \brief exponential distribution
            \author Tariel Vincent

          *The probability density function (pdf) of an exponential distribution is

            \f[         f(x;\lambda) = \left\{ \begin{array}{cc} \lambda e^{-\lambda x}, & x \ge 0, \\ 0, & x < 0. \end{array} \right. \f]
          *
          *
          *
          *
        */

private:
    F32 _lambda;
    DistributionUniformReal  distreal01;
public:
    /*!
    \fn DistributionExponential(F32 lambda);
    *
    * constructor with the exponentiel parameter
    */
    DistributionExponential(F32 lambda);
    void reset(F32 lambda);
    F32 randomVariable()const ;
    DistributionExponential * clone()const ;
    F32 operator()(F32 value)const  ;
};


class POP_EXPORTS DistributionPoisson:public Distribution, public DistributionDiscrete
{
    /*!
    \class pop::DistributionPoisson
    \ingroup Distribution
    \brief poisson distribution
    \author Tariel Vincent

  *the discrete Poisson distribution  is a discrete probability distribution that expresses the probability of a given number of events occurring
  * in a fixed interval of time and/or space if these events occur with a known average rate and independently of the time since the last event
  *
  * \f[ f(k; \lambda)=\frac{\lambda^k e^{-\lambda}}{k!}\f]
  *
  *
*/
private:
    std::vector<F32> v_table;
    F32 _lambda;
    int _maxlambda;
    DistributionPoisson  *flambdalargemult;
    DistributionPoisson  *flambdalargerest;
    int mult;
    DistributionUniformReal  distreal01;
        void init();
public:
    ~DistributionPoisson();
    DistributionPoisson(F32 lambda);
    F32 randomVariable()const ;
    F32 randomVariable(F32 lambda)const ;
    DistributionPoisson * clone()const ;
    F32 operator()(F32 value)const ;
};




class POP_EXPORTS DistributionDirac:public Distribution, public DistributionDiscrete
{
    /*!
        \class pop::DistributionDirac
        \ingroup Distribution
        \brief dirac distribution
        \author Tariel Vincent

      * The delta dirac distribution is useless for the genereration of random variable
      * since it returns always the same value. However for some calculations, we need it see
      *    <a href=http://en.wikipedia.org/wiki/Dirac_delta_function>wiki</a>\n
      *  In our implementation, this function is equal to\n f(x')=1/step for x-step/2<=x'<=x+step/2 0 otherwise \n
      *  with x its parameter

      *
    */
private:
    F32 _x;
public:
    DistributionDirac(F32 x);
    F32 randomVariable()const ;
    DistributionDirac * clone()const ;
    F32 operator()(F32 value)const ;
};

class POP_EXPORTS DistributionTriangle:public Distribution
{
    /*!
     *   \class pop::DistributionTriangle
     *   \ingroup Distribution
     *   \brief the triangle function, hat function, or tent function
     *   \author Tariel Vincent
     *
    */
public:

    F32 _x_min;
    F32 _x_max;
    F32 _x_peak;
    DistributionUniformReal  _distreal01;
    DistributionTriangle(F32 xmin,F32 xmax,F32 peak);
    F32 randomVariable()const ;
    DistributionTriangle * clone()const ;
    F32 operator()(F32 value)const ;
};


/// @endcond
}
#endif /* CFUNCTIONGENERATEALEA_H_ */
