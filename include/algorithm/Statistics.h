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
#ifndef STATISTICS_H
#define STATISTICS_H
#include<numeric>
#include"data/distribution/Distribution.h"
#include"data/distribution/DistributionFromDataStructure.h"
#include"data/distribution/DistributionMultiVariate.h"
#include"data/distribution/DistributionMultiVariateFromDataStructure.h"
#include"data/vec/Vec.h"
#include"data/mat/MatN.h"
#include"data/mat/Mat2x.h"

namespace pop
{

/*!
\defgroup Statistics Statistics
\ingroup Algorithm
\brief Distribution In -> Distribution Out (derivate, integral, computedStaticticsFromRealRealizations,...)

Before to start with this class of algorithm, you should read the documentation of the class Distribution since this class
acts on this data-structure. A classical calculus is the derivate:
\code
Distribution f("x^2", "EXPRESSION");//f(x)=x
for(int i =0;i<4;i++)
    std::cout<<f(i)<<" ";
f = pop::Statistics::derivate(f,0,10);//now f(x)=2*x
for(int i =0;i<4;i++)
    std::cout<<f(i)<<" ";
\endcode
This code produces this output: \n
0 1 4 9 \n
0.01 2.01 4.01 6.01 \n
*/

struct POP_EXPORTS Statistics
{

    /*!
        \class pop::Statistics
        \ingroup Statistics
        \brief Mathematical analysis of function with symbolic link (mathematics) oriented stastistics
        \author Tariel Vincent
    */
    //-------------------------------------
    //
    //! \name Distribution
    //@{
    //-------------------------------------
    /*! \fn DistributionRegularStep integral(const Distribution &f, F32 xmin, F32 xmax,F32 step=0.01);
     * \param f input distribution
     * \param xmin lower bound
     * \param xmax upper bound
     * \param step step
     * \return the integral as a Distribution
     *
     *  calculate the integral defined in the range (xmin,xmax) \f$F(x)=\left\{ \begin{array}{ll}
     * \int_{x_{min}}^x f(x) dx & \mbox{for }x_{min}\leq x \leq x_{max}\\
     * 0 &\mbox{otherwise}  \end{array} \right.\f$ where the integral is computed using the  Riemann sum.
     * \code
     * Distribution f("abs(x)");
     * Distribution fint = Statistics::integral(f,-1,1);
     * std::cout<<fint(1)<<std::endl;//=1
     * Distribution::multiDisplay(f,fint,-1,1).save("../doc/image2/integral.jpg");
     * \endcode
     * \image html integral.jpg
    */
    static DistributionRegularStep  integral(const Distribution &f, F32 xmin, F32 xmax,F32 step=0.01);


    /*! \fn DistributionRegularStep derivate(const Distribution &f, F32 xmin, F32 xmax,F32 step);
     * \param f input distribution
     * \param xmin lower bound
     * \param xmax upper bound
     * \param step step
     * \return the derivate as a Distribution
     *
     *  calculate the derivate \f$f'(x)=\left\{ \begin{array}{ll}
     *  df(x)/dx & \mbox{for }x_{min}\leq x \leq x_{max}\\
     * 0 &\mbox{otherwise}  \end{array} \right.\f$ where the derivate is computed using this discrete approximation (f(x+step)-f(x))/step.
     * \code
     * Distribution f("atan(x)");
     * Distribution fd = Statistics::derivate(f,-PI,PI);
     * std::cout<<fd(1)<<std::endl;//=1
     * Distribution::multiDisplay(f,fd,-PI,PI).save("../doc/image2/derivate.jpg");
     * \endcode
     * \image html derivate.jpg
    */
    static DistributionRegularStep derivate(const Distribution &f, F32 xmin, F32 xmax,F32 step=0.01);



    /*! \brief calculate the inverse of the function
     * \param f input distribution
     * \param xmin lower bound
     * \param xmax upper bound
     * \param step step
     * \param error error threshold
     * \return the inverse as a Distribution
     *
     *  calculate the inverse of the function : \f$f(x) = y\,\,\text{if and only if}\,\,f^{-1}(y) = x\f$, the lower an upper bound of the output function become f(xmin) f(xmax) respectively.
     *  The output distribution is a regular step function with a step equal to minimum  f(i+step)-f(i).

     *  \code
     *  Distribution f("x^2");
     *  Distribution finverse = Statistics::inverse(f,0,1);
     *  Distribution::multiDisplay(f,finverse,0,1).save("../doc/image2/inverse.jpg");
     *  \endcode
     * \image html inverse.jpg
     */
    static DistributionRegularStep inverse(const Distribution &f, F32 xmin, F32 xmax,F32 step=0.01,F32 error=0.01);



    /*! \fn F32 argMin(const Distribution &f,F32 xmin, F32 xmax,F32 step);
     * \param f input distribution
     * \param xmin lower bound
     * \param xmax upper bound
     * \param step step
     * \return arg min
     *
     *  return the argument of the minimum \f$ \arg\min_{x_{min}\leq x \leq x_{max}} f(x)\f$ where the computation is done on the associated regular step function
     *  \code
     *  Distribution f("(x-0.5)^2");
     *  F32 value = Statistics::argMin(f,-10,10);
     *  std::cout<<"ArgMin "<<value<<std::endl;//=0.5
     *  f.display(-10,10).save("../doc/image2/ArgMin.jpg");//Visual checking
     *  \endcode
     * \image html ArgMin.jpg
    */
    static F32 argMin(const Distribution &f,F32 xmin, F32 xmax,F32 step=0.01);
    /*!
     * \param f input distribution
     * \param xmin lower bound
     * \param xmax upper bound
     * \param step step
     * \return vector of argument miminum local
     *
     * \code
     * Distribution f("(x-0.5)^2*(x-1)^2");
     * Vec<F32> values = Statistics::argMinLocal(f,-10,10);
     * std::cout<<"ArgMinLocal "<<values<<std::endl;
     * f.display(0.25,1.25).save("../doc/image2/ArgMinLocal.jpg");
     *  \endcode
     * \image html ArgMinLocal.jpg
     */
    static Vec<F32> argMinLocal(const Distribution &f,F32 xmin, F32 xmax,F32 step=0.01);
    /*!
     * \param f input distribution
     * \param xmin lower bound
     * \param xmax upper bound
     * \param step step
     * \return vector of argument maximum local
     *
     * \sa argMinLocal
     */
    static std::vector<F32> argMaxLocal(const Distribution &f,F32 xmin, F32 xmax,F32 step=0.01);

    /*! \fn F32 argMax(const Distribution &f,F32 xmin, F32 xmax,F32 step);
     * \param f input distribution
     * \param xmin lower bound
     * \param xmax upper bound
     * \param step step
     * \return argument minimum
     *
     * \sa  argMin
    */
    static F32 argMax(const Distribution &f,F32 xmin, F32 xmax,F32 step=0.01);
    /*! \fn F32 minValue(const Distribution &f,F32 xmin, F32 xmax,F32 step);
     * \param f input distribution
     * \param xmin lower bound
     * \param xmax upper bound
     * \param step step
     * \return minimum value
     *
     *  return the minimum \f$ \min_{x_{min}\leq x \leq x_{max}} f(x)\f$ where the computation is done on the associated regular step function
    */
    static F32 minValue(const Distribution &f,F32 xmin, F32 xmax,F32 step=0.01);

    /*! \fn F32 maxValue(const Distribution &f,F32 xmin, F32 xmax,F32 step);
     * \param f input distribution
     * \param xmin lower bound
     * \param xmax upper bound
     * \param step step
     * \return maximum value
     *
     *  return the maximum\f$ \min_{x_{min}\leq x \leq x_{max}} f(x)\f$ where the computation is done on the associated regular step function
    */
    static F32 maxValue(const Distribution &f,F32 xmin, F32 xmax,F32 step=0.01);

    /*! \fn DistributionRegularStep toProbabilityDistribution(const Distribution &f,F32 xmin, F32 xmax,F32 step);
     * \param f input distribution
     * \param xmin lower bound
     * \param xmax upper bound
     * \param step step
     * \return F32
     *
     *  Convert the input distribution to a probability distribution defined in the range (xmin,xmax)
     *  \code
     *  Distribution f("x");//this function is not a probability distribution
     *  Distribution P = Statistics::toProbabilityDistribution(f,0,1);//create a probability distribution
     *  for(unsigned int i =0;i<100;i++)
     *      std::cout<<P.randomVariable()<<std::endl;//So generate some random variables following this law P(0<X<1)=2x
     *  \endcode
    */
    static DistributionRegularStep toProbabilityDistribution( const Distribution &f,F32 xmin, F32 xmax,F32 step=0.01);


    /*!
     * \param f input distribution
     * \param xmin lower bound
     * \param xmax upper bound
     * \param step step
     * \return F32
     *
     *  Convert the input probability distribution to a Cumulative distribution function defined in the range (xmin,xmax)
     *  \code
     *  DistributionNormal f(0,0.4);
     *  Distribution fcumul = Statistics::toCumulativeProbabilityDistribution(f,-2,2);
     *  Distribution::multiDisplay(f,fcumul,-2,2).save("../doc/image2/cumulative.jpg");
     *  \endcode
     * \image html cumulative.jpg
    */
    static DistributionRegularStep toCumulativeProbabilityDistribution( const Distribution &f,F32 xmin, F32 xmax,F32 step=0.01);



    /*! \fn DistributionRegularStep toStepFunction( const Distribution &f,F32 xmin, F32 xmax,F32 step=0.01);
     * \param f input distribution
     * \param xmin lower bound
     * \param xmax upper bound
     * \param step step
     * \return F32
     *
     *  Convert the input distribution to a step function defined in the range (xmin,xmax) for optimisation purpose
    */
    static DistributionRegularStep toStepFunction( const Distribution &f,F32 xmin, F32 xmax,F32 step=0.01);



    /*! \fn F32 moment(const Distribution &f, int moment,F32 xmin, F32 xmax,F32 step=0.01);
     * \param f input distribution
      * \param moment moment
     * \param xmin lower bound
     * \param xmax upper bound
     * \param step step
     * \return F32
     *
     *  Compute the moment of the distribution \f$\frac{\int_{x_{min}}^{x_{max}} x^n f(x) dx}{\int_{x_{min}}^{x_{max}} f(x) dx}\f$
     *
     * \code
     * DistributionNormal f(3,2);// mean=3 and sigma=2
     * std::cout<<Statistics::moment(f,1,-10,10)<<std::endl;//moment of order 1 (mean)=3
     * std::cout<<Statistics::moment(f,2,-10,10)-std::pow(Statistics::moment(f,1,-10,10),2)<<std::endl;//E{f^2}-E{f}^2=sigma^2=4
     * \endcode
    */
    static F32 moment(const Distribution &f, int moment,F32 xmin, F32 xmax,F32 step=0.01);


    /*! \fn F32 norm(const Distribution &f, F32 norm,F32 xmin, F32 xmax,F32 step=0.01);
     * \param f input distribution
      * \param norm  norm
     * \param xmin lower bound
     * \param xmax upper bound
     * `\param step step
     * \return F32
     *
     *  Compute the norm of the distribution \f$ \int_{x_{min}}^{x_{max}} |f(x)|^n dx\f$
    */
    static F32 norm(const Distribution &f, F32 norm,F32 xmin, F32 xmax,F32 step=0.01);

    /*! \fn F32 productInner(const Distribution &f,const Distribution &g, F32 xmin, F32 xmax,F32 step=0.01);
     * \param f first input distribution
     * \param g second input distribution
     * \param xmin lower bound
     * \param xmax upper bound
     * \param step step
     * \return F32
     *
     *  Compute the inner product of two distribution \f$\int_{x_{min}}^{x_{max}} f(x)*g(x) dx\f$
    */

    static F32 productInner(const Distribution &f,const Distribution &g, F32 xmin, F32 xmax,F32 step=0.01);

    /*! \fn F32 FminusOneOfYMonotonicallyIncreasingFunction(const Distribution &f, F32 y,F32 xmin, F32 xmax,F32 mindiff);
     * \param f first input distribution
     * \param y F32 value
     * \param xmin lower bound
     * \param xmax upper bound
     * \param mindiff mindiff
     * \return F32
     *
     *  find x such that f(x)=y with the property that f is monotone increasing function with mindiff the maximu error
    */

    static F32 FminusOneOfYMonotonicallyIncreasingFunction(const Distribution &f, F32 y,F32 xmin, F32 xmax,F32 mindiff);



    /*!
     * \brief Estimated the probability distribution from a collection of realizations with a regular step function
     * \param v collection of realizations
     * \param step  step of the step function (or staircase function
     * \param min minimum value (by default the min element of the collection)
     * \param max maximum value (by default the max element of the collection)
     * \return probability distribution
     *
     *  Estimated the  probability distribution from a collection of realizations with a regular step function
     *
     *  In the following code, we valide this algorithm by comparing a theorical distribution and the statistical one.
     * \code
     * Distribution ftheorical(10,5,"NORMAL");
     * VecF32 realizations;
     * for(int i=0;i<300000;i++){
     *   F32 v = ftheorical.randomVariable();
     *   realizations.push_back(v);
     * }
     * Distribution dstatistic = Statistics::computedStaticticsFromRealRealizations(realizations,0.1,-10,30);
     * Distribution::multiDisplay(ftheorical,dstatistic,-10,30).save("../doc/image2/computedstatistic.jpg");
     * \endcode
     * \image html computedstatistic.jpg
     * In this code, we valide this algorithm by comparing a theorical distribution and the statistical one. We start with a normal distribution \f$f_X(x)\f$. We consider the random variable
     * Y as the multiplication of the random varibale X : Y=X*X such that X is a random variable following the normal distribution. We sample a number of times this variable and we estimate
     * its Probability density function \f$f_Y(y)\f$.  The theorotical one is equal to  \f$f_Y(y) = \left| \frac{d}{dy} (g^{-1}(y)) \right| \cdot f_X(g^{-1}(y)).\f$ with \f$g(x)=x^2\f$

     * \code
     * Distribution d(10,2,"NORMAL");

     * Distribution fpower2("x^2");
     * Distribution finverse= Statistics::inverse(fpower2,1,30,0.1);
     * Distribution frho = d.rho(finverse);
     * Distribution fderivate=Statistics::derivate(finverse,1,300,1);
     * Distribution ftheorical = fderivate * frho;

     * VecF32 realizations;
     * for(int i=0;i<300000;i++){
     * F32 v = d.randomVariable();
     * realizations.push_back(v*v);
     * }
     * Distribution dstatistic = Statistics::computedStaticticsFromRealRealizations(realizations,2,1,200);
     * Distribution::multiDisplay(ftheorical,dstatistic,1,200).save("../doc/image2/computedstatistic2.jpg");
     * \endcode
     * \image html computedstatistic2.jpg
    */
    static DistributionRegularStep computedStaticticsFromRealRealizations( const VecF32 & v,F32 step =0.1,F32 min=NumericLimits<F32>::minimumRange(),F32 max=NumericLimits<F32>::maximumRange());
    static DistributionRegularStep computedStaticticsFromRealRealizationsWithWeight( const VecF32 & v,const VecF32 & weight,F32 step =0.1,F32 min=NumericLimits<F32>::minimumRange(),F32 max=NumericLimits<F32>::maximumRange());


    /*!
     * \brief estimated the probability distribution from a collection of integer realizations with a regular step function
     * \param v collection of realizations
     * \return probability distribution
     *
     *  Estimated the discrete probability distribution from a collection of integer realizations
     *
     * \code
     * Distribution danalytic(100,"POISSON");
     * VecI32 realizations;
     * for(int i=0;i<2000000;i++){
     *    realizations.push_back(danalytic.randomVariable());
     * }
     * Distribution dstat = Statistics::computedStaticticsFromIntegerRealizations(realizations);
     * F32 mean =    Statistics::moment(dstat,1,0,300);
     * F32 variance = Statistics::moment(dstat,2,0,300)-mean*mean;

     * std::cout<<"mean     "<<mean<<std::endl;//100
     * std::cout<<"variance "<<variance<<std::endl;//100
     * \endcode
     */
    static DistributionIntegerRegularStep computedStaticticsFromIntegerRealizations( const VecI32 & v);




    /*!
     * \brief find the maximum value of the integral interval such that \a integralvalue =\f$\int_{min}^{a} f(x)dx \f$ with \a a< \a maxsupremum
     * \param f input distribution
     * \param integralvalue value of the integral
     * \param min minimum interval
     * \param maxsupremum     minimum interval
     * \param step step size
     * \return maximum interval
     *
     * find the maximum value of the integral interval such that \a integralvalue =\f$\int_{min}^{a} f(x)dx \f$ with \a a< \a maxsupremum
     *
     * \code
     * F32 porosity = 0.6;
     * Distribution f("1/(2*pi)^(1./2)*exp(-(x^2)/2)");
     * F32 beta= Statistics::maxRangeIntegral(f,porosity,-3,3,0.001);//beta=0.257
     * \endcode
    */
    static F32 maxRangeIntegral(const Distribution & f,F32 integralvalue,F32 min, F32 maxsupremum ,F32 step=0.01 );

    //@}
    //-------------------------------------
    //
    //! \name DistributionMultiVariate
    //@{
    //-------------------------------------






    /*! \fn F32 moment(const DistributionMultiVariate &f, VecF32 moment,VecF32 xmin, VecF32 xmax,F32 step=0.01);
     * \param f input distribution
      * \param moment moment
     * \param xmin lower bound
     * \param xmax upper bound
     * \param step step
     * \return F32
     *
     *  Compute the moment of the distribution \f$\frac{\int_{x_{min}}^{x_{max}} x^n f(x) dx}{\int_{x_{min}}^{x_{max}} f(x) dx}\f$
     \code
    DistributionMultiVariate dmulti(DistributionNormal(25,10),2);
    VecF32 xmin(2),xmax(2);
    xmin(0)=-25;xmin(1)=-25;
    xmax(0)=75;xmax(1)=75;
    VecF32 m(2);
    m(0)=1;m(1)=1;
    cout<<Statistics::moment(dmulti,m,xmin,xmax)<<endl;
     \endcode
    */

    static F32 moment(const DistributionMultiVariate &f, VecF32 moment,VecF32 xmin, VecF32 xmax,F32 step=0.1);
    /*! \fn VecF32 argMin(const DistributionMultiVariate &f,VecF32 xmin, VecF32 xmax,F32 step);
     * \param f input distribution
     * \param xmin lower bound
     * \param xmax upper bound
     * \param step step
     * \return F32
     *
     *  return the argument of the minimum \f$ \arg\min_{x_{min}\leq x \leq x_{max}} f(x)\f$ where the computation is done on the associated regular step function
    */
    static VecF32 argMin(const DistributionMultiVariate &f,VecF32 xmin, VecF32 xmax,F32 step=0.01);
    /*! \fn VecF32 argMax(const DistributionMultiVariate &f,VecF32 xmin, VecF32 xmax,F32 step);
     * \param f input distribution
     * \param xmin lower bound
     * \param xmax upper bound
     * \param step step
     * \return F32
     *
     *  return the argument of the maximum \f$ \arg\min_{x_{min}\leq x \leq x_{max}} f(x)=\f$ where the computation is done on the associated regular step function
    */
    static VecF32 argMax(const DistributionMultiVariate &f,VecF32 xmin, VecF32 xmax,F32 step=0.01);
    /*! \fn F32 minValue(const DistributionMultiVariate &f,VecF32 xmin, VecF32 xmax,F32 step);
     * \param f input distribution
     * \param xmin lower bound
     * \param xmax upper bound
     * \param step step
     * \return F32
     *
     *  return the minimum \f$ \min_{x_{min}\leq x \leq x_{max}} f(x)\f$ where the computation is done on the associated regular step function
    */
    static F32 minValue(const DistributionMultiVariate &f,VecF32 xmin, VecF32 xmax,F32 step=0.01);

    /*! \fn F32 maxValue(const DistributionMultiVariate &f,VecF32 xmin, VecF32 xmax,F32 step);
     * \param f input distribution
     * \param xmin lower bound
     * \param xmax upper bound
     * \param step step
     * \return F32
     *
     *  return the maximum\f$ \min_{x_{min}\leq x \leq x_{max}} f(x)\f$ where the computation is done on the associated regular step function
    */
    static F32 maxValue(const DistributionMultiVariate &f,VecF32 xmin, VecF32 xmax,F32 step=0.01);

    /*! \fn  static Distribution  integral(const DistributionMultiVariate &f, VecF32 xmin, VecF32 xmax,F32 step=0.01);
     * \param f input distribution
     * \param xmin lower bound
     * \param xmax upper bound
     * \param step step
     * \return the integral of the DistributionMultiVariate
     *
     *  calculate the integral defined in the range (xmin,xmax) \f$F(x)=\left\{ \begin{array}{ll}
     * \int_{x_{min}}^x f(x) dx & \mbox{for }x_{min}\leq x \leq x_{max}\\
     * 0 &\mbox{otherwise}  \end{array} \right.\f$ where the integral is computed using the upper Riemann sum.

     * \code
     * DistributionMultiVariate d("x*y","x,y");
     * VecF32 xmin(2),xmax(2);
     * xmin(0)=0;xmin(1)=0;
     * xmax(0)=1;xmax(1)=1;
     * DistributionMultiVariate dinter =Statistics::integral(d,xmin,xmax,0.01);//dinter(x,y)=1/4*x^2*y^2
     * VecF32 x(2);
     *
     * x(0)=1;x(1)=1;
     * cout<<dinter(x)<<endl;
     * \endcode
    */
    static DistributionMultiVariateRegularStep  integral(const DistributionMultiVariate &f, VecF32 xmin, VecF32 xmax,F32 step=0.01);
    /*! \fn DistributionMultiVariate toProbabilityDistribution(const DistributionMultiVariate &f,VecF32 xmin, VecF32 xmax,F32 step);
     * \param f input distribution
     * \param xmin lower bound
     * \param xmax upper bound
     * \param step step
     * \return F32
     *
     *  Convert the input distribution to a probability distribution defined in the range (xmin,xmax)
     * \code
     * DistributionMultiVariateExpression dxy;
     * dxy.fromRegularExpression("x","x","y");
     * VecF32 xmin(2),xmax(2);
     * xmin(0)=0;xmin(1)=0;
     * xmax(0)=1;xmax(1)=1;
     * DistributionMultiVariateRegularStep dint = Statistics::toProbabilityDistribution(dxy,xmin,xmax,0.01);
     * for(int i =0;i<100;i++)
     *    std::cout<<dint.randomVariable()<<std::endl;
     * \endcode

    */
    static DistributionMultiVariateRegularStep toProbabilityDistribution(const DistributionMultiVariate &f,VecF32 xmin, VecF32 xmax,F32 step=0.01);


    /*!
     * \param f input bivariate distribution
     * \param xmin lower bound
     * \param xmax upper bound
     * \param step step
     * \return matrix value
     *
     * \code
     * DistributionMultiVariate d("-100*(x-0.5)^2*(y-0.5)^2+2","x,y");

     * VecF32 xmin(2),xmax(2);
     * xmin(0)=0;xmin(1)=0;
     * xmax(0)=1;xmax(1)=1;
     * Mat2F32 m = Statistics::toMatrix(d,xmin,xmax,0.1);
     * Scene3d scene;
     * Visualization::topography(scene,m);
     * Visualization::axis(scene,2,0.5,0);
     * scene.display();
     * \endcode
     * \image html tomatrix.jpg
     */
    static Mat2F32 toMatrix( const DistributionMultiVariate &f,VecF32 xmin, VecF32 xmax,F32 step=0.01);

    //@}

private:
    static void __inverseVecN(F32 x,F32 y, F32 &x_minus_one, F32& y_minus_two);
};





}
#endif // STATISTICS_H
