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
#ifndef DISTRIBUTIONMULTIVARIATE_H
#define DISTRIBUTIONMULTIVARIATE_H
#include"data/distribution/Distribution.h"
#include"data/vec/Vec.h"
#include"data/mat/MatN.h"
#include"data/mat/Mat2x.h"


namespace pop
{
/*! \ingroup DistributionFunction
* \defgroup DistributionMultiVariate DistributionMultiVariate
* \brief mapping from the multi variate real numbers to the real numbers
*/

class POP_EXPORTS DistributionMultiVariate
{
    /*!
        \class pop::DistributionMultiVariate
        \ingroup DistributionMultiVariate
        \brief mapping from multi variate real numbers to the real numbers
        \author Tariel Vincent

      * The class DistributionMultiVariate, not named function to avoid any confusions with the function concept,  is a mapping from the multi variate numbers
      * to the real number where the input element, x=(x_0,x_1,..x_n), completely determines the output element, y,  by this relation: y = f(x_0,x_1,..x_n).
      * For this class, the evaluation of the output element from an input element is a symbolic (implicit) relation for instance,
      * for a function equal to \f$x=(x_0,x_1) \mapsto x_1\exp(-x_0^2)\f$, the output element is evaluated by the multiplication of the input element by itself.\n
      *
      * This class allows basic operation as addition, subtraction,...
      *
      *
      *
      * \note I am still not satisfy with this class, and some major modifications can occur in the next release
      *
      * \sa
      *

      *
    */
private:
    static unsigned long init[4];
    static unsigned long length;
protected:
    DistributionMultiVariate * _deriveddistribution;
     std::string _key;
    static MTRand_int32 irand;
    mutable F64 _step;
public:
    typedef  VecF64 E;
    typedef  F64 F;
    /*!
    \fn DistributionMultiVariate();
    *
    * default constructor
    */
    DistributionMultiVariate();


    /*!
    \fn DistributionMultiVariate(const DistributionMultiVariate & d);
    * \param d copied multi variate distribution
    *
    *  copy constructor
    *
    * \code
    DistributionMultiVariateExpression dexp;
    dexp.fromRegularExpression("x*y+y","x","y");
    DistributionMultiVariate d(dexp);
    VecF64 v(2);
    v(0)=6;
    v(1)=4;
    std::cout<<d(v)<<std::endl;
    * \endcode
    */
    DistributionMultiVariate(const DistributionMultiVariate & d);

    /*!
    \fn DistributionMultiVariate(const Distribution & d,int nbr_variable_coupled);
   * \param d single  distribution
   * \param nbr_variable_coupled number of coupled random variables
    *
    * generate a random number H(X={x,...,z,})=F(X={x}) for x =...=z
    * \code
    Distribution dmulti(DistributionUniformInt(0,255));
    DistributionMultiVariate dcoupled(dmulti,3);
    VecF64 v = dcoupled.randomVariable();
    v.display();
    \endcode
    The vector v contain three times the same ranmdom variables  following the Uniform probability distribution
    */
    DistributionMultiVariate(const Distribution &F,int nbr_variable_coupled);
    /*!
    \fn DistributionMultiVariate(const Distribution & f);
    *
    *  Construct a distribution such that h(x)=f(x) Idem for the probability distribution H(X={x})=F(X={x})
    *
    * \sa Distribution
    */
    DistributionMultiVariate(const Distribution & f);
    /*!
    \fn DistributionMultiVariate(const DistributionMultiVariate & f,const DistributionMultiVariate & g);
    * \param f input multi variate distribution
    * \param g input multi variate distribution
    *
    *  Construct a distribution such that h(x,y)=f(x)*g(y) Idem for the probability distribution H(X={x,y})=F(X={x})G(Y={y})
    */
    DistributionMultiVariate(const DistributionMultiVariate & f,const DistributionMultiVariate & g);


    /*!
    \fn DistributionMultiVariate(std::string expresssion,std::string variable);
    * \brief regular expression
    * \param expresssion regular expression
    * \param variable variable seperate by ','
    *
    * \code
    DistributionMultiVariate d("exp(-1/3*x^3-x-y^2)","x,y");
    VecF64 v(2);
    v(0)=-1;
    v(0)=0;
    cout<<d(v)<<endl;
    * \endcode

    */
    DistributionMultiVariate(std::string expresssion,std::string variable);


    /*!
    \fn DistributionMultiVariate(VecF64 mean, Mat2F64 covariance);
    * \brief normal disptribution http://en.wikipedia.org/wiki/Multivariate_normal_distribution
    * \param mean mean value
    * \param covariance  covariance (nonnegative-definite matrix)
    *
    \code
    double mux=0;
    double muy=0;
    double sigmax=3;
    double sigmay=1;
    double rho = -0.7;

    VecF64 v(2);
    v(0)=mux;
    v(1)=muy;
    Mat2F64 m(2,2);
    m(0,0)=sigmax*sigmax;    m(0,1)=sigmax*sigmay*rho;
    m(1,0)=sigmax*sigmay*rho;m(1,1)=sigmay*sigmay;

    DistributionMultiVariate multi(v,m);

    Mat2UI8 img(512,512);
    MatNDisplay windows(img);
    while(!windows.is_closed())
    {
        VecF64 v = multi.randomVariable();
        Vec2F64 p= v.toVecN<2>();
        Vec2I32 x= p*25+Vec2F64(img.getDomain())*0.5;
        if(img.isValid(x(0),x(1))){
            img(x)=255;
            windows.display(img);
        }
    }
    \endcode

    */
    DistributionMultiVariate(VecF64 mean, Mat2F64 covariance);


    /*!
    \fn virtual ~DistributionMultiVariate();
    *
    *  virtual destructor
    */
    virtual ~DistributionMultiVariate();

    /*!
    \fn     virtual F64 operator()(const VecF64& v)const;
    * \param v input  value Vec
    * \return y
    *
    *  Unary function to call y=f(Vec )
    */
    virtual F64 operator()(const VecF64& v)const;

    /*!
    \fn  virtual VecF64 randomVariable()const ;
    * \return multivariate random number
    *
    *  Generate random variable, X, following the probability DistributionMultiVariate f.\n
    *  You have to implement this member in any derived class for an analytical DistributionMultiVariate. Otherwise,
    *
    */
    virtual VecF64 randomVariable()const ;

    /*!
    \fn virtual DistributionMultiVariate * clone();
    * \return  VecNer DistributionMultiVariate
    *
    *  clone pattern used in the factory
    */
    virtual DistributionMultiVariate * clone()const ;


    /*!
    \fn DistributionMultiVariate & operator =(const DistributionMultiVariate& d);
    \param d other DistributionMultiVariate
    *
    * Basic assignement of the other DistributionMultiVariate to this DistributionMultiVariate
    */
    DistributionMultiVariate & operator =(const DistributionMultiVariate& d);

    /*!
    \fn DistributionMultiVariate rho(const DistributionMultiVariate &g)const;
    * \param g other  DistributionMultiVariate
    * \return   DistributionMultiVariate
    *
    *  h(x) = (*this)(g(x))
    */
    DistributionMultiVariate rho(const DistributionMultiVariate &g)const;

    /*!
    \fn DistributionMultiVariate  operator +(const DistributionMultiVariate& d)const;
    * \param d other DistributionMultiVariate
    * \return   DistributionMultiVariate
    *
    *  Addition operator :  h(x) = (*this)(x)+d(x)
    */
    DistributionMultiVariate  operator +(const DistributionMultiVariate& d)const;

    /*!
    \fn DistributionMultiVariate  operator -(const DistributionMultiVariate& d)const;
    * \param d other DistributionMultiVariate
    * \return   DistributionMultiVariate
    *
    *  Subtraction operator : h(x) = (*this)(x)-d(x)
    */
    DistributionMultiVariate  operator -(const DistributionMultiVariate& d)const;
    /*!
    \fn DistributionMultiVariate  operator *(const DistributionMultiVariate& d)const;
    * \param d other DistributionMultiVariate
    * \return   DistributionMultiVariate
    *
    *  Multiplication operator : h(x) = (*this)(x)*d(x)
    */
    DistributionMultiVariate  operator *(const DistributionMultiVariate& d)const;

    /*!
    \fn DistributionMultiVariate  operator /(const DistributionMultiVariate& d)const;
    * \param d other DistributionMultiVariate
    * \return   DistributionMultiVariate
    *
    *  Division operator : h(x) = (*this)(x)/d(x)
    */
    DistributionMultiVariate  operator /(const DistributionMultiVariate& d)const;

    /*!
    \fn DistributionMultiVariate  operator -()const;
    * \return   DistributionMultiVariate
    *
    *  Unary minus operator : h(x) = -(*this)(x)
    */
    DistributionMultiVariate  operator -()const;


    /*!
    \fn int getNbrVariable()const;
    * \return nbrvariable number of variables
    *
    *  Get the number of variables
    */
    virtual int getNbrVariable()const;


    /*!
    \fn virtual void setStep(F64 step)const;
    * \param step F64 step
    *
    *  You can ignore the implementation of this member in the derived class.
    * This member inform the object that the subinterval of the partition for calculation as a given step
    *
    * \sa IteratorDistributionMultiVariateSamplingRiemann DistributionMultiVariateDiscrete
    */
    virtual void setStep(F64 step)const;

    DistributionMultiVariate * ___getPointerImplementation();
    const DistributionMultiVariate * ___getPointerImplementation()const ;
    void ___setPointererImplementation(DistributionMultiVariate * d);
};
/*!
\fn pop::DistributionMultiVariate maximum(const pop::DistributionMultiVariate & d1, const pop::DistributionMultiVariate & d2);
* \param d1 other  DistributionMultiVariate
* \param d2 other  DistributionMultiVariate
* \return  DistributionMultiVariate
*
*  h(x) = max(f(x),g(x))
*/
pop::DistributionMultiVariate maximum(const pop::DistributionMultiVariate & d1, const pop::DistributionMultiVariate & d2);

/*!
\fn pop::DistributionMultiVariate minimum(const pop::DistributionMultiVariate & d1, const pop::DistributionMultiVariate & d2);
* \param d1 input  DistributionMultiVariate
* \param d2 input  DistributionMultiVariate
* \return  DistributionMultiVariate
*
*  h(x) = min(f(x),g(x))
*/
pop::DistributionMultiVariate minimum(const pop::DistributionMultiVariate & d1, const pop::DistributionMultiVariate & d2);


}


#endif // DISTRIBUTIONMULTIVARIATE_H
