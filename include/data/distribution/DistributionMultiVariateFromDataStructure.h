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
#ifndef DISTRIBUTIONMULTIVARIATEFROMDATASTRUCTURE_H
#define DISTRIBUTIONMULTIVARIATEFROMDATASTRUCTURE_H
#include <cstring>
#include <queue>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <limits>

#include"data/distribution/DistributionMultiVariate.h"
#include"dependency/fparser.hh"
#include"data/mat/MatN.h"
#include"data/mat/Mat2x.h"

#include"data/vec/Vec.h"
#include"data/distribution/DistributionAnalytic.h"


namespace pop
{
/// @cond DEV

class POP_EXPORTS DistributionMultiVariateUniformInt:public DistributionMultiVariate
{
    /*!
        \class pop::DistributionMultiVariateUniformInt
        \ingroup DistributionMultiVariate
        \brief uniform int
        \author Tariel Vincent

    */
private:

    DistributionUniformReal _d;
public:
    VecI32 _xmin;
    VecI32 _xmax;
    int getDIM()const;
    /*!
    \fn DistributionMultiVariateUniformInt(const VecI32& xmin,const VecI32& xmax );
    *
    *   constructor the uniform int distribution between the range [xmin,xmax]
    *
    */
    DistributionMultiVariateUniformInt(const VecI32& xmin,const VecI32& xmax );

    /*!
    \fn DistributionMultiVariateUniformInt(const DistributionMultiVariateUniformInt & dist);
    *
    *  copy constructor
    */
    DistributionMultiVariateUniformInt(const DistributionMultiVariateUniformInt & dist);
    VecF64 randomVariable()const throw(pexception);
    DistributionMultiVariateUniformInt * clone()const throw(pexception);
};



class POP_EXPORTS DistributionMultiVariateUnitSphere:public DistributionMultiVariate
{
    /*!
        \class pop::DistributionMultiVariateUnitSphere
        \ingroup DistributionMultiVariate
        \brief unit sphere distribution
        \author Tariel Vincent

      * Sphere VecN Picking http://mathworld.wolfram.com/SphereVecNPicking.html
      *
      *

      *
    */
private:
    int _dim;
    DistributionUniformReal d2pi;
    DistributionUniformReal d2;
public:
    int getDIM()const;
    /*!
    \fn DistributionMultiVariateUnitSphere(int dimension);
    *
    *   constructor the UnitSphere distribution with the given dimension (d=3=sphere)
    *
    */
    DistributionMultiVariateUnitSphere(int dimension);

    /*!
    \fn DistributionMultiVariateUnitSphere(const DistributionMultiVariateUnitSphere & dist);
    *
    *  copy constructor
    */
    DistributionMultiVariateUnitSphere(const DistributionMultiVariateUnitSphere & dist);
    VecF64 randomVariable()const throw(pexception);
    DistributionMultiVariateUnitSphere * clone()const throw(pexception);

};


class POP_EXPORTS DistributionMultiVariateRegularStep:public DistributionMultiVariate
{
private:
    void generateRepartition();

    /*!
        \class pop::DistributionMultiVariateRegularStep
        \ingroup DistributionMultiVariate
        \brief step function with regular spacing
        \author Tariel Vincent

    * As DistributionRegularStep.Actually, we cannot convert a matrix to a DistributionMultiVariateRegularStep (a work to do)
    * \sa DistributionMultiVariateRegularStep
     *
    */
public:
    DistributionUniformReal uni;
    std::vector<F64 >_repartition;
    VecF64 _xmin;
    MatN<2,F64> _mat2d;
//    MatN<3,F64> _mat3d;
//    MatN<4,F64> _mat4d;
//    MatN<5,F64> _mat5d;


    /*!
    * \fn DistributionMultiVariateRegularStep();
    *
    *   constructor the regular step distribution from a matrix such that the first column contained the X-values with a regular spacing between successive
    *  value and the second colum Y-values.
    */
    DistributionMultiVariateRegularStep();
    DistributionMultiVariateRegularStep(const DistributionMultiVariateRegularStep & dist);

    DistributionMultiVariateRegularStep(const MatN<2,F64> data_x_y, VecF64& xmin,F64 step);
//    DistributionMultiVariateRegularStep(const MatN<3,F64> data_x_y_z, VecF64& xmin,F64 step);
//    DistributionMultiVariateRegularStep(const MatN<4,F64> data_x_y_z_w, VecF64& xmin,F64 step);
    virtual F64 operator ()(const VecF64&  value)const;
    virtual DistributionMultiVariateRegularStep * clone()const throw(pexception);
    virtual VecF64 randomVariable()const throw(pexception);

    virtual int getNbrVariable()const;
};




class POP_EXPORTS DistributionMultiVariateFromDistribution : public DistributionMultiVariate
{
private:
    /*!
        \class pop::DistributionMultiVariateFromDistribution
        \ingroup DistributionMultiVariate
        \brief convert a Distribution to a DistributionMultiVariate
        \author Tariel Vincent
    *
    * \code
    Distribution d("std::exp(x*std::log(x))");
    DistributionMultiVariateFromDistribution dmulti;
    dmulti.fromDistribution(d);
    VecF64 v(1);
    v(0)=6;
    std::cout<<dmulti(v)<<std::endl;
    *\endcode
    * \sa Distribution
    */

    Distribution _f;

public:
    ~DistributionMultiVariateFromDistribution();
    DistributionMultiVariateFromDistribution();
    DistributionMultiVariateFromDistribution(const DistributionMultiVariateFromDistribution & dist);

    virtual DistributionMultiVariateFromDistribution * clone()const throw(pexception);
    virtual F64 operator ()(const VecF64&  value)const;
    VecF64 randomVariable()const throw(pexception);
    void setStep(F64 step)const;
    void fromDistribution(const Distribution &d);
    Distribution  toDistribution()const;
        virtual int getNbrVariable()const;
};


class POP_EXPORTS DistributionMultiVariateNormal:public DistributionMultiVariate
{
private:
    /*!
        \class pop::DistributionMultiVariateExpression
        \ingroup DistributionMultiVariate
        \brief Multivariate normal distribution
        \author Tariel Vincent
    * see http://en.wikipedia.org/wiki/Multivariate_normal_distribution
    \code
    double mux=0;
    double muy=0;
    double sigmax=3;
    double sigmay=1;
    double rho = -0.7;
    DistributionMultiVariateNormal multi;
    VecF64 v(2);

    v(0)=mux;
    v(1)=muy;
    Mat2F64 m(2,2);
    m(0,0)=sigmax*sigmax;    m(0,1)=sigmax*sigmay*rho;
    m(1,0)=sigmax*sigmay*rho;m(1,1)=sigmay*sigmay;
    multi.fromMeanVecAndCovarianceMatrix(v,m);

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
    Mat2F64 _sigma;//covariance matrix
    VecF64 _mean;//mean std::vector
    Mat2F64 _a;// _A _A^t = _sigma to sample random variables
    F64 _determinant_sigma;
    Mat2F64 _sigma_minus_one;
    DistributionNormal _standard_normal;
public:
    DistributionMultiVariateNormal();
    DistributionMultiVariateNormal(const DistributionMultiVariateNormal & dist);
    virtual F64 operator ()(const VecF64&  value)const;
    VecF64 randomVariable()const throw(pexception);
    virtual DistributionMultiVariateNormal * clone()const throw(pexception);
    void fromMeanVecAndCovarianceMatrix(VecF64 mean, Mat2F64 covariance);
    void fromMeanVecAndCovarianceMatrix(std::pair<VecF64, Mat2F64> meanvectorAndcovariancematrix);
    std::pair<VecF64,Mat2F64> toMeanVecAndCovarianceMatrix()const;
    virtual int getNbrVariable()const;

};



class POP_EXPORTS DistributionMultiVariateExpression:public DistributionMultiVariate
{
private:
    /*!
        \class pop::DistributionMultiVariateExpression
        \ingroup DistributionMultiVariate
        \brief Pencil function
        \author Tariel Vincent and Juha Nieminen

    * Parsing a regular expression with a single variable  x. For instance,
    *  std::string exp = "std::exp(x*std::log(x))*y"
    *
    \code
        DistributionMultiVariateExpression dexp;
        dexp.fromRegularExpression("std::exp(x*std::log(x))*y","x","y");
        VecF64 v(2);
        v(0)=6;
        v(1)=4;
        std::cout<<dexp(v)<<std::endl;
    \endcode
    *
    */
     std::string _func;
     std::string _concatvar;
    mutable FunctionParser fparser;


public:
    bool fromRegularExpression(std::pair<std::string,std::string> regularexpressionAndconcatvar);
    int _nbrvariable;
    DistributionMultiVariateExpression();
    DistributionMultiVariateExpression(const DistributionMultiVariateExpression & dist);
    virtual F64 operator ()(const VecF64&  value)const;
    VecF64 randomVariable()const throw(pexception);
    virtual DistributionMultiVariateExpression * clone()const throw(pexception);

    bool fromRegularExpression(std::string expression,std::string var1);
    bool fromRegularExpression(std::string expression,std::string var1,std::string var2);
    bool fromRegularExpression(std::string expression,std::string var1,std::string var2,std::string var3);
    bool fromRegularExpression(std::string expression,std::string var1,std::string var2,std::string var3,std::string var4);
    std::pair<std::string,std::string> toRegularExpression()const;

    virtual int getNbrVariable()const;
};
/// @endcond

}
#endif // DISTRIBUTIONMULTIVARIATEFROMDATASTRUCTURE_H
