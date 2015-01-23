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
#include"3rdparty/fparser.hh"
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
    /*!
    \fn DistributionMultiVariateUniformInt(const VecI32& xmin,const VecI32& xmax );
    *
    *   constructor the uniform int distribution between the range [xmin,xmax]
    *
    */
    DistributionMultiVariateUniformInt(const VecI32& xmin,const VecI32& xmax );
    VecF32 randomVariable()const ;
    DistributionMultiVariateUniformInt * clone()const ;
    unsigned int getNbrVariable()const;
        virtual F32 operator()(const VecF32& v)const;
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
    /*!
    \fn DistributionMultiVariateUnitSphere(int dimension);
    *
    *   constructor the UnitSphere distribution with the given dimension (d=3=sphere)
    *
    */
    DistributionMultiVariateUnitSphere(int dimension);
    VecF32 randomVariable()const ;
    DistributionMultiVariateUnitSphere * clone()const ;
    unsigned int getNbrVariable()const;
    virtual F32 operator()(const VecF32& v)const;

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
    std::vector<F32 >_repartition;
    VecF32 _xmin;
    MatN<2,F32> _mat2d;
    double _step;
    DistributionMultiVariateRegularStep(const MatN<2,F32> data_x_y, VecF32& xmin,F32 step);
    F32 operator ()(const VecF32&  value)const;
    DistributionMultiVariateRegularStep * clone()const ;
    VecF32 randomVariable()const ;
    unsigned int getNbrVariable()const;
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
    DistributionExpression d("std::exp(x*std::log(x))");
    DistributionMultiVariateFromDistribution dmulti(f);
    dmulti.fromDistribution(d);
    VecF32 v(1);
    v(0)=6;
    std::cout<<dmulti(v)<<std::endl;
    *\endcode
    * \sa Distribution
    */

    Distribution *_f;

public:
    ~DistributionMultiVariateFromDistribution();
    DistributionMultiVariateFromDistribution(const DistributionMultiVariateFromDistribution &f);
    DistributionMultiVariateFromDistribution operator ()(const DistributionMultiVariateFromDistribution &f);
    DistributionMultiVariateFromDistribution(const Distribution &f);
    virtual DistributionMultiVariateFromDistribution * clone()const ;
    virtual F32 operator ()(const VecF32&  value)const;
    VecF32 randomVariable()const ;
    unsigned int getNbrVariable()const;
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
    F32 mux=0;
    F32 muy=0;
    F32 sigmax=3;
    F32 sigmay=1;
    F32 rho = -0.7;
    DistributionMultiVariateNormal multi;
    VecF32 v(2);

    v(0)=mux;
    v(1)=muy;
    Mat2F32 m(2,2);
    m(0,0)=sigmax*sigmax;    m(0,1)=sigmax*sigmay*rho;
    m(1,0)=sigmax*sigmay*rho;m(1,1)=sigmay*sigmay;
    multi.fromMeanVecAndCovarianceMatrix(v,m);

    Mat2UI8 img(512,512);
    MatNDisplay windows;
    while(0==0)
    {
        VecF32 v = multi.randomVariable();
        Vec2F32 p= v;
        Vec2I32 x= p*25+Vec2F32(img.getDomain())*0.5;
        if(img.isValid(x(0),x(1))){
            img(x)=255;
            windows.display(img);
        }
    }
    \endcode
    */
    Mat2F32 _sigma;//covariance matrix
    VecF32 _mean;//mean std::vector
    Mat2F32 _a;// _A _A^t = _sigma to sample random variables
    F32 _determinant_sigma;
    Mat2F32 _sigma_minus_one;
    DistributionNormal _standard_normal;
public:

    virtual F32 operator ()(const VecF32&  value)const;
    VecF32 randomVariable()const ;
    virtual DistributionMultiVariateNormal * clone()const ;
    DistributionMultiVariateNormal(VecF32 mean, Mat2F32 covariance);
    unsigned int getNbrVariable()const;
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
        DistributionMultiVariateExpression dexp("std::exp(x*std::log(x))*y","x","y");
        VecF32 v(2);
        v(0)=6;
        v(1)=4;
        std::cout<<dexp(v)<<std::endl;
    \endcode
    *
    */
    std::string _func;
    std::string _concatvar;
    mutable FunctionParser fparser;
    int _nbrvariable;
    bool fromRegularExpression(std::pair<std::string,std::string> regularexpressionAndconcatvar);
public:
    virtual F32 operator ()(const VecF32&  value)const;
    VecF32 randomVariable()const ;
    virtual DistributionMultiVariateExpression * clone()const ;

    DistributionMultiVariateExpression(std::string expression,std::string var1);
    DistributionMultiVariateExpression(std::string expression,std::string var1,std::string var2);
    DistributionMultiVariateExpression(std::string expression,std::string var1,std::string var2,std::string var3);
    DistributionMultiVariateExpression(std::string expression,std::string var1,std::string var2,std::string var3,std::string var4);
    unsigned int getNbrVariable()const;
};
/// @endcond
}
#endif // DISTRIBUTIONMULTIVARIATEFROMDATASTRUCTURE_H
