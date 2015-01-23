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
#ifndef CLIBRARYDISTRIBUTIONMANUAL_HPP
#define CLIBRARYDISTRIBUTIONMANUAL_HPP
#include <cstring>
#include <queue>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <limits>

#include"data/distribution/Distribution.h"
#include"data/distribution/DistributionAnalytic.h"
#include"data/mat/MatN.h"
#include"data/mat/Mat2x.h"

#include"3rdparty/fparser.hh"


namespace pop
{

/// @cond DEV

class POP_EXPORTS DistributionRegularStep:public Distribution
{
private:
    /*!
        \class pop::DistributionRegularStep
        \ingroup Distribution
        \brief step function with regular spacing
        \author Tariel Vincent

    * The step function is a piecewise constant function having only finitely many regular discrete pieces:
    * \f[ f(x)=\sum_{i=0}^{n} a_i {\mathbf 1}_{x\in [xmin+i*spacing,xmin+(i+1)*spacing]} \f].
    *  To define it, we use an input matrix such that the first column contained the X-values with a regular spacing between successive
    *  value and the second colum Y-values.
    * \image html stepfunction.png
     *
    */
    F32 _xmin;
    F32 _xmax;
    F32 _spacing;
    DistributionUniformReal uni;
    std::vector<F32 >_table;
    std::vector<F32 >_repartition;
    void generateRepartition();
public:
        static std::string getKey();
    /*!
    * \fn DistributionRegularStep();
    *
    *   constructor the regular step distribution from a matrix such that the first column contained the X-values with a regular spacing between successive
    *  value and the second colum Y-values.
    */
    DistributionRegularStep();
    /*!
    * \fn DistributionRegularStep(const Mat2F32 & matrix);
    *
    *   constructor the regular step distribution from a matrix such that the first column contained the X-values with a regular spacing between successive
    *  value and the second colum Y-values.
    */
    DistributionRegularStep(const Mat2F32 & matrix);

    /*!
    \fn DistributionRegularStep(const DistributionRegularStep & dist);
    *
    *  copy constructor
    */
    DistributionRegularStep(const DistributionRegularStep & dist);



    virtual F32 operator ()(F32 value)const ;
    virtual DistributionRegularStep * clone()const ;
    virtual F32 randomVariable()const ;


    /*!
    * \param matrix matrix 2d matrix  with two columns with x values and y values respectively
    * \param step set this value if the x-valus spacing is not constant
    *
    *  set the distribution from a matrix such that the first column contained the X-values with a regular spacing between successive
    *  values and the second colum Y-values.\n
    * However if the spacing between successive X-values is not regular but at least a monotonically increasing function, the step is set by default
    * as half of the minumum value of the difference of successive X-values.  X-values width you can define the sampling
    * of the step function with the argument step. Of course,you can define it!
    */
    void fromMatrix(const Mat2F32 & matrix,F32 step=0);

    /*!
    \fn Mat2F32 toMatrix()const;
    *
    *  return a two columns matrix where the first row contains the X-values with a regular spacing and the second column
    * the associated Y-values
    */
    Mat2F32 toMatrix()const;
    /*!
    \fn F32 fMinusOneForMonotonicallyIncreasing(F32 y)const;
    *
    *
    */
    F32 fMinusOneForMonotonicallyIncreasing(F32 y)const;

    void smoothGaussian(F32 sigma);
    virtual F32 getXmin()const;
    virtual F32 getXmax()const;
    virtual F32 getStep()const;
};

class POP_EXPORTS DistributionIntegerRegularStep:public Distribution, public DistributionDiscrete
{
private:
    /*!
        \class pop::DistributionIntegerRegularStep
        \ingroup Distribution
        \brief step function with regular spacing
        \author Tariel Vincent

    * The step function is a piecewise constant function having only finitely many regular discrete pieces:
    * \f[ f(x)=\sum_{i=0}^{n} a_i {\mathbf 1}_{x\in [xmin+i,xmin+(i+1)]} \f].
    *  To define it, we use an input matrix such that the first column contained the X-values with a regular spacing between successive
    *  value equal to 1 and the second colum Y-values.
    * \image html stepfunction.png
     *
    */
    F32 _xmin;
    F32 _xmax;
    DistributionUniformReal uni;
    std::vector<F32 >_table;
    std::vector<F32 >_repartition;
    void generateRepartition();
public:
        static std::string getKey();
    ~DistributionIntegerRegularStep();
    /*!
    * \fn DistributionIntegerRegularStep();
    *
    *   constructor the regular step distribution from a matrix such that the first column contained the X-values with a regular spacing between successive
    *  value equal to 1 and the second colum Y-values.
    */
    DistributionIntegerRegularStep();
    /*!
    * \fn DistributionIntegerRegularStep(const Mat2F32 & matrix);
    *
    *   constructor the regular step distribution from a matrix such that the first column contained the X-values with a regular spacing between successive
    *  value equal to 1  and the second colum Y-values.
    */
    DistributionIntegerRegularStep(const Mat2F32 & matrix);

    /*!
    \fn DistributionIntegerRegularStep(const DistributionIntegerRegularStep & dist);
    *
    *  copy constructor
    */
    DistributionIntegerRegularStep(const DistributionIntegerRegularStep & dist);



    virtual F32 operator ()(F32 value)const ;
    virtual DistributionIntegerRegularStep * clone()const ;
    virtual F32 randomVariable()const ;


    /*!
    \fn void fromMatrix(const Mat2F32 & matrix)
    *
    *  set the distribution from a matrix such that the first column contained the X-values with a regular spacing between successive
    *  values equal to 1 and the second colum Y-values.\n
    * However if the spacing between successive X-values is not regular but at least a monotonically increasing function, the step is set by default
    * as half of the minumum value of the difference of successive X-values.  X-values width you can define the sampling
    * of the step function with the argument step. Of course,you can define it!
    */
    void fromMatrix(const Mat2F32 & matrix);

    /*!
    \fn Mat2F32 toMatrix()const;
    *
    *  return a two columns matrix where the first row contains the X-values with a regular spacing and the second column
    * the associated Y-values
    */
    Mat2F32 toMatrix()const;


    virtual F32 getXmin()const;
    virtual F32 getXmax()const;
};




class POP_EXPORTS DistributionExpression:public Distribution
{
private:
    /*!
        \class pop::DistributionExpression
        \ingroup Distribution
        \brief Parse a regular expression
        \author Tariel Vincent and Juha Nieminen

    * Parsing a regular expression with a single variable  x. For instance,
    \code
    std::string exp = "x*log(x)";
    DistributionExpression d(exp);
    Distribution derivate = Statistics::derivate(d,0.001,10);
    Distribution::multiDisplay(d,derivate,0.001,2.);
    \endcode
    *
    *
    */
     std::string func;
    mutable FunctionParser fparser;
public:
        static std::string getKey();
    ~DistributionExpression();
    DistributionExpression();
    DistributionExpression(std::string regularexpression);
    DistributionExpression(const DistributionExpression & dist);
    virtual F32 operator ()(F32 value)const ;
    F32 randomVariable()const ;
    virtual DistributionExpression * clone()const ;
    bool fromRegularExpression(std::string expression);
     std::string toRegularExpression()const;
};
/// @endcond
}
#endif // CLIBRARYDISTRIBUTIONMANUAL_HPP
