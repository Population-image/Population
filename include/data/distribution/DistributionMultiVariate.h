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
        \brief abstract class for mapping from multi variate real numbers to the real numbers
        \author Tariel Vincent

      * The class DistributionMultiVariate, not named function to avoid any confusions with the function concept,  is a mapping from the multi variate numbers
      * to the real number where the input element, x=(x_0,x_1,..x_n), completely determines the output element, y,  by this relation: y = f(x_0,x_1,..x_n).
      * For this class, the evaluation of the output element from an input element is a symbolic (implicit) relation for instance,
      * for a function equal to \f$x=(x_0,x_1) \mapsto x_1\exp(-x_0^2)\f$, the output element is evaluated by the multiplication of the input element by itself.\n
      *
      * \note I am still not satisfy with this class, and some major modifications can occur in the next release
    */
public:
    /*!
    \fn virtual ~DistributionMultiVariate();
    *
    *  virtual destructor
    */
    virtual ~DistributionMultiVariate(){}

    /*!
    \fn     virtual F32 operator()(const VecF32& v)const;
    * \param v input  value Vec
    * \return y
    *
    *  Unary function to call y=f(Vec )
    */
    virtual F32 operator()(const VecF32& v)const=0;

    /*!
    \fn  virtual VecF32 randomVariable()const ;
    * \return multivariate random number
    *
    *  Generate random variable, X, following the probability DistributionMultiVariate f.\n
    *  You have to implement this member in any derived class for an analytical DistributionMultiVariate. Otherwise,
    *
    */
    virtual VecF32 randomVariable()const=0;

    /*!
    \fn virtual DistributionMultiVariate * clone();
    * \return  VecNer DistributionMultiVariate
    *
    *  clone pattern used in the factory
    */
    virtual DistributionMultiVariate * clone()const=0;

    /*!
    \fn int getNbrVariable()const;
    * \return nbrvariable number of variables
    *
    *  Get the number of variables
    */
    virtual unsigned int getNbrVariable()const=0;
};
}

#endif // DISTRIBUTIONMULTIVARIATE_H
