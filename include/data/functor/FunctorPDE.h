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
#ifndef FUNCTORPDE_H
#define FUNCTORPDE_H
#include<vector>
#include"data/typeF/TypeTraitsF.h"
#include"data/functor/FunctorF.h"
#include"data/mat/MatN.h"
#include"data/mat/Mat2x.h"
#include"data/functor/FunctorMatN.h"
namespace pop
{
/// @cond DEV
/*! \ingroup Functor
* \defgroup FunctorPDE FunctorPDE
* \brief Collection of functors representing differential operator in finite difference scheme
*
* For instance the homogeneous diffusion equation in finite difference scheme
* \code
    Mat2UI8 img;
    img.load("../image/Lena.bmp");
    Mat2F32 img_timet(img);//operation on float
    Mat2F32 img_timet_plus_one(img);
    FunctorPDE::Laplacien<> laplacien;

    F32 D=0.25;//diffusion coefficient
    MatNDisplay disp;
    for(unsigned int index_time=0;index_time<300;index_time++){
        std::cout<<index_time<<std::endl;
        ForEachDomain2D(x,img_timet) {
            img_timet_plus_one(x) = img_timet(x) + D*laplacien(img_timet,x);
        }
        img_timet = img_timet_plus_one;//timet<-timet_plus_one
        disp.display(img_timet);//display the current image
    }
* \endcode
*
*
*/
class POP_EXPORTS FunctorPDE
{
public:

    /*!
     * \class pop::FunctorPDE
     * \ingroup FunctorPDE
     * \brief Collection of functors representing differential operator in finite difference scheme
     * \author Tariel Vincent
     *
     * \code
    Mat2UI8 img;
    img.load("../image/Lena.bmp");
    Mat2F32 img_timet(img);//operation on float
    Mat2F32 img_timet_plus_one(img);
    FunctorPDE::Laplacien<> laplacien;

    F32 D=0.25;//diffusion coefficient
    MatNDisplay disp;
    for(unsigned int index_time=0;index_time<300;index_time++){
        std::cout<<index_time<<std::endl;
        ForEachDomain2D(x,img_timet) {
            img_timet_plus_one(x) = img_timet(x) + D*laplacien(img_timet,x);
        }
        img_timet = img_timet_plus_one;//timet<-timet_plus_one
        disp.display(img_timet);//display the current image
    }
     * \endcode
     *
    */

    //-------------------------------------
    //
    //! \name partial derivate
    //@{
    //-------------------------------------

    /*!
    \class pop::FunctorPDE::PartialDerivateCentered
    \brief Partial derivative in centered difference finite  \f$\frac{\partial f}{\partial x_i}\f$ and \f$\frac{\partial \overrightarrow{f}}{\partial x_i}\f$
    \tparam div number in centered difference finite \f$(f(i+1,j)-f(i-1,j))/DIV\f$
    *
    * Functor to approximate the partial derivative in centered difference finite. For a VecN falling outside the field domain, the partial derivative is NULL (Neumann boundary condition).
    */
    class POP_EXPORTS PartialDerivateCentered
    {
    public:
        static const F32 DIV_INVERSE;
        /*!
        * \brief partial derivate for scalar/vectoriel field
        * \param  f input scalar/vectoriel field
        * \param  x input VecN
        * \param  coordinate derivative direction
        * \return a scalar value
        *
        * For a scalar field, the functor returns the partial derivative \f$\frac{\partial f}{\partial x_i}\f$ in centered difference finite that is in 2d  equal \f$(f(i+1,j)-f(i-1,j))/DIV\f$ for coordinate=0 and x=(i,j).\n
        * For a vectoriel field, the functor returns the partial derivative \f$(\frac{\partial f_0}{\partial x_i},\frac{\partial f_1}{\partial x_i}),\ldots\f$ in centered difference finite.
        *
        *
        *
        */
        template<int DIM,typename TypePixel>
        inline TypePixel operator()(const MatN<DIM,TypePixel> &f, typename MatN<DIM,TypePixel>::E  x,int coordinate)
        {
            x(coordinate)++;
            TypePixel y;
            if( x(coordinate)<f.getDomain()(coordinate)){
                y= f(x);
                x(coordinate)-=2;
                if(x(coordinate)>=0)
                {
                    y-=f(x);
                    y*=0.5;
                    x(coordinate)++;
                    return y;
                }
                else{
                    x(coordinate)++;
                    return TypePixel(0);
                }
            }else{
                x(coordinate)--;
                return TypePixel(0);
            }

        }
    };
    /*!
    \class pop::FunctorPDE::PartialDerivateForward
    \brief Partial derivative in forward difference finite  \f$\frac{\partial f}{\partial x_i}\f$ and \f$\frac{\partial  \overrightarrow{f}}{\partial x_i}\f$
    *
    * Functor to approximate the partial derivative in forward difference finite.
    * For a VecN falling outside the field domain, the partial derivative is NULL (Neumann boundary condition).
    */
    class POP_EXPORTS PartialDerivateForward
    {
    public:
        /*!
        * \brief partial derivate for scalar/vectoriel field
        * \param  f input scalar/vectoriel field
        * \param  x input VecN
        * \param  coordinate derivative direction
        * \return a scalar value
        *
        * For a scalar field, the functor returns the partial derivative \f$\frac{\partial f}{\partial x_i}\f$ in centered difference finite that is in 2d  equal \f$f(i+1,j)-f(i,j)\f$ for coordinate=0 and x=(i,j).\n
        * For a vectoriel field, the functor returns the partial derivative \f$(\frac{\partial f_0}{\partial x_i},\frac{\partial f_1}{\partial x_i}),\ldots\f$.
        */
        template<int DIM,typename TypePixel>
        inline TypePixel operator()(const MatN<DIM,TypePixel> &f,  typename MatN<DIM,TypePixel>::E  x,int coordinate)
        {
            TypePixel y=TypePixel(0)-f(x);
            x(coordinate)++;
            if( x(coordinate)<f.getDomain()(coordinate)){
                y+=f(x);
                x(coordinate)--;
                return y;
            }else{
                x(coordinate)--;
                return TypePixel(0);
            }

        }

    };
    /*!
    \class pop::FunctorPDE::PartialDerivateBackward
    \brief Partial derivative in backward difference finite  \f$\frac{\partial f}{\partial x_i}\f$ and \f$\frac{\partial  \overrightarrow{f}}{\partial x_i}\f$
    *
    * Functor to approximate the partial derivative in backward difference finite
    * For a VecN falling outside the field domain, the partial derivative is NULL (Neumann boundary condition).
    */
    class POP_EXPORTS PartialDerivateBackward
    {
    public:
        /*!
        * \brief partial derivate for scalar/vectoriel field
        * \param  f input scalar/vectoriel field
        * \param  x input VecN
        * \param  coordinate derivative direction
        * \return a scalar value
        *
        * For a scalar field, the functor returns the partial derivative \f$\frac{\partial f}{\partial x_i}\f$ in centered difference finite that is in 2d  equal \f$f(i,j)-f(i-1,j)\f$ for coordinate=0 and x=(i,j).\n
        * For a vectoriel field, the functor returns the partial derivative \f$(\frac{\partial f_0}{\partial x_i},\frac{\partial f_1}{\partial x_i}),\ldots\f$.
        */
        template<int DIM,typename TypePixel>
        inline TypePixel operator()(const MatN<DIM,TypePixel> &f,  typename MatN<DIM,TypePixel>::E  x,int coordinate)
        {
            TypePixel y=f(x);
            x(coordinate)--;
            if( x(coordinate)>=0){
                y-=f(x);
                x(coordinate)++;
                return y;
            }else{
                x(coordinate)++;
                return TypePixel(0);
            }

        }
    };
    /*!
    \class pop::FunctorPDE::PartialDerivateSecondCentered
    \brief second partial derivative in centered difference finite  \f$\frac{\partial^2 f}{\partial x_i\partial x_j}\f$ and \f$\frac{\partial^2 \overrightarrow{f}}{\partial x_i\partial x_j}\f$
    *
    * Functor to approximate the second partial derivative with Neumann boundary condition
    *
    */
    class POP_EXPORTS PartialDerivateSecondCentered
    {
    private:
        PartialDerivateForward partiatforward;
        PartialDerivateBackward partiatbackward;
        PartialDerivateCentered partiatcentered;
    public:
        /*!
        * \brief partial derivate for scalar/vectoriel field
        * \param  f input scalar/vectoriel field
        * \param  x input VecN
        * \param  coordinate_i first derivative direction
        * \param  coordinate_j second derivative direction
        * \return a scalar value
        *
        * For a scalar field, the functor returns the second partial derivative \f$\frac{\partial^2 f}{\partial x_j \partial x_i }\f$ in centered difference finite that is in 2d  equal
        *   - \f$f(i+1,j)+f(i-1,j)-2*f(i,j)\f$ for coordinate_i=coordinate_j=0
        *   - \f$(f(i+1,j+1)-f(i-1,j+1))-(f(i+1,j-1)-f(i-1,j-1))/4\f$ for coordinate_i=0 and coordinate_j=1.\n
        *
        * For a vectoriel field, the functor returns the vector partial derivative \f$(\frac{\partial f_0}{\partial x_i},\frac{\partial f_1}{\partial x_i}),\ldots\f$.
        */
        template<int DIM,typename TypePixel>
        inline TypePixel operator()(const MatN<DIM,TypePixel> &f, typename MatN<DIM,TypePixel>::E  x,int coordinate_i,int coordinate_j )
        {
            if(coordinate_i==coordinate_j){
                return partiatforward(f,x,coordinate_i)-partiatbackward(f,x,coordinate_i);
            }
            else{
                TypePixel y(0);
                x(coordinate_j)++;
                if( x(coordinate_j)<f.getDomain()(coordinate_j)){
                    y=partiatcentered(f,x,coordinate_i);
                }
                x(coordinate_j)-=2;
                if( x(coordinate_j)>=0){
                    y-=partiatcentered(f,x,coordinate_i);
                }
                x(coordinate_j)++;
                return y*0.5;
            }
        }

    };

    /*!
    \class pop::FunctorPDE::PartialDerivateCenteredInBulk
    \brief Partial derivative in centered difference finite  \f$\frac{\partial f}{\partial x_i}\f$ and \f$\frac{\partial \overrightarrow{f}}{\partial x_i}\f$
    *
    * Functor to approximate the partial derivative in centered difference finite for a scalar/vectoriel field. For a VecN falling outside the bulk \f$\{x:bulk(x)\neq 0\} \f$, the partial derivative is NULL (Neumann boundary condition).
    * \sa PartialDerivateCentered
    */
    template<int DIM>
    class POP_EXPORTS PartialDerivateCenteredInBulk
    {
    private:
        const MatN<DIM,UI8> * _bulk;
    public:
        void setBulk(const MatN<DIM,UI8> & bulk){
            _bulk = & bulk;
        }
        /*!
        * \brief partial derivate for scalar/vectoriel field
        * \param  f input scalar/vectoriel field
        * \param  x input VecN
        * \param  coordinate derivative direction
        * \return a scalar value
        *
        * The functor returns the partial derivative \f$\frac{\partial f}{\partial x_i}\f$ in centered difference finite that is in 2d  equal \f$(f(i+1,j)-f(i-1,j))/2\f$ for coordinate=0 and x=(i,j).
        *
        */
        template<typename TypePixel>
        inline TypePixel operator()(const MatN<DIM,TypePixel> &f, const typename MatN<DIM,TypePixel>::E & x,int coordinate)
        {
            x(coordinate)++;
            TypePixel y;
            if( x(coordinate)<f.getDomain()(coordinate)&&_bulk->operator()(x)!=0){
                y= f(x);
                x(coordinate)-=2;
                if(x(coordinate)>=0&&_bulk->operator()(x)!=0)
                {
                    y-=f(x);
                    y/=2;
                    x(coordinate)++;
                    return y;
                }
                else{
                    x(coordinate)++;
                    return TypePixel(0);
                }
            }else{
                x(coordinate)--;
                return TypePixel(0);
            }

        }
    };
    /*!
    \class pop::FunctorPDE::PartialDerivateForwardInBulk
    \brief Partial derivative in forward difference finite  \f$\frac{\partial f}{\partial x_i}\f$ and \f$\frac{\partial \overrightarrow{f}}{\partial x_i}\f$
    *
    * Functor to approximate the partial derivative in forward difference finite. For a VecN falling outside the bulk \f$\{x:bulk(x)\neq 0\} \f$, the partial derivative is NULL (Neumann boundary condition).
    */
    template<int DIM>
    class POP_EXPORTS PartialDerivateForwardInBulk
    {
    private:
        const MatN<DIM,UI8> * _bulk;
    public:
        void setBulk(const MatN<DIM,UI8> & bulk){
            _bulk = & bulk;
        }
        /*!
        * \brief partial derivate for scalar/vectoriel field
        * \param  f input scalar/vectoriel field
        * \param  x input VecN
        * \param  coordinate derivative direction
        * \return a scalar value
        *
        * The functor returns the partial derivative \f$\frac{\partial f}{\partial x_i}\f$ in centered difference finite.\n
        * For a VecN falling outside the field domain, the partial derivative is NULL (Neumann boundary condition).
        * \sa PartialDerivateForward
        */
        template<typename TypePixel>
        inline TypePixel operator()(const MatN<DIM,TypePixel> &f, const typename MatN<DIM,TypePixel>::E & x,int coordinate)
        {
            TypePixel y=-f(x);
            x(coordinate)++;
            if( x(coordinate)<f.getDomain()(coordinate) &&_bulk->operator()(x)!=0){
                y+=f(x);
                x(coordinate)--;
                return y;
            }else{
                x(coordinate)--;
                return TypePixel(0);
            }

        }
    };
    /*!
    \class pop::FunctorPDE::PartialDerivateBackwardInBulk
    \brief Partial derivative in backward difference finite  \f$\frac{\partial f}{\partial x_i}\f$ and \f$\frac{\partial \overrightarrow{f}}{\partial x_i}\f$
    *
    * Functor to approximate the partial derivative in backward difference finite
    * For a VecN falling outside the bulk \f$\{x:bulk(x)\neq 0\} \f$, the partial derivative is NULL (Neumann boundary condition).
    */
    template<int DIM>
    class POP_EXPORTS PartialDerivateBackwardInBulk
    {
    private:
        const MatN<DIM,UI8> * _bulk;
    public:
        void setBulk(const MatN<DIM,UI8> & bulk){
            _bulk = & bulk;
        }
        /*!
        * \brief partial derivate for scalar/vectoriel field
        * \param  f input scalar/vectoriel field
        * \param  x input VecN
        * \param  coordinate derivative direction
        * \return a scalar value
        *
        * The functor returns the partial derivative \f$\frac{\partial f}{\partial x_i}\f$.
        * \sa PartialDerivateBackward
        */
        template<typename TypePixel>
        inline TypePixel operator()(const MatN<DIM,TypePixel> &f, const typename MatN<DIM,TypePixel>::E & x,int coordinate)
        {
            TypePixel y=f(x);
            x(coordinate)--;
            if( x(coordinate)>=0&&_bulk->operator()(x)!=0){
                y-=f(x);
                x(coordinate)++;
                return y;
            }else{
                x(coordinate)++;
                return TypePixel(0);
            }

        }
    };
    /*!
    \class pop::FunctorPDE::PartialDerivateSecondCenteredInBulk
    \brief second partial derivative in centered difference finite  \f$\frac{\partial^2 f}{\partial x_i\partial x_j}\f$ and \f$\frac{\partial^2 \overrightarrow{f}}{\partial x_i\partial x_j}\f$
    *
    * Functor to approximate the second partial derivative. For a VecN falling outside the bulk \f$\{x:bulk(x)\neq 0\} \f$, the partial derivative is NULL (Neumann boundary condition).
    *
    *
    */
    template<int DIM>
    class POP_EXPORTS PartialDerivateSecondCenteredInBulk
    {
    private:
        const MatN<DIM,UI8> * _bulk;
    public:
        void setBulk(const MatN<DIM,UI8> & bulk){
            _bulk = & bulk;
            partiatforward.setBulk(bulk);
            partiatbackward.setBulk(bulk);
            partiatcentered.setBulk(bulk);
        }
        PartialDerivateForwardInBulk<DIM > partiatforward;
        PartialDerivateBackwardInBulk<DIM > partiatbackward;
        PartialDerivateCenteredInBulk<DIM > partiatcentered;
        /*!
        * \brief partial derivate for scalar/vectoriel field
        * \param  f input scalar/vectoriel field
        * \param  x input VecN
        * \param  coordinate_i first derivative direction
        * \param  coordinate_j second derivative direction
        * \return a scalar value
        *
        * The functor returns the second partial derivative \f$\frac{\partial^2 f}{\partial x_j \partial x_i }\f$ in centered difference finite that is in 2d  equal
        *   - \f$f(i+1,j)+f(i-1,j)-2*f(i,j)\f$ for coordinate_i=coordinate_j=0
        *   - \f$(f(i+1,j+1)-f(i-1,j+1))-(f(i+1,j-1)-f(i-1,j-1))\f$ for coordinate_i=0 and coordinate_j=1
        */
        template<typename TypePixel>
        inline TypePixel operator()(const MatN<DIM,TypePixel> &f, const typename MatN<DIM,TypePixel>::E & x,int coordinate_i,int coordinate_j )
        {
            if(coordinate_i==coordinate_j){
                return partiatforward(f,x,coordinate_i)-partiatbackward(f,x,coordinate_i);
            }
            else{
                TypePixel y(0);
                x(coordinate_j)++;
                if( x(coordinate_j)<f.getDomain()(coordinate_j)&&_bulk->operator()(x)!=0){
                    y=partiatcentered(f,x,coordinate_i);
                }
                x(coordinate_j)-=2;
                if( x(coordinate_j)>=0&&_bulk->operator()(x)!=0){
                    y-=partiatcentered(f,x,coordinate_i);
                }
                x(coordinate_j)++;
                return y;
            }
        }
    };
    /*!
    \class pop::FunctorPDE::PartialDerivateCenteredMultiPhaseField
    \brief Partial derivative in centered difference finite  \f$\frac{\partial f}{\partial x_i}\f$ and \f$\frac{\partial \overrightarrow{f}}{\partial x_i}\f$
    *
    * Functor to approximate the partial derivative in centered difference finite for a scalar/vectoriel field in multi-phase field formalism.
    * \sa PartialDerivateCentered
    */
    template<int DIM,typename TypePixelLabel>
    class POP_EXPORTS PartialDerivateCenteredMultiPhaseField
    {
    private:
        const MatN<DIM,TypePixelLabel> * _label;
    public:
        void setLabelPhase(const MatN<DIM,TypePixelLabel> & label){
            _label = & label;
        }
        /*!
        * \brief partial derivate for scalar/vectoriel field
        * \param  f input scalar/vectoriel field
        * \param  x input VecN
        * \param  coordinate derivative direction
        * \return a scalar value
        *
        * The functor returns the partial derivative \f$\frac{\partial f}{\partial x_i}\f$ in centered difference finite that is in 2d  equal \f$(f(i+1,j)-f(i-1,j))/2\f$ for coordinate=0 and x=(i,j).
        *
        */
        template<typename TypePixel>
        inline TypePixel operator()(const MatN<DIM,TypePixel> &f, const typename MatN<DIM,TypePixel>::E & x,int coordinate)
        {
            TypePixelLabel label = _label->operator()(x);
            x(coordinate)++;
            TypePixel y;
            if( x(coordinate)<f.getDomain()(coordinate)){
                if(label==_label->operator()(x))
                    y= f(x);
                else
                    y=-f(x);
                x(coordinate)-=2;
                if(x(coordinate)>=0)
                {
                    if(label==_label->operator()(x))
                        y-=f(x);
                    else
                        y+=f(x);
                    y/=2;
                    x(coordinate)++;
                    return y;
                }
                else{
                    x(coordinate)++;
                    return TypePixel(0);
                }
            }else{
                x(coordinate)--;
                return TypePixel(0);
            }

        }
    };
    /*!
    \class pop::FunctorPDE::PartialDerivateForwardMultiPhaseField
    \brief Partial derivative in forward difference finite  \f$\frac{\partial f}{\partial x_i}\f$ and \f$\frac{\partial \overrightarrow{f}}{\partial x_i}\f$
    *
    * Functor to approximate the partial derivative in forward difference finite.
    */
    template<int DIM,typename TypePixelLabel>
    class POP_EXPORTS PartialDerivateForwardMultiPhaseField
    {
    private:
        const MatN<DIM,TypePixelLabel> * _label;
    public:

        void setLabelPhase(const MatN<DIM,TypePixelLabel> & label){
            _label = & label;
        }
        /*!
        * \brief partial derivate for scalar/vectoriel field
        * \param  f input scalar/vectoriel field
        * \param  x input VecN
        * \param  coordinate derivative direction
        * \return a scalar value
        *
        * The functor returns the partial derivative \f$\frac{\partial f}{\partial x_i}\f$ in centered difference finite.\n
        * For a VecN falling outside the field domain, the partial derivative is NULL (Neumann boundary condition).
        * \sa PartialDerivateForward
        */
        template<typename TypePixel>
        inline TypePixel operator()(const MatN<DIM,TypePixel> &f, const typename MatN<DIM,TypePixel>::E & x,int coordinate)
        {
            TypePixelLabel label = _label->operator()(x);
            TypePixel y=-f(x);
            x(coordinate)++;
            if( x(coordinate)<f.getDomain()(coordinate)){
                if(label==_label->operator()(x))
                    y+=f(x);
                else
                    y-=f(x);
                x(coordinate)--;
                return y;
            }else{
                x(coordinate)--;
                return TypePixel(0);
            }

        }
    };
    /*!
    \class pop::FunctorPDE::PartialDerivateBackwardMultiPhaseField
    \brief Partial derivative in backward difference finite  \f$\frac{\partial f}{\partial x_i}\f$ and \f$\frac{\partial \overrightarrow{f}}{\partial x_i}\f$
    *
    * Functor to approximate the partial derivative in backward difference finite
    * For a VecN falling outside the bulk \f$\{x:bulk(x)\neq 0\} \f$, the partial derivative is NULL (Neumann boundary condition).
    */
    template<int DIM,typename TypePixelLabel>
    class POP_EXPORTS PartialDerivateBackwardMultiPhaseField
    {
    private:
        const MatN<DIM,TypePixelLabel> * _label;
    public:
        void setLabelPhase(const MatN<DIM,TypePixelLabel> & label){
            _label = & label;
        }
        /*!
        * \brief partial derivate for scalar/vectoriel field
        * \param  f input scalar/vectoriel field
        * \param  x input VecN
        * \param  coordinate derivative direction
        * \return a scalar value
        *
        * The functor returns the partial derivative \f$\frac{\partial f}{\partial x_i}\f$.
        * \sa PartialDerivateBackward
        */
        template<typename TypePixel>
        inline TypePixel operator()(const MatN<DIM,TypePixel> &f, const typename MatN<DIM,TypePixel>::E & x,int coordinate)
        {
            TypePixelLabel label = _label->operator()(x);
            TypePixel y=f(x);
            x(coordinate)--;
            if( x(coordinate)>=0){
                if(label==_label->operator()(x))
                    y-=f(x);
                else
                    y+=f(x);
                x(coordinate)++;
                return y;
            }else{
                x(coordinate)++;
                return TypePixel(0);
            }

        }
    };
    /*!
    \class pop::FunctorPDE::PartialDerivateSecondCenteredMultiPhaseField
    \brief second partial derivative in centered difference finite  \f$\frac{\partial^2 f}{\partial x_i\partial x_j}\f$ and \f$\frac{\partial^2 \overrightarrow{f}}{\partial x_i\partial x_j}\f$
    *
    * Functor to approximate the second partial derivative.
    *
    *
    */
    template<int DIM,typename TypePixelLabel>
    class POP_EXPORTS PartialDerivateSecondCenteredMultiPhaseField
    {
    private:

        const MatN<DIM,TypePixelLabel> * _label;
    public:
        void setLabelPhase(const MatN<DIM,TypePixelLabel> & label){
            _label = & label;
            partiatforward.setLabelPhase(label);
            partiatbackward.setLabelPhase(label);
            partiatcentered.setLabelPhase(label);
        }
        PartialDerivateForwardMultiPhaseField<DIM,TypePixelLabel > partiatforward;
        PartialDerivateBackwardMultiPhaseField<DIM,TypePixelLabel > partiatbackward;
        PartialDerivateCenteredMultiPhaseField<DIM,TypePixelLabel > partiatcentered;
        /*!
        * \brief partial derivate for scalar/vectoriel field
        * \param  f input scalar/vectoriel field
        * \param  x input VecN
        * \param  coordinate_i first derivative direction
        * \param  coordinate_j second derivative direction
        * \return a scalar value
        *
        * The functor returns the second partial derivative \f$\frac{\partial^2 f}{\partial x_j \partial x_i }\f$ in centered difference finite that is in 2d  equal
        *   - \f$f(i+1,j)+f(i-1,j)-2*f(i,j)\f$ for coordinate_i=coordinate_j=0
        *   - \f$(f(i+1,j+1)-f(i-1,j+1))-(f(i+1,j-1)-f(i-1,j-1))\f$ for coordinate_i=0 and coordinate_j=1
        */
        template<typename TypePixel>
        inline TypePixel operator()(const MatN<DIM,TypePixel> &f, const typename MatN<DIM,TypePixel>::E & x,int coordinate_i,int coordinate_j )
        {
            if(coordinate_i==coordinate_j){
                return partiatforward(f,x,coordinate_i)-partiatbackward(f,x,coordinate_i);
            }
            else{

                TypePixel y(0);
                x(coordinate_j)++;
                if( x(coordinate_j)<f.getDomain()(coordinate_j)){
                    y=partiatcentered(f,x,coordinate_i);
                }
                x(coordinate_j)-=2;
                if( x(coordinate_j)>=0){
                    y-=partiatcentered(f,x,coordinate_i);
                }
                x(coordinate_j)++;
                return 0;
            }
        }
    };
    /*!
    \class pop::FunctorPDE::PartialDerivateCenteredInBulkMultiPhaseField
    \brief Partial derivative in centered difference finite  \f$\frac{\partial f}{\partial x_i}\f$ and \f$\frac{\partial \overrightarrow{f}}{\partial x_i}\f$
    *
    * Functor to approximate the partial derivative in centered difference finite for a scalar/vectoriel field in multi-phase field formalism. For a VecN falling outside the bulk \f$\{x:bulk(x)\neq 0\} \f$, the partial derivative is NULL (Neumann boundary condition).
    * \sa PartialDerivateCentered
    */
    template<int DIM,typename TypePixelLabel>
    class POP_EXPORTS PartialDerivateCenteredInBulkMultiPhaseField
    {
    private:
        const MatN<DIM,UI8> * _bulk;
        const MatN<DIM,TypePixelLabel> * _label;
    public:
        void setBulk(const MatN<DIM,UI8> & bulk){
            _bulk = & bulk;
        }
        void setLabelPhase(const MatN<DIM,TypePixelLabel> & label){
            _label = & label;
        }
        /*!
        * \brief partial derivate for scalar/vectoriel field
        * \param  f input scalar/vectoriel field
        * \param  x input VecN
        * \param  coordinate derivative direction
        * \return a scalar value
        *
        * The functor returns the partial derivative \f$\frac{\partial f}{\partial x_i}\f$ in centered difference finite that is in 2d  equal \f$(f(i+1,j)-f(i-1,j))/2\f$ for coordinate=0 and x=(i,j).
        *
        */
        template<typename TypePixel>
        inline TypePixel operator()(const MatN<DIM,TypePixel> &f, const typename MatN<DIM,TypePixel>::E & x,int coordinate)
        {
            TypePixelLabel label = _label->operator()(x);
            x(coordinate)++;
            TypePixel y;
            if( x(coordinate)<f.getDomain()(coordinate)&&_bulk->operator()(x)!=0){
                if(label==_label->operator()(x))
                    y= f(x);
                else
                    y=-f(x);
                x(coordinate)-=2;
                if(x(coordinate)>=0&&_bulk->operator()(x)!=0)
                {
                    if(label==_label->operator()(x))
                        y-=f(x);
                    else
                        y+=f(x);
                    y/=2;
                    x(coordinate)++;
                    return y;
                }
                else{
                    x(coordinate)++;
                    return TypePixel(0);
                }
            }else{
                x(coordinate)--;
                return TypePixel(0);
            }

        }
    };
    /*!
    \class pop::FunctorPDE::PartialDerivateForwardInBulkMultiPhaseField
    \brief Partial derivative in forward difference finite  \f$\frac{\partial f}{\partial x_i}\f$ and \f$\frac{\partial \overrightarrow{f}}{\partial x_i}\f$
    *
    * Functor to approximate the partial derivative in forward difference finite. For a VecN falling outside the bulk \f$\{x:bulk(x)\neq 0\} \f$, the partial derivative is NULL (Neumann boundary condition).
    */
    template<int DIM,typename TypePixelLabel>
    class POP_EXPORTS PartialDerivateForwardInBulkMultiPhaseField
    {
    private:
        const MatN<DIM,UI8> * _bulk;
        const MatN<DIM,TypePixelLabel> * _label;
    public:
        void setBulk(const MatN<DIM,UI8> & bulk){
            _bulk = & bulk;
        }
        void setLabelPhase(const MatN<DIM,TypePixelLabel> & label){
            _label = & label;
        }
        /*!
        * \brief partial derivate for scalar/vectoriel field
        * \param  f input scalar/vectoriel field
        * \param  x input VecN
        * \param  coordinate derivative direction
        * \return a scalar value
        *
        * The functor returns the partial derivative \f$\frac{\partial f}{\partial x_i}\f$ in centered difference finite.\n
        * For a VecN falling outside the field domain, the partial derivative is NULL (Neumann boundary condition).
        * \sa PartialDerivateForward
        */
        template<typename TypePixel>
        inline TypePixel operator()(const MatN<DIM,TypePixel> &f, const typename MatN<DIM,TypePixel>::E & x,int coordinate)
        {
            TypePixelLabel label = _label->operator()(x);
            TypePixel y=-f(x);
            x(coordinate)++;
            if( x(coordinate)<f.getDomain()(coordinate) &&_bulk->operator()(x)!=0){
                if(label==_label->operator()(x))
                    y+=f(x);
                else
                    y-=f(x);
                x(coordinate)--;
                return y;
            }else{
                x(coordinate)--;
                return TypePixel(0);
            }

        }
    };
    /*!
    \class pop::FunctorPDE::PartialDerivateBackwardInBulkMultiPhaseField
    \brief Partial derivative in backward difference finite  \f$\frac{\partial f}{\partial x_i}\f$ and \f$\frac{\partial \overrightarrow{f}}{\partial x_i}\f$
    *
    * Functor to approximate the partial derivative in backward difference finite
    * For a VecN falling outside the bulk \f$\{x:bulk(x)\neq 0\} \f$, the partial derivative is NULL (Neumann boundary condition).
    */
    template<int DIM,typename TypePixelLabel>
    class POP_EXPORTS PartialDerivateBackwardInBulkMultiPhaseField
    {
    private:
        const MatN<DIM,UI8> * _bulk;
        const MatN<DIM,TypePixelLabel> * _label;
    public:
        void setBulk(const MatN<DIM,UI8> & bulk){
            _bulk = & bulk;
        }
        void setLabelPhase(const MatN<DIM,TypePixelLabel> & label){
            _label = & label;
        }
        /*!
        * \brief partial derivate for scalar/vectoriel field
        * \param  f input scalar/vectoriel field
        * \param  x input VecN
        * \param  coordinate derivative direction
        * \return a scalar value
        *
        * The functor returns the partial derivative \f$\frac{\partial f}{\partial x_i}\f$.
        * \sa PartialDerivateBackward
        */
        template<typename TypePixel>
        inline TypePixel operator()(const MatN<DIM,TypePixel> &f, const typename MatN<DIM,TypePixel>::E & x,int coordinate)
        {
            TypePixelLabel label = _label->operator()(x);
            TypePixel y=f(x);
            x(coordinate)--;
            if( x(coordinate)>=0&&_bulk->operator()(x)!=0){
                if(label==_label->operator()(x))
                    y-=f(x);
                else
                    y+=f(x);
                x(coordinate)++;
                return y;
            }else{
                x(coordinate)++;
                return TypePixel(0);
            }

        }
    };
    /*!
    \class pop::FunctorPDE::PartialDerivateSecondCenteredInBulkMultiPhaseField
    \brief second partial derivative in centered difference finite  \f$\frac{\partial^2 f}{\partial x_i\partial x_j}\f$ and \f$\frac{\partial^2 \overrightarrow{f}}{\partial x_i\partial x_j}\f$
    *
    * Functor to approximate the second partial derivative. For a VecN falling outside the bulk \f$\{x:bulk(x)\neq 0\} \f$, the partial derivative is NULL (Neumann boundary condition).
    *
    *
    */
    template<int DIM,typename TypePixelLabel>
    class POP_EXPORTS PartialDerivateSecondCenteredInBulkMultiPhaseField
    {
    private:
        const MatN<DIM,UI8> * _bulk;
        const MatN<DIM,TypePixelLabel> * _label;
    public:
        void setLabelPhase(const MatN<DIM,TypePixelLabel> & label){
            _label = & label;
            partiatforward.setLabelPhase(label);
            partiatbackward.setLabelPhase(label);
            partiatcentered.setLabelPhase(label);
        }
        void setBulk(const MatN<DIM,UI8> & bulk){
            _bulk = & bulk;
            partiatforward.setBulk(bulk);
            partiatbackward.setBulk(bulk);
            partiatcentered.setBulk(bulk);
        }
        PartialDerivateForwardInBulkMultiPhaseField<DIM,TypePixelLabel> partiatforward;
        PartialDerivateBackwardInBulkMultiPhaseField<DIM,TypePixelLabel> partiatbackward;
        PartialDerivateCenteredInBulkMultiPhaseField<DIM,TypePixelLabel> partiatcentered;
        /*!
        * \brief partial derivate for scalar/vectoriel field
        * \param  f input scalar/vectoriel field
        * \param  x input VecN
        * \param  coordinate_i first derivative direction
        * \param  coordinate_j second derivative direction
        * \return a scalar value
        *
        * The functor returns the second partial derivative \f$\frac{\partial^2 f}{\partial x_j \partial x_i }\f$ in centered difference finite that is in 2d  equal
        *   - \f$f(i+1,j)+f(i-1,j)-2*f(i,j)\f$ for coordinate_i=coordinate_j=0
        *   - \f$(f(i+1,j+1)-f(i-1,j+1))-(f(i+1,j-1)-f(i-1,j-1))\f$ for coordinate_i=0 and coordinate_j=1
        */
        template<typename TypePixel>
        inline TypePixel operator()(const MatN<DIM,TypePixel> &f, const typename MatN<DIM,TypePixel>::E & x,int coordinate_i,int coordinate_j )
        {
            if(coordinate_i==coordinate_j){
                return partiatforward(f,x,coordinate_i)-partiatbackward(f,x,coordinate_i);
            }
            else{
                TypePixel y(0);
                x(coordinate_j)++;
                if( x(coordinate_j)<f.getDomain()(coordinate_j)&&_bulk->operator()(x)!=0){
                    y=partiatcentered(f,x,coordinate_i);
                }
                x(coordinate_j)-=2;
                if( x(coordinate_j)>=0&&_bulk->operator()(x)!=0){
                    y-=partiatcentered(f,x,coordinate_i);
                }
                x(coordinate_j)++;
                return 0;
            }
        }
    };


    //@}
    //-------------------------------------
    //
    //! \name y=F(f,x) with f a field, x a VecN and y a scalar/vectoriel value
    //@{
    //-------------------------------------

    /*!
    \class pop::FunctorPDE::Gradient
    \brief Gradient of a scalar/vectoriel field \f$\overrightarrow{\nabla} f\f$
    * \tparam PartialDerivate  partial derivate
    *
    */
    template<typename PartialDerivate=PartialDerivateCentered >
    class POP_EXPORTS Gradient
    {
    public:
        PartialDerivate partial;
        /*!
        * \param  f input scalar field
        * \param  x input VecN
        * \return a vectoriel value
        *
        * The functor returns the vector \f$\overrightarrow{\nabla} f\f$ that is to \f$\begin{pmatrix}\frac{\partial f}{\partial x_0}\\\frac{\partial f}{\partial x_1}\\\vdots \end{pmatrix}\f$
        * with the partial derivate given by the template paramater
        */
        template<int DIM,typename TypePixel>
        inline VecN<DIM, TypePixel> operator()(const MatN<DIM,TypePixel> & f,  const typename MatN<DIM,TypePixel>::E & x)
        {
            VecN<DIM, TypePixel> grad;
            for(int i = 0;i<DIM;i++)
                grad(i)=partial(f,x,i);
            return grad;
        }

    };
    /*!
    \class pop::FunctorPDE::Divergence
    \brief  Divergence of the vertoriel field \f$\overrightarrow{\nabla}\cdot \overrightarrow{f}\f$
    *

    *
    */
    template<typename PartialDerivate=PartialDerivateCentered >
    class POP_EXPORTS Divergence
    {
    public:
        PartialDerivate partial;
        /*!
        * \param  f input vectoriel field
        * \param  x input VecN
        * \return a scalar value
        *
        * The functor returns the scalar value \f$\overrightarrow{\nabla}\cdot \overrightarrow{f}\f$ that is \f$\sum_{i=0}^{n-1} \frac{\partial f_i}{\partial x_i}\f$
        * with the partial derivate given by the template paramater
        *
        */
        template<int DIM,typename TypePixel>
        TypePixel operator()(const MatN<DIM,VecN<DIM,TypePixel> > & f,  const typename MatN<DIM,TypePixel>::E & x)
        {
            TypePixel divergence(0);
            for(int i = 0;i<DIM;i++){
                VecN<DIM,TypePixel> div=partial(f,x,i);
                divergence+=div(i);
            }
            return divergence;
        }
    };
    /*!
    \class pop::FunctorPDE::Curl
    \brief  Curl of the 3d vectoriel field \f$\overrightarrow{\nabla}\times \overrightarrow{f}\f$
    *
    *
    */
    template<typename PartialDerivate=PartialDerivateCentered >
    class POP_EXPORTS Curl
    {
    public:
        PartialDerivate partial;
        /*!
        * \param  f input vectoriel field
        * \param  x input VecN
        * \return a vectoriel value
        *
        * The functor returns the scalar value \f$\overrightarrow{\nabla}\times \overrightarrow{f}\f$ that is \f$\begin{pmatrix}\frac{\partial f_2}{\partial x_1}-\frac{\partial f_1}{\partial x_2} \\\frac{\partial f_0}{\partial x_2}-\frac{\partial f_2}{\partial x_0} \\ \frac{\partial f_1}{\partial x_0}-\frac{\partial f_0}{\partial x_1}   \end{pmatrix}\f$
        * with the partial derivate given by the template paramater
        *
        */
        template<int DIM,typename TypePixel>
        TypePixel operator()(const MatN<DIM,TypePixel> & f,  const typename MatN<DIM,TypePixel>::E & x)
        {
            TypePixel curl;
            TypePixel divx = partial(f,x,0);
            TypePixel divy = partial(f,x,1);
            TypePixel divz = partial(f,x,2);
            curl(0) = divy(2)-divz(1);
            curl(1) = divz(0)-divx(2);
            curl(2) = divx(1)-divy(0);
            return curl;
        }
    };
    /*!
    \class pop::FunctorPDE::Laplacien
    \brief  Laplacien of a scalar/vectoriel field \f$\overrightarrow{\nabla} \cdot (\overrightarrow{\nabla} f)\f$ or\f$\overrightarrow{\nabla} \cdot (\overrightarrow{\nabla} \overrightarrow{f})\f$
    *
    *
    */
    template<typename PartialDerivateSecond=PartialDerivateSecondCentered>
    class POP_EXPORTS Laplacien
    {
    public:
        PartialDerivateSecond partialsecond;

        /*!
        * \param  f input scalar/vectoriel field
        * \param  x input VecN
        * \return a scalar/vectoriel value
        *
        * The functor returns the laplacien value \f$\overrightarrow{\nabla} \cdot (\overrightarrow{\nabla} f)=\sum_i \frac{\partial^2 f}{\partial x^2} \f$
        * with the partial second derivate given by the template paramater
        *
        */
        template<int DIM,typename TypePixel>
        TypePixel operator()(const MatN<DIM,TypePixel> & f,  const typename MatN<DIM,TypePixel>::E & x){
            TypePixel laplacien(0);
            for(int i = 0;i<DIM;i++){
                laplacien+=partialsecond(f,x,i,i);
            }
            return laplacien;
        }

    };
    /*!
    \class pop::FunctorPDE::HessianMatrix
    \brief  Hessian matrix of a scalar field
    *
\f$H(f) = \begin{bmatrix}
\dfrac{\partial^2 f}{\partial x_1^2} & \dfrac{\partial^2 f}{\partial x_1\,\partial x_2} & \cdots & \dfrac{\partial^2 f}{\partial x_1\,\partial x_n} \\[2.2ex]
\dfrac{\partial^2 f}{\partial x_2\,\partial x_1} & \dfrac{\partial^2 f}{\partial x_2^2} & \cdots & \dfrac{\partial^2 f}{\partial x_2\,\partial x_n} \\[2.2ex]
\vdots & \vdots & \ddots & \vdots \\[2.2ex]
\dfrac{\partial^2 f}{\partial x_n\,\partial x_1} & \dfrac{\partial^2 f}{\partial x_n\,\partial x_2} & \cdots & \dfrac{\partial^2 f}{\partial x_n^2}
\end{bmatrix}.\f$
    *
    *
    */
    template<typename PartialDerivateSecond=PartialDerivateSecondCentered>
    class POP_EXPORTS HessianMatrix
    {
    public:
        PartialDerivateSecond partialsecond;
        /*!
        * \param  f input scalar/vectoriel field
        * \param  x input VecN
        * \return a scalar/vectoriel value
        *
        * The functor returns the laplacien value \f$\overrightarrow{\nabla} \cdot (\overrightarrow{\nabla} f)=\sum_i \frac{\partial^2 f}{\partial x^2} \f$
        * with the partial second derivate given by the template paramater
        *
        */
        template<int DIM,typename TypePixel>
        inline Mat2x<F32,DIM,DIM> operator()(const MatN<DIM,TypePixel> & f, const typename MatN<DIM,TypePixel>::E & x){
            typedef Mat2x<F32,DIM,DIM> Mat2x22f;
            Mat2x22f m;
            for(int i = 0;i<DIM;i++){
                for(int j = 0;j<DIM;j++){
                    if(j<i)
                        m(i,j)=m(j,i);
                    else
                        m(i,j)= partialsecond.operator()(f,x,i,j);
                }
            }
            return m;
        }

    };
    //@}

    //@}
    //-------------------------------------
    //
    //! \name Composed functor
    //@{
    //-------------------------------------

    /*!
    \class pop::FunctorPDE::DiffusionMalikPerona
    \brief  Malika Peronna \f$ D( \overrightarrow{\nabla} f )= c(\| \overrightarrow{\nabla} f *G_\sigma  \| )\overrightarrow{\nabla} f \f$ with \f$c(x)= \frac{1}{1 + \left(\frac{x}{K}\right)^2}  \f$
    * where \f$G_\sigma\f$ the centered gaussian(normal) kernel with \f$\sigma\f$ the standard deviation as scale paramete
    *
    *
    */


    class POP_EXPORTS DiffusionMalikPerona
    {
    private:

        F32 _kpower2;
        template<typename TypePixel>
        inline F32 f(TypePixel v) {
            return 0.5 /(1 + pop::productInner(v,v)/(_kpower2));
        }
        PartialDerivateBackward derivate_backward;
        PartialDerivateForward derivate_forward;
    public:

        /*!
        * \param K the constant in the monotonically decreasing function c
        *
        * Construct the functor
        *
        */
        DiffusionMalikPerona(F32 K)
            :_kpower2(K*K/(80))
        {
        }
        DiffusionMalikPerona(const DiffusionMalikPerona& k)
            :_kpower2(k._kpower2)
        {
            //        std::cout<<_kpower2<<std::endl;
        }


        template<int DIM,typename TypePixel>
        TypePixel operator()(const MatN<DIM,TypePixel > &field,const  typename MatN<DIM,TypePixel>::E & x){
            TypePixel flow=0;
            for(unsigned int i=0;i<DIM;i++){
                TypePixel diff_backward=-derivate_backward(field,x,i);
                TypePixel diff_forward=derivate_forward(field,x,i);
                F32 c_backward = f(diff_backward);
                F32 c_forward = f(diff_forward);
                flow+=(c_backward*diff_backward+c_forward*diff_forward);
            }
            return field(x)+flow*(0.5/DIM);
        }
    };

    //@}

    /*!
    \class pop::FunctorPDE::FreeEnergy
    \brief  the opposite of the derivative of the F32 well potential \f$−\phi^2/2 + \phi^4/4\f$,
    *
    *
    */
    class POP_EXPORTS  FreeEnergy
    {
    private:
        /*!
        * \param x input value
        * \return \f$\phi(1-\phi^2)\f$,
        *
        * Construct the functor
        *
        */
        template<typename Type>
        Type operator()(Type x){
            return x*(1-x*x);
        }
        void init(){}
    };
};
/// @endcond
}
#endif // FUNCTOTPDE_H
