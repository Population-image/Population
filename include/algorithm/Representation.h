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
#ifndef REPRESENTATION_H
#define REPRESENTATION_H
#include"data/utility/Exception.h"
#include"data/utility/CollectorExecutionInformation.h"
#include"data/functor/FunctorF.h"
#include"algorithm/FunctionProcedureFunctorF.h"
#include"algorithm/Visualization.h"
#include"algorithm/GeometricalTransformation.h"
namespace pop
{

/*!
\defgroup Representation Representation
\ingroup Algorithm
\brief Matrix In -> Matrix Out (FFT)



In general, an matrix is a space-domain, a paving of a domain of the Euclidean space containing an information in each cell.
This space-domain representation is usually the direct matrix obtained by a microscopy. However, there is different kinds of
representations of the information (see \ref DOCRepresentation).
The Fourier transform allows the correspondence between the space-domain and the frequency-domain. The wavelet transform allows the correspondence between the space-domain and the scale-space-domain.
The choice of the representation depends on the particular problem.
For instance, the noise can be seen as fluctuation with a short length correlation. In the Fourier space, this noise corresponds to high frequency and it is removed easily with a low pass.
\code
Mat2UI8 img;//2d grey-level image object
img.load("../image/Lena.bmp");//replace this path by those on your computer
Mat2F64 noisemap;
Distribution d(DistributionNormal(0,20));
Processing::randomField(img.getDomain(),d,noisemap);
img = Mat2F64(img) +noisemap;
img.save("../doc/image2/lenanoise.jpg");
Mat2ComplexF64 imgcomplex;
Convertor::fromRealImaginary(img,imgcomplex);
Mat2ComplexF64 fft = Representation::FFT(imgcomplex,1);
Mat2UI8 filterlowpass = Representation::lowPass(fft,60);
imgcomplex = Representation::FFT(filterlowpass,-1);
Mat2F64 imgd;
Convertor::toRealImaginary(imgcomplex,imgd);
img = Processing::greylevelRange(imgd,0,255);
img.display();
img.save("../doc/image2/lenalowpass.jpg");
\endcode
\image html lenanoise.jpg "noisy image"
\image html lenalowpass.jpg "filter image with low pass"
*/

struct POP_EXPORTS Representation
{

    /*!
        \class pop::Representation
        \ingroup Representation
        \brief Working with representation
        \author Tariel Vincent
        *

        *
        * Nowadays, Population library includes only the gate between the direct space and the fourier space.

    */

    //-------------------------------------
    //
    //! \name FFT
    //@{
    //-------------------------------------

    /*!
         * \param in input matrix with ComplexF64 as pixel/voxel type
         * \param direction for direction =1 direct FFT, otherwise inverse FFT
         * \return return the fft
        * \brief Apply the FFT on the input matrix

        In this example, we apply a low pass filter:
         \code
        Mat2UI8 img;//2d grey-level image object
        img.load("/home/vincent/Dropbox/MyArticle/PhaseField/lena.pgm");//replace this path by those on your computer
        Mat2ComplexF64 imgcomplex;
        Convertor::fromRealImaginary(img,imgcomplex);
        Mat2ComplexF64 fft = Representation::FFT(imgcomplex);
        Mat2ComplexF64 filterlowpass(fft.getDomain());
        Vec2I32 x(0,0);
        Draw::disk(filterlowpass,x,10,NumericLimits<ComplexF64>::maximumRange(),MATN_BOUNDARY_CONDITION_MIRROR);
        fft = Processing::mask(fft,filterlowpass);
        imgcomplex = Representation::FFT(fft,-1);
        Mat2F64 imgd;
        Convertor::toRealImaginary(imgcomplex,imgd);
        img = Processing::greylevelRange(imgd,0,255);
        img.display();
        \endcode
         *
        */
    template<int DIM>
    static MatN<DIM,ComplexF64>  FFT(const MatN<DIM,ComplexF64> & f ,int direction=1) throw(pexception)
    {
        MatN<DIM,ComplexF64> in;
        if(isPowerOfTwo(f.getDomain()(0))==false||isPowerOfTwo(f.getDomain()(1))==false){
            in =truncateMulitple2(f);
        }else{
            in =f;
        }
        CollectorExecutionInformationSingleton::getInstance()->startExecution("FunctionProcedureFFT",COLLECTOR_EXECUTION_RATIO);

        typename MatN<DIM-1,ComplexF64>::E x;
        MatN<DIM,ComplexF64>  F (in);

        for(int fixed_coordinate=0;fixed_coordinate<DIM;fixed_coordinate++){
            //        for(int fixed_coordinate=Function::DIM-;fixed_coordinate>=0;fixed_coordinate--)
            //        int fixed_coordinate=2;
            //        {
            CollectorExecutionInformationSingleton::getInstance()->info("Direction"+BasicUtility::Any2String(fixed_coordinate));
            x = in.getDomain().removeCoordinate(fixed_coordinate);
            typename MatN<DIM-1,ComplexF64>::IteratorEDomain it(x);
            MatN<1,ComplexF64> lign(in.getDomain()(fixed_coordinate));
            MatN<1,ComplexF64> lign2(in.getDomain()(fixed_coordinate));
            while(it.next()){
                typename MatN<DIM,ComplexF64>::E x;
                for(int i=0;i<DIM;i++){
                    if(i<fixed_coordinate)
                        x(i) =it.x()(i);
                    else if(i>fixed_coordinate)
                        x(i) =it.x()(i-1);
                }
                //                std::cout<<x<<std::endl;
                for(x(fixed_coordinate)=0;x(fixed_coordinate)<in.getDomain()(fixed_coordinate);x(fixed_coordinate)++){
                    lign(x(fixed_coordinate))=F(x);
                }
                lign2 = Representation::FFTOneDimension(lign,direction);
                for(x(fixed_coordinate)=0;x(fixed_coordinate)<in.getDomain()(fixed_coordinate);x(fixed_coordinate)++){
                    F(x)=lign2(x(fixed_coordinate));
                }


            }
        }
        typename MatN<DIM,ComplexF64>::IteratorEDomain b(F.getDomain());
        if(direction!=1)
        {
            while(b.next())
            {
                int mult = F.getDomain().multCoordinate();
                (F)(b.x())*= mult;
            }
        }
        return F;
    }

    /*! \brief visualization of the fourrier matrix in log scale h(x) = log( ||f(x)||+1)
         * \param fft input FFT matrix
         * \return grey level matrix
        *
         \code
    Mat2UI8 img;
    img.load("../image/eutel.bmp");
    Mat2ComplexF64 imgc;
    imgc.fromRealImaginary(img);
    imgc = Representation::FFT(imgc,1);
    img = Representation::FFTDisplay(imgc);
    img.display();
    \endcode
        */
    template<int DIM>
    static MatN<DIM,UI8> FFTDisplay(const MatN<DIM,ComplexF64> & fft)
    {


        MatN<DIM,F64> imgf(fft.getDomain());
        MatN<DIM,UI8> img(fft.getDomain());
        typename MatN<DIM,ComplexF64>::IteratorEDomain it(img.getIteratorEDomain());
        while(it.next()){
            double v = std::log(normValue(fft(it.x()))+1)/std::log(10.);
            imgf(it.x())=v;
        }
        FunctorF::FunctorAccumulatorMin<F32 > funcmini;
        it.init();
        F64 mini = FunctionProcedureFunctorAccumulatorF(imgf,funcmini,it);
        FunctorF::FunctorAccumulatorMax<F64 > funcmaxi;
        it.init();
        F64 maxi = FunctionProcedureFunctorAccumulatorF(imgf,funcmaxi,it);
        it.init();
        while(it.next())
            img(it.x()) = ((imgf(it.x())-mini)*255/(maxi-mini));

        MatN<DIM,UI8> out(fft.getDomain());
		it.init();
        while(it.next()){
            typename MatN<DIM,UI8>::E xtran = it.x()+ img.getDomain()/2;
            MatNBoundaryConditionPeriodic::apply(img.getDomain(),xtran);
            out(xtran)=img(it.x());

        }
        return out;
    }

    //@}
    //-------------------------------------
    //
    //! \name Filter in Fourier space
    //@{
    //-------------------------------------

    /*! \brief low pass filter (frequency higher of the threshold are removed)
         * \param f input fourier matrix
         * \param frenquency_threshold threshold
         * \return output fourier matrix
        *
     \code
    Mat2UI8 img;
    img.load("/home/vincent/Desktop/bin/Population/doc/html/lena.bmp");
    Mat2ComplexF64 imgc;
    imgc.fromRealImaginary(img);
    imgc = Representation::FFT(imgc,1);
    imgc = Representation::lowPass(imgc,7);
    imgc = Representation::FFT(imgc,0);
    imgc.toRealImaginary(img);
    img.display();
    \endcode
        */

    template<int DIM>
    static MatN<DIM,ComplexF64> lowPass(const MatN<DIM,ComplexF64> & f,double frenquency_threshold){
        MatN<DIM,ComplexF64> filter(f.getDomain());
        typename MatN<DIM,ComplexF64>::IteratorENeighborhoodPeriodic it(filter.getIteratorENeighborhood(frenquency_threshold,2));
        it.init(0);
        while(it.next()){
            filter(it.x())=f(it.x());
        }
        return filter;
    }
    /*! \brief low pass filter (frequency lower of the threshold are removed)
         * \param f input fourier matrix
         * \param frenquency_threshold threshold
         * \return output fourier matrix
        *
     \code
    Mat2UI8 img;
    img.load("/home/vincent/Desktop/bin/Population/doc/html/lena.bmp");
    Mat2ComplexF64 imgc;
    imgc.fromRealImaginary(img);
    imgc = Representation::FFT(imgc,1);
    imgc = Representation::highPass(imgc,7);
    imgc = Representation::FFT(imgc,0);
    imgc.toRealImaginary(img);
    img.display();
    \endcode
        */
    template<int DIM>
    static MatN<DIM,ComplexF64> highPass(const MatN<DIM,ComplexF64> & f,double frenquency_threshold){
        MatN<DIM,ComplexF64> filter(f);
        typename MatN<DIM,ComplexF64>::IteratorENeighborhoodPeriodic it(filter.getIteratorENeighborhood(frenquency_threshold,2));
        it.init(0);
        while(it.next()){
            filter(it.x())=0;
        }
        return filter;
    }
    //@}

    //-------------------------------------
    //
    //! \name Multiple of two
    //@{
    //-------------------------------------
    /*! \fn static Function truncateMulitple2(const Function & in)
     * \param in input matrix
     * \return output matrix with a domain multiple of 2 for each coordinate
     * \brief truncate the matrix in order to have domain multiple of 2 for each coordinate
     *
    */
    template<int DIM,typename TypePixel>
    static MatN<DIM,TypePixel> truncateMulitple2(const MatN<DIM,TypePixel> & in)
    {
        typename MatN<DIM,TypePixel>::E x;
        for(int i=0;i<DIM;i++){
            int j=1;
            while(j<=in.getDomain()(i)){
                j*=2;
            }
            x(i) = j/2;
        }
        MatN<DIM,TypePixel> out(x);
        typename MatN<DIM,TypePixel>::IteratorEDomain it       (out.getIteratorEDomain());
        while(it.next()){
            out(it.x())= in(it.x());
        }
        return out;
    }
    //@}
private:
    template<int DIM,typename TypePixel>
    static MatN<DIM,TypePixel>   FFTOneDimension(const MatN<DIM,TypePixel> & in,int direction ){
        if(in.getDomain()(0)>1){

            typename MatN<DIM,TypePixel>::E x;
            x =in.getDomain();
            x(0)/=2;
            MatN<DIM,TypePixel>  fodd (x);
            MatN<DIM,TypePixel> feven (x);

            for(int i =0; i< fodd.getDomain()(0);i++){
                (fodd)(i)= (in)(2*i);
                (feven)(i)= (in)(2*i+1);
            }
            MatN<DIM,TypePixel>  Fodd = Representation::FFTOneDimension(fodd,direction);
            MatN<DIM,TypePixel>  Feven = Representation::FFTOneDimension(feven,direction);
            MatN<DIM,TypePixel> out (in.getDomain());
            int half = in.getDomain()(0)/2;

            ComplexF64  w1,w;
            w1.real() = std::cos(2*PI/in.getDomain()(0) );
            if(direction==1)
                w1.img() = -std::sin(2*PI/in.getDomain()(0) );
            else
                w1.img() = std::sin(2*PI/in.getDomain()(0) );
            w.real()=1;
            w.img()=0;

            for(int i =0; i< Fodd.getDomain()(0);i++)
            {
                (out)(i)=1/2.0 * ((Fodd)(i) +w* (Feven)(i)) ;
                (out)(i+half)=1/2.0 * ((Fodd)(i) -w* (Feven)(i));
                w = w*w1;
            }
            return out;
        }
        else
        {
            typename MatN<DIM,TypePixel>::E x;
            x(0)=1;
            MatN<DIM,TypePixel> out (x);
            x(0)=0;
            (out) (x) = (in) (x);
            return out;
        }
    }

    template<typename T>
    static bool isPowerOfTwo (T v)
    {
    	return ((v & -v)) == v;
    }
};
}
#endif // REPRESENTATION_H
