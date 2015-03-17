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
#ifndef REPRESENTATION_H
#define REPRESENTATION_H
#include"data/functor/FunctorF.h"
#include"algorithm/ForEachFunctor.h"
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
Mat2F32 noisemap;
Distribution d(DistributionNormal(0,20));
Processing::randomField(img.getDomain(),d,noisemap);
img = Mat2F32(img) +noisemap;
img.save("../doc/image2/lenanoise.jpg");
Mat2ComplexF32 imgcomplex;
Convertor::fromRealImaginary(img,imgcomplex);
Mat2ComplexF32 fft = Representation::FFT(imgcomplex,1);
Mat2UI8 filterlowpass = Representation::lowPass(fft,60);
imgcomplex = Representation::FFT(filterlowpass,-1);
Mat2F32 imgd;
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
         * \param f input matrix with ComplexF32 as pixel/voxel type
         * \param direction for direction =1 direct FFT, otherwise inverse FFT
         * \return return the fft
        * \brief Apply the FFT on the input matrix

        In this example, we apply a low pass filter:
         \code
        Mat2UI8 img;//2d grey-level image object
        img.load("/home/vincent/Dropbox/MyArticle/PhaseField/lena.pgm");//replace this path by those on your computer
        Mat2ComplexF32 imgcomplex;
        Convertor::fromRealImaginary(img,imgcomplex);
        Mat2ComplexF32 fft = Representation::FFT(imgcomplex);
        Mat2ComplexF32 filterlowpass(fft.getDomain());
        Vec2I32 x(0,0);
        Draw::disk(filterlowpass,x,10,NumericLimits<ComplexF32>::maximumRange(),MATN_BOUNDARY_CONDITION_MIRROR);
        fft = Processing::mask(fft,filterlowpass);
        imgcomplex = Representation::FFT(fft,-1);
        Mat2F32 imgd;
        Convertor::toRealImaginary(imgcomplex,imgd);
        img = Processing::greylevelRange(imgd,0,255);
        img.display();
        \endcode
         *
        */
    template<int DIM>
    static MatN<DIM,ComplexF32>  FFT(const MatN<DIM,ComplexF32> & f ,int direction=1)
    {
        MatN<DIM,ComplexF32> in;
        if(isPowerOfTwo(f.getDomain()(0))==false||isPowerOfTwo(f.getDomain()(1))==false){
            in =truncateMulitple2(f);
        }else{
            in =f;
        }

        typename MatN<DIM-1,ComplexF32>::E x;
        MatN<DIM,ComplexF32>  F (in);

        for(int fixed_coordinate=0;fixed_coordinate<DIM;fixed_coordinate++){
            x = in.getDomain().removeCoordinate(fixed_coordinate);
            typename MatN<DIM-1,ComplexF32>::IteratorEDomain it(x);
            MatN<1,ComplexF32> lign(in.getDomain()(fixed_coordinate));
            MatN<1,ComplexF32> lign2(in.getDomain()(fixed_coordinate));
            while(it.next()){
                typename MatN<DIM,ComplexF32>::E xx;
                for(int i=0;i<DIM;i++){
                    if(i<fixed_coordinate)
                        xx(i) =it.x()(i);
                    else if(i>fixed_coordinate)
                        xx(i) =it.x()(i-1);
                }
                for(xx(fixed_coordinate)=0;xx(fixed_coordinate)<in.getDomain()(fixed_coordinate);xx(fixed_coordinate)++){
                    lign(xx(fixed_coordinate))=F(xx);
                }
                lign2 = Representation::FFT(lign,direction);
                for(xx(fixed_coordinate)=0;xx(fixed_coordinate)<in.getDomain()(fixed_coordinate);xx(fixed_coordinate)++){
                    F(xx)=lign2(xx(fixed_coordinate));
                }


            }
        }
        typename MatN<DIM,ComplexF32>::IteratorEDomain b(F.getDomain());
        if(direction!=1)
        {
            int mult = F.getDomain().multCoordinate();
            while(b.next())
            {
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
    Mat2ComplexF32 imgc;
    imgc.fromRealImaginary(img);
    imgc = Representation::FFT(imgc,1);
    img = Representation::FFTDisplay(imgc);
    img.display();
    \endcode
        */
    template<int DIM>
    static MatN<DIM,UI8> FFTDisplay(const MatN<DIM,ComplexF32> & fft)
    {


        MatN<DIM,F32> imgf(fft.getDomain());
        MatN<DIM,UI8> img(fft.getDomain());
        typename MatN<DIM,ComplexF32>::IteratorEDomain it(img.getIteratorEDomain());
        while(it.next()){
            F32 v = std::log(normValue(fft(it.x()))+1)/std::log(10.);
            imgf(it.x())=v;
        }
        FunctorF::FunctorAccumulatorMin<F32 > funcmini;
        it.init();
        F32 mini = forEachFunctorAccumulator(imgf,funcmini,it);
        FunctorF::FunctorAccumulatorMax<F32 > funcmaxi;
        it.init();
        F32 maxi = forEachFunctorAccumulator(imgf,funcmaxi,it);
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
    Mat2ComplexF32 imgc;
    imgc.fromRealImaginary(img);
    imgc = Representation::FFT(imgc,1);
    imgc = Representation::lowPass(imgc,7);
    imgc = Representation::FFT(imgc,0);
    imgc.toRealImaginary(img);
    img.display();
    \endcode
        */

    template<int DIM>
    static MatN<DIM,ComplexF32> lowPass(const MatN<DIM,ComplexF32> & f,F32 frenquency_threshold){
        MatN<DIM,ComplexF32> filter(f.getDomain());
        typename MatN<DIM,ComplexF32>::IteratorENeighborhoodPeriodic it(filter.getIteratorENeighborhood(frenquency_threshold,2));
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
    Mat2ComplexF32 imgc;
    imgc.fromRealImaginary(img);
    imgc = Representation::FFT(imgc,1);
    imgc = Representation::highPass(imgc,7);
    imgc = Representation::FFT(imgc,0);
    imgc.toRealImaginary(img);
    img.display();
    \endcode
        */
    template<int DIM>
    static MatN<DIM,ComplexF32> highPass(const MatN<DIM,ComplexF32> & f,F32 frenquency_threshold){
        MatN<DIM,ComplexF32> filter(f);
        typename MatN<DIM,ComplexF32>::IteratorENeighborhoodPeriodic it(filter.getIteratorENeighborhood(frenquency_threshold,2));
        it.init(0);
        while(it.next()){
            filter(it.x())=0;
        }
        return filter;
    }
    /*!
     * \param f input matrix
     * \return output matrix with a float as pixel/voxel type
     *
     *  calculated the 2-VecNd correlation function in any direction by FFT  P = FFT^(-1)(FFT(f)FFT(f)^*)
    */

    template<int DIM,typename PixelType>
    static MatN<DIM,F32> correlationDirectionByFFT(const MatN<DIM,PixelType> & f){

        MatN<DIM,PixelType> bint;
        bint = pop::Representation::truncateMulitple2(f);
        MatN<DIM,F32> binfloat(bint);
        typename MatN<DIM,PixelType>::IteratorEDomain it (binfloat.getIteratorEDomain());
        binfloat = pop::ProcessingAdvanced::greylevelRange(binfloat,it,0,1);


        MatN<DIM,ComplexF32>  bin_complex(bint.getDomain());
        Convertor::fromRealImaginary(binfloat,bin_complex);
        MatN<DIM,ComplexF32>  fft = pop::Representation::FFT(bin_complex,1);

        it.init();
        while(it.next()){
            ComplexF32 c = fft(it.x());
            ComplexF32 c1 = fft(it.x());
            fft(it.x()).real() = (c*c1.conjugate()).real();
            fft(it.x()).img() =0;
        }
        fft  = pop::Representation::FFT(fft,0);
        MatN<DIM,F32>  fout(bint.getDomain());
        Convertor::toRealImaginary(fft,fout);
        return  fout;

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
    template<int DIM,typename PixelType>
    static MatN<DIM,PixelType> truncateMulitple2(const MatN<DIM,PixelType> & in)
    {
        typename MatN<DIM,PixelType>::E x;
        for(int i=0;i<DIM;i++){
            int j=1;
            while(j<=in.getDomain()(i)){
                j*=2;
            }
            x(i) = j/2;
        }
        MatN<DIM,PixelType> out(x);
        typename MatN<DIM,PixelType>::IteratorEDomain it       (out.getIteratorEDomain());
        while(it.next()){
            out(it.x())= in(it.x());
        }
        return out;
    }
    //@}
    static inline MatN<1,ComplexF32>  FFT(const MatN<1,ComplexF32> & in ,int direction=1){
        if(isPowerOfTwo(in.size())==false){
            std::cerr<<"Is no power of 2"<<std::endl;
        }

        if(in.size()>1){

            MatN<1,ComplexF32> fodd (in.size()/2);
            MatN<1,ComplexF32> feven (in.size()/2);

            for(size_t i =0; i< fodd.size();i++){
                (fodd)(i)= (in)(2*i);
                (feven)(i)= (in)(2*i+1);
            }
            MatN<1,ComplexF32>  Fodd = Representation::FFT(fodd,direction);
            MatN<1,ComplexF32>  Feven = Representation::FFT(feven,direction);
            MatN<1,ComplexF32> out (in.getDomain());
            int half = in.size()/2;

            ComplexF32  w1,w;
            w1.real() = std::cos(2*PI/in.size() );
            if(direction==1)
                w1.img() = -std::sin(2*PI/in.size() );
            else
                w1.img() = std::sin(2*PI/in.size() );
            w.real()=1;
            w.img()=0;

            for(int i =0; i< Fodd.getDomain()(0);i++)
            {
                (out)(i)=0.5f * ((Fodd)(i) +w* (Feven)(i)) ;
                (out)(i+half)=0.5f * ((Fodd)(i) -w* (Feven)(i));
                w = w*w1;
            }
            return out;
        }
        else
        {
            return in;
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
