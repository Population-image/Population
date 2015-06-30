#ifndef CONVERTOR_H
#define CONVERTOR_H
#include"data/utility/BasicUtility.h"
#include"data/functor/FunctorF.h"
#include"data/vec/VecN.h"
#include"data/mat/MatN.h"
namespace pop
{
/*!
\defgroup Convertor Convertor
\ingroup Algorithm
\brief Matrix In -> Matrix Out (toRGB, fromRGB, toYUV, fromRealImaginary)

Conversion: vectoriel image (vector field, color image,complex image) <-> list of scalar images
\code
Mat2RGBUI8 img;//2d grey-level image object
img.load((std::string(POP_PROJECT_SOURCE_DIR)+"/image/Lena.bmp").c_str());//replace this path by those on your computer
Mat2UI8 r,g,b;
Convertor::toRGB(img,r,g,b);
r.display();
g.display();
b.display();
\endcode
\image html lenared.jpg "red channel"
\image html lenagreen.jpg "green channel"
\image html lenablue.jpg "blue channel"
*/
struct POP_EXPORTS Convertor
{

    /*!
        \class pop::Convertor
        \ingroup Convertor
        \brief convertor facilities
        \author Tariel Vincent
    */
    template<int DIM,typename TypeRGBChannel>
    static  void toHSL(const pop::MatN<DIM,RGB<TypeRGBChannel> > & fRGB, MatN<DIM,TypeRGBChannel>& img_H, MatN<DIM,TypeRGBChannel> & img_S, MatN<DIM,TypeRGBChannel> & img_L)
    {
        img_H.resize(fRGB.getDomain());
        img_S.resize(fRGB.getDomain());
        img_L.resize(fRGB.getDomain());
        typename MatN<DIM,TypeRGBChannel>::iterator ith = img_H.begin();
        typename MatN<DIM,TypeRGBChannel>::iterator itl = img_S.begin();
        typename MatN<DIM,TypeRGBChannel>::iterator its = img_L.begin();
        typename MatN<DIM,RGB<TypeRGBChannel> >::const_iterator it = fRGB.begin();
        typename MatN<DIM,RGB<TypeRGBChannel> >::const_iterator itend = fRGB.end();
        while (it != itend) {
            it->toHSL( *ith, *itl, *its);
            ith++; itl++; its++;it++;
        }
    }
    /*!
    \param fRGB rgb matrix
    \param r red channel output
    \param g green channel output
    \param b green channel output
    *
    * Extract the RGB channels of the rgb matrix
    */
    template<int DIM,typename TypeRGBChannel>
    static void toRGB(const MatN<DIM,RGB<TypeRGBChannel> > & fRGB,  MatN<DIM,TypeRGBChannel> &r,MatN<DIM,TypeRGBChannel> &g,MatN<DIM,TypeRGBChannel> &b)
    {
        r.resize(fRGB.getDomain());
        g.resize(fRGB.getDomain());
        b.resize(fRGB.getDomain());
        FunctorF::FunctorFromVectorToScalarCoordinate<TypeRGBChannel,RGB<TypeRGBChannel> > fred(0);
        FunctorF::FunctorFromVectorToScalarCoordinate<TypeRGBChannel,RGB<TypeRGBChannel> > fgreen(1);
        FunctorF::FunctorFromVectorToScalarCoordinate<TypeRGBChannel,RGB<TypeRGBChannel> > fblue(2);
        std::transform(fRGB.begin(),fRGB.end(),r.begin(),fred);
        std::transform(fRGB.begin(),fRGB.end(),g.begin(),fgreen);
        std::transform(fRGB.begin(),fRGB.end(),b.begin(),fblue);
    }

    /*!
    \param r red channel input
    \param g green channel input
    \param b green channel input
    \param fRGB
    *
    * set the RGB channels of the rgb matrix
    */
    template<int DIM,typename TypeRGBChannel>
    static void fromRGB(const MatN<DIM,TypeRGBChannel> &r,const MatN<DIM,TypeRGBChannel> &g,const MatN<DIM,TypeRGBChannel> &b, MatN<DIM,RGB<TypeRGBChannel> >& fRGB)
    {

        POP_DbgAssertMessage(r.getDomain()==g.getDomain()&&g.getDomain()==b.getDomain(),"In MatN::fromRGB, R G B matrixs must have the same size");
        fRGB.resize(r.getDomain());
        FunctorF::FunctorFromMultiCoordinatesToVector<TypeRGBChannel,RGB<TypeRGBChannel> > f;
        transform3(r.begin(),r.end(),g.begin(),b.begin(),fRGB.begin(),f);
    }

    /*!
    \param fRGB rgb input image
    \param y Y channel output
    \param u U channel output
    \param v V channel output
    *
    * Extract the YUV channels of the rgb matrix http://en.wikipedia.org/wiki/YUV
    */
    template<int DIM,typename TypeRGBChannel>
    static void toYUV(const MatN<DIM,RGB<TypeRGBChannel> > & fRGB, MatN<DIM,TypeRGBChannel> &y,MatN<DIM,TypeRGBChannel> &u,MatN<DIM,TypeRGBChannel> &v)
    {
        y.resize(fRGB.getDomain());
        u.resize(fRGB.getDomain());
        v.resize(fRGB.getDomain());
        typename MatN<DIM,TypeRGBChannel>::iterator  ity = y.begin();
        typename MatN<DIM,TypeRGBChannel>::iterator itu = u.begin();
        typename MatN<DIM,TypeRGBChannel>::iterator itv = v.begin();
        typename MatN<DIM,RGB<TypeRGBChannel> >::const_iterator it = fRGB.begin();
        typename MatN<DIM,RGB<TypeRGBChannel> >::const_iterator itend = fRGB.end();
        while (it != itend) {
            it->toYUV( *ity, *itu, *itv);
            ity++; itu++; itv++;it++;
        }
    }
    /*!
    \param y Y channel input
    \param u U channel input
    \param v V channel input
    \param fRGB rgb output image
    *
    * set the YUV channels of the rgb matrix http://en.wikipedia.org/wiki/YUV
    */
    template<int DIM,typename TypeRGBChannel>
    static void fromYUV(const MatN<DIM,TypeRGBChannel> &y,const MatN<DIM,TypeRGBChannel> &u,const MatN<DIM,TypeRGBChannel> &v, MatN<DIM,RGB<TypeRGBChannel> >& fRGB)
    {

        POP_DbgAssertMessage(y.getDomain()==u.getDomain()&&y.getDomain()==v.getDomain(),"In MatN::fromRGB, Y U B matrixs must have the same size");
        fRGB.resize(y.getDomain());
        typename MatN<DIM,TypeRGBChannel>::const_iterator ity = y.begin();
        typename MatN<DIM,TypeRGBChannel>::const_iterator ityend = y.end();
        typename MatN<DIM,TypeRGBChannel>::const_iterator itu = u.begin();
        typename MatN<DIM,TypeRGBChannel>::const_iterator itv = v.begin();
        typename MatN<DIM,RGB<TypeRGBChannel> >::iterator it = fRGB.begin();
        while (ity != ityend) {
            it->fromYUV(*ity,*itu,*itv);
            ity++; itu++; itv++;it++;
        }
    }
    /*!

    \param real real input
    \param img imaginary input
    \param f complex image output
    *
    * h(x) =real(x)+i*img(x)
    */
    template<int DIM,typename TypeRealImaginary>
    static void fromRealImaginary(const MatN<DIM,TypeRealImaginary> &real,const MatN<DIM,TypeRealImaginary> &img,MatN<DIM,Complex<TypeRealImaginary> > & f)
    {
        POP_DbgAssertMessage(real.getDomain()==img.getDomain(),"In MatN::fromRealImaginary, real and img matrixs must have the same size");
        f.resize(real.getDomain());
        FunctorF::FunctorFromMultiCoordinatesToVector<TypeRealImaginary,Complex<TypeRealImaginary> > func;
        std::transform(real.begin(),real.end(),img.begin(),f.begin(),func);
    }
    /*!
    \param real real input
    \param f complex image output
    *
    * h(x) =real(x) with 0 for imaginary part
        In this example, we apply a low pass filter:
        \code
        Mat2UI8 img;//2d grey-level image object
        img.load((std::string(POP_PROJECT_SOURCE_DIR)+"/image/Lena.bmp").c_str());//replace this path by those on your computer
        Mat2F32 noisemap;
        DistributionNormal d(0,20);
        Processing::randomField(img.getDomain(),d,noisemap);
        Mat2F32 imgf = Mat2F32(img) +noisemap;
        Mat2ComplexF32 imgcomplex;
        Convertor::fromRealImaginary(imgf,imgcomplex);
        Mat2ComplexF32 fft(imgcomplex);
        fft = Representation::FFT(fft,FFT_FORWARD);
        Mat2ComplexF32 filterlowpass = Representation::lowPass(fft,60);
        imgcomplex = Representation::FFT(filterlowpass,FFT_BACKWARD);
        Representation::scale(imgcomplex);
        Mat2F32 imgd;
        Convertor::toRealImaginary(imgcomplex,imgd);
        img = Processing::greylevelRange(imgd,0,255);
        img.display();
        \endcode

    */
    template<int DIM,typename TypeRealImaginary>
    static void fromRealImaginary(const MatN<DIM,TypeRealImaginary> &real,MatN<DIM,Complex<TypeRealImaginary> > & f)
    {
        f.resize(real.getDomain());
        FunctorF::FunctorFromMultiCoordinatesToVector<TypeRealImaginary,Complex<TypeRealImaginary> > func;
        std::transform(real.begin(),real.end(),f.begin(),func);
    }
    /*!
    \param fcomplex complex image input
    \param real real output
    \param imaginary imaginary output
    *
    * real(x)=Real(f(x)) and img(x)=Imaginary(f(x))
    */
    template<int DIM,typename TypeRealImaginary>
    static void toRealImaginary(const MatN<DIM,Complex<TypeRealImaginary> > & fcomplex, MatN<DIM,TypeRealImaginary>  &real, MatN<DIM,TypeRealImaginary>  &imaginary)
    {
        real.resize(fcomplex.getDomain());
        imaginary.resize(fcomplex.getDomain());
        FunctorF::FunctorFromVectorToScalarCoordinate<TypeRealImaginary,Complex<TypeRealImaginary> > freal(0);
        FunctorF::FunctorFromVectorToScalarCoordinate<TypeRealImaginary,Complex<TypeRealImaginary> > fimg(1);
        std::transform(fcomplex.begin(),fcomplex.end(),real.begin(),freal);
        std::transform(fcomplex.begin(),fcomplex.end(),imaginary.begin(),fimg);
    }
    /*!
    \param fcomplex complex image input
    \param real real output
    *
    *  real(x)=Real(f(x))
    */
    template<int DIM,typename TypeRealImaginary>
    static void toRealImaginary(const MatN<DIM,Complex<TypeRealImaginary> > & fcomplex, MatN<DIM,TypeRealImaginary>  &real)
    {
        real.resize(fcomplex.getDomain());
        FunctorF::FunctorFromVectorToScalarCoordinate<TypeRealImaginary,Complex<TypeRealImaginary> > freal(0);
        std::transform(fcomplex.begin(),fcomplex.end(),real.begin(),freal);
    }
    /*!
    \param vecf vector of field
    \param f field of vector
    *
    *  vecf(0)(x)=f(x)(0)
    */

    template<typename FunctionVec, typename FunctionScalar>
    static void fromVecN( const VecN<FunctionVec::F::DIM,FunctionScalar >& vecf,FunctionVec& f){
        fromVecN(vecf,f,Int2Type<FunctionVec::F::DIM>());
    }
    /*!
     * \param vecf vector of field
     * \param f field of vector
     *
     *  vecf(0)(x)=f(x)(0)
     *
     * \code
     * Mat2UI8 img;//2d grey-level image object
     * img.load((std::string(POP_PROJECT_SOURCE_DIR)+"/image/Lena.bmp").c_str());//replace this path by those on your computer
     * //gradient vector field
     * MatN<2,Vec2F32> gradient_vector_field = Processing::gradientVecGaussian(img,1);
     * //get each gradient in each direction in a
     * VecN<2,Mat2F32> fields_gradient;
     * pop::Convertor::toVecN(gradient_vector_field,fields_gradient);//gradient_vector_field(x)(coordinate)=fields_gradient(coordinate)(x)
     * fields_gradient(0).display("gradient",false);//gradient in  vertical direction
     * fields_gradient(1).display();//gradient in  horizontal direction
     * \endcode
     * \image html lenagrad0.jpg "gradient in vertical direction"
     * \image html lenagrad1.jpg "gradient in horizontal direction"
     */

    template<typename FunctionVec, typename FunctionScalar>
    static void toVecN( const FunctionVec& f, VecN<FunctionVec::F::DIM,FunctionScalar >& vecf){
        toVecN(f,vecf,Int2Type<FunctionVec::F::DIM>());
    }





private:
    template <typename In1, typename In2, typename In3, typename Out, typename FUNC>
    static Out transform3(In1 first1, In1 last1, In2 first2, In3 first3, Out out, FUNC f) {
        while (first1 != last1) {
            *out++ = f(*first1++, *first2++, *first3++);
        }
        return out;
    }
    template <typename In1, typename In2, typename In3,typename In4, typename Out, typename FUNC>
    static Out transform4(In1 first1, In1 last1, In2 first2, In3 first3, In4 first4,Out out, FUNC f) {
        while (first1 != last1) {
            *out++ = f(*first1++, *first2++, *first3++,*first4++);
        }
        return out;
    }
    template<typename FunctionVec, typename FunctionScalar>
    static void fromVecN( const VecN<1,FunctionScalar >& vecf,FunctionVec& f,Int2Type<1>)
    {
        const FunctionScalar & f0= vecf(0);
        f.resize(f0.getDomain());
        FunctorF::FunctorFromMultiCoordinatesToVector<typename FunctionScalar::F,typename FunctionVec::F> func;
        std::transform(f0.begin(),f0.end(),f.begin(),func);
    }
    template<typename FunctionVec, typename FunctionScalar>
    static void fromVecN( const VecN<2,FunctionScalar >& vecf,FunctionVec& f,Int2Type<2>)
    {
        const FunctionScalar & f0= vecf(0);
        const FunctionScalar & f1= vecf(1);
        f.resize(f0.getDomain());
        FunctorF::FunctorFromMultiCoordinatesToVector<typename FunctionScalar::F,typename FunctionVec::F> func;
        std::transform(f0.begin(),f0.end(),f1.begin(),f.begin(),func);
    }
    template<typename FunctionVec, typename FunctionScalar>
    static void fromVecN( const VecN<3,FunctionScalar >& vecf,FunctionVec& f,Int2Type<3>)
    {
        const FunctionScalar & f0= vecf(0);
        const FunctionScalar & f1= vecf(1);
        const FunctionScalar & f2= vecf(2);
        f.resize(f0.getDomain());
        FunctorF::FunctorFromMultiCoordinatesToVector<typename FunctionScalar::F,typename FunctionVec::F> func;
        transform3(f0.begin(),f0.end(),f1.begin(),f2.begin(),f.begin(),func);
    }
    template<typename FunctionVec, typename FunctionScalar>
    static void fromVecN( const VecN<4,FunctionScalar >& vecf,FunctionVec& f,Int2Type<4>)
    {
        const FunctionScalar & f0= vecf(0);
        const FunctionScalar & f1= vecf(1);
        const FunctionScalar & f2= vecf(2);
        const FunctionScalar & f3= vecf(3);
        f.resize(f0.getDomain());
        FunctorF::FunctorFromMultiCoordinatesToVector<typename FunctionScalar::F,typename FunctionVec::F> func;
        transform4(f0.begin(),f0.end(),f1.begin(),f2.begin(),f3.begin(),f.begin(),func);
    }
    template<typename FunctionVec, typename FunctionScalar>
    static void toVecN(const FunctionVec& f, VecN<1,FunctionScalar >& vecf,Int2Type<1>)
    {
        FunctionScalar & f0= vecf(0);
        f0.resize(f.getDomain());
        FunctorF::FunctorFromVectorToScalarCoordinate<typename FunctionScalar::F,typename FunctionVec::F> func0(0);
        std::transform(f.begin(),f.end(),f0.begin(),func0);
    }
    template<typename FunctionVec, typename FunctionScalar>
    static void toVecN( const FunctionVec& f, VecN<2,FunctionScalar >& vecf,Int2Type<2>)
    {
        FunctionScalar & f0= vecf(0);
        FunctionScalar & f1= vecf(1);
        f0.resize(f.getDomain());
        f1.resize(f.getDomain());
        FunctorF::FunctorFromVectorToScalarCoordinate<typename FunctionScalar::F,typename FunctionVec::F> func0(0);
        FunctorF::FunctorFromVectorToScalarCoordinate<typename FunctionScalar::F,typename FunctionVec::F> func1(1);
        std::transform(f.begin(),f.end(),f0.begin(),func0);
        std::transform(f.begin(),f.end(),f1.begin(),func1);

    }
    template<typename FunctionVec, typename FunctionScalar>
    static void toVecN( const FunctionVec& f, VecN<3,FunctionScalar >& vecf,Int2Type<3>)
    {
        FunctionScalar & f0= vecf(0);
        FunctionScalar & f1= vecf(1);
        FunctionScalar & f2= vecf(2);

        f0.resize(f.getDomain());
        f1.resize(f.getDomain());
        f2.resize(f.getDomain());
        FunctorF::FunctorFromVectorToScalarCoordinate<typename FunctionScalar::F,typename FunctionVec::F> func0(0);
        FunctorF::FunctorFromVectorToScalarCoordinate<typename FunctionScalar::F,typename FunctionVec::F> func1(1);
        FunctorF::FunctorFromVectorToScalarCoordinate<typename FunctionScalar::F,typename FunctionVec::F> func2(2);
        std::transform(f.begin(),f.end(),f0.begin(),func0);
        std::transform(f.begin(),f.end(),f1.begin(),func1);
        std::transform(f.begin(),f.end(),f2.begin(),func2);
    }
    template<typename FunctionVec, typename FunctionScalar>
    static void toVecN( const FunctionVec& f, VecN<4,FunctionScalar >& vecf,Int2Type<4>)
    {
        FunctionScalar & f0= vecf(0);
        FunctionScalar & f1= vecf(1);
        FunctionScalar & f2= vecf(2);
        FunctionScalar & f3= vecf(3);
        f0.resize(f.getDomain());
        f1.resize(f.getDomain());
        f2.resize(f.getDomain());
        f3.resize(f.getDomain());
        FunctorF::FunctorFromVectorToScalarCoordinate<typename FunctionScalar::F,typename FunctionVec::F> func0(0);
        FunctorF::FunctorFromVectorToScalarCoordinate<typename FunctionScalar::F,typename FunctionVec::F> func1(1);
        FunctorF::FunctorFromVectorToScalarCoordinate<typename FunctionScalar::F,typename FunctionVec::F> func2(2);
        FunctorF::FunctorFromVectorToScalarCoordinate<typename FunctionScalar::F,typename FunctionVec::F> func3(3);
        std::transform(f.begin(),f.end(),f0.begin(),func0);
        std::transform(f.begin(),f.end(),f1.begin(),func1);
        std::transform(f.begin(),f.end(),f2.begin(),func2);
        std::transform(f.begin(),f.end(),f3.begin(),func3);
    }
};
}
#endif // CONVERTOR_H
