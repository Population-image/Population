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
\image html lenanoise.jpg "noisy image"
\image html lenalowpass.jpg "filter image with low pass"
*/

enum FFT_WAY{
    FFT_FORWARD=1,
    FFT_BACKWARD=-1,
};
namespace Private{
template<typename T>
struct FFTAbtract{
    virtual void apply(T*  data,FFT_WAY way=FFT_FORWARD)=0;
};

template<unsigned N, typename T=F32>
struct FFTDanielsonLanczos :public FFTAbtract<T>{
    void apply(T* data,FFT_WAY way=FFT_FORWARD) {
        _sift(data);
        _exec(data, way);
    }
    FFTDanielsonLanczos<N/2,T> _scale_two;
    void _sift(T* data){
        int m;
        int n = N<<1;
        int j=1;
        for (int i=1; i<n; i+=2) {
            if (j>i) {
                std::swap(data[j-1], data[i-1]);
                std::swap(data[j], data[i]);
            }
            m = N;
            while (m>=2 && j>m) {
                j -= m;
                m >>= 1;
            }
            j += m;
        };
    }
    void _exec(T* data,FFT_WAY way){
        _scale_two._exec(data,way);
        _scale_two._exec(data+N,way);

        T wtemp,tempr,tempi,wr,wi,wpr,wpi;
        wtemp = sin(pop::PI/N);
        wpr = -2.0*wtemp*wtemp;
        wpi = -way*sin(2*pop::PI/N);

        wr = 1.0;
        wi = 0.0;
        for (unsigned i=0; i<N; i+=2) {
            tempr = data[i+N]*wr - data[i+N+1]*wi;
            tempi = data[i+N]*wi + data[i+N+1]*wr;
            data[i+N] = data[i]-tempr;
            data[i+N+1] = data[i+1]-tempi;
            data[i] += tempr;
            data[i+1] += tempi;

            wtemp = wr;
            wr += wr*wpr - wi*wpi;
            wi += wi*wpr + wtemp*wpi;
        }

    }
};

template<typename T>
class FFTDanielsonLanczos<4,T>:public FFTAbtract<T>  {
public:
    void apply(T* data,FFT_WAY way =FFT_FORWARD){
        std::swap(data[2],data[4]);
        std::swap(data[3],data[5]);
        _exec(data,way);
    }
    void _exec(T* data,FFT_WAY way=FFT_FORWARD) {
        T tr = data[2];
        T ti = data[3];
        data[2] = data[0]-tr;
        data[3] = data[1]-ti;
        data[0] += tr;
        data[1] += ti;
        tr = data[6];
        ti = data[7];
        if(way==FFT_FORWARD){
            data[6] = data[5]-ti;
            data[7] = tr-data[4];
        }else{
            data[6] = -(data[5]-ti);
            data[7] = -(tr-data[4]);
        }
        data[4] += tr;
        data[5] += ti;

        tr = data[4];
        ti = data[5];
        data[4] = data[0]-tr;
        data[5] = data[1]-ti;
        data[0] += tr;
        data[1] += ti;
        tr = data[6];
        ti = data[7];
        data[6] = data[2]-tr;
        data[7] = data[3]-ti;
        data[2] += tr;
        data[3] += ti;

    }
};

template<typename T>
struct FFTDanielsonLanczos<1,T>:public FFTAbtract<T> {
    void apply(T* ,FFT_WAY =FFT_FORWARD){}
    void _exec(T* ,FFT_WAY ) {}
};


template<int POW>
struct Pow2{
    enum{value=2*Pow2<POW-1>::value};
};
template<>
struct Pow2<0>{
    enum{value=1};
};
template<int SCALE,int SCALE2>
struct _InitFFT{
    static void init(Vec<FFTAbtract<F32> *>&fft_op,Vec<int>& fft_size ){
        fft_op.push_back(new FFTDanielsonLanczos<Pow2<SCALE>::value >);
        fft_size.push_back(Pow2<SCALE>::value);
        _InitFFT<SCALE+1,SCALE2>::init(fft_op,fft_size);
    }
};
template<int SCALE>
struct _InitFFT<SCALE,SCALE>{
    static void init(Vec<FFTAbtract<F32> *>&fft_op,Vec<int>& fft_size ){
        fft_op.push_back(new FFTDanielsonLanczos<Pow2<SCALE>::value >);
        fft_size.push_back(Pow2<SCALE>::value);
    }
};



template<int SCALE1=2,int SCALE2=12>
struct FFTConcrete{
    Vec<FFTAbtract<F32> *> _fft_op;
    Vec<int> _fft_size;
    int _select;
    FFTConcrete():_select(-1){
        _InitFFT<SCALE1,SCALE2>::init(_fft_op,_fft_size);
    }
    void select(int nbr_element){
        for(unsigned int i=0;i<=_fft_size.size();i++){
            if(_fft_size[i]==nbr_element){
                _select =i;
            }
            if(i==_fft_size.size()){
                std::cerr<<"Cannot find the FFT for this size"<<std::endl;
            }
        }
    }

    void apply(F32 *  data,int nbr_element,FFT_WAY way=FFT_FORWARD){
        if(_select>=0)
            _fft_op(_select)->apply(data,way);
        else{
            for(unsigned int i=0;i<=_fft_size.size();i++){
                if(_fft_size[i]==nbr_element){
                    _fft_op(i)->apply(data,way);
                    break;
                }
                if(i==_fft_size.size()){
                    std::cerr<<"Cannot find the FFT for this size"<<std::endl;
                }
            }
        }
    }
};
}

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

         *
        */
    static inline Mat2ComplexF32  FFT(const Mat2ComplexF32 & f ,FFT_WAY way=FFT_FORWARD)
    {
        Mat2ComplexF32  data;
        truncatePower2(f,data);
        Private::FFTConcrete<> fft;
        for(unsigned int i=0;i<data.sizeI();i++){
            ComplexF32 *line = data.data()+i*data.sizeJ();
            fft.apply(line->data(),data.sizeJ(),way);
        }
        ComplexF32 t[data.sizeI()];
        for(unsigned int j=0;j<data.sizeJ();j++){
            unsigned int position =j;
            for(unsigned int i=0;i<data.sizeI();i++){
                t[i]=data(position);
                position+=data.sizeJ();
            }
            fft.apply(t[0].data(),data.sizeI(),way);
            position =j;
            for(unsigned int i=0;i<data.sizeI();i++){
                data(position)=t[i];
                position+=data.sizeJ();
            }
        }
        return data;
    }
    static inline Mat3ComplexF32 FFT(const Mat3ComplexF32 & f,FFT_WAY way=FFT_FORWARD) {
        Mat3ComplexF32  data;
        truncatePower2(f,data);
        Private::FFTConcrete<> fft;
        for(unsigned int i=0;i<data.sizeI();i++){
            for(unsigned int k=0;k<data.sizeK();k++){
                ComplexF32 *line = data.data()+i*data.sizeJ()+k*data.sizeJ()*data.sizeI();
                fft.apply(line->data(),data.sizeJ(),way);
            }
        }
        ComplexF32 t[data.sizeI()];
        for(unsigned int j=0;j<data.sizeJ();j++){
            for(unsigned int k=0;k<data.sizeK();k++){
                unsigned int position =j+k*data.sizeJ()*data.sizeI();
                for(unsigned int i=0;i<data.sizeI();i++){
                    t[i]=data(position);
                    position+=data.sizeJ();
                }
                fft.apply(t[0].data(),data.sizeI(),way);
                position =j+k*data.sizeJ()*data.sizeI();
                for(unsigned int i=0;i<data.sizeI();i++){
                    data(position)=t[i];
                    position+=data.sizeJ();
                }
            }
        }
        ComplexF32 t2[data.sizeI()];
        for(unsigned int i=0;i<data.sizeI();i++){
            for(unsigned int j=0;j<data.sizeJ();j++){
                unsigned int position =j+i*data.sizeI();
                for(unsigned int k=0;k<data.sizeK();k++){
                    t2[k]=data(position);
                    position+=data.sizeJ()*data.sizeI();
                }
                fft.apply(t2[0].data(),data.sizeK(),way);
                position =j+i*data.sizeI();
                for(unsigned int k=0;k<data.sizeK();k++){
                    data(position)=t2[k];
                    position+=data.sizeJ()*data.sizeI();
                }
            }
        }
        return data;
    }
    static inline MatN<1,ComplexF32> FFT(MatN<1,ComplexF32> & f ,FFT_WAY way=FFT_FORWARD)
    {
        MatN<1,ComplexF32> data(f);
        Private::FFTConcrete<> fft;
        fft.apply(data.data()->data(),data.sizeI(),way);
        return data;
    }

    /*! \brief visualization of the fourrier matrix in log scale h(x) = log( ||f(x)||+1)
         * \param fft input FFT matrix
         * \return grey level matrix
        *
         \code
        Mat2UI8 img;
        img.load((std::string(POP_PROJECT_SOURCE_DIR)+"/image/eutel.bmp"));
        Mat2ComplexF32 imgc;
        Convertor::fromRealImaginary(Mat2F32(img),imgc);
        imgc = Representation::FFT(imgc);
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
     *  calculated the 2-Points correlation function in any direction by FFT  P = FFT^(-1)(FFT(f)FFT(f)^*)

     * \code
        Mat2UI8 img;
        img.load((std::string(POP_PROJECT_SOURCE_DIR)+"/image/eutel.bmp"));
        Representation::correlationDirectionByFFT(img).display();
        Mat2ComplexF32 imgc;
        Convertor::fromRealImaginary(Mat2F32(img),imgc);
        imgc = Representation::FFT(imgc);
        img = Representation::FFTDisplay(imgc);
        img.display();
     * \endcode
     */

    template<int DIM,typename PixelType>
    static MatN<DIM,F32> correlationDirectionByFFT(const MatN<DIM,PixelType> & f){

        MatN<DIM,PixelType> bint;
        pop::Representation::truncatePower2(f,bint);
        MatN<DIM,F32> binfloat(bint);
        typename MatN<DIM,PixelType>::IteratorEDomain it (binfloat.getIteratorEDomain());
        binfloat = pop::ProcessingAdvanced::greylevelRange(binfloat,it,0,1);


        MatN<DIM,ComplexF32>  bin_complex(bint.getDomain());
        Convertor::fromRealImaginary(binfloat,bin_complex);
        MatN<DIM,ComplexF32>  fft = pop::Representation::FFT(bin_complex,FFT_FORWARD);

        it.init();
        while(it.next()){
            ComplexF32 c = fft(it.x());
            ComplexF32 c1 = fft(it.x());
            fft(it.x()).real() = (c*c1.conjugate()).real();
            fft(it.x()).img() =0;
        }
        fft  = pop::Representation::FFT(fft,FFT_BACKWARD);
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
    static void truncatePower2(const MatN<DIM,PixelType>& in, MatN<DIM,PixelType>& out){
        VecN<DIM,I32> x;
        for(int i=0;i<DIM;i++){
            x(i) = (1<<(int)std::floor(std::log(in.getDomain()(i))/std::log(2)));
        }
        __resize(in,out,x,x);
    }
    template<typename MatN1,int DIM,typename PixelType>
    static void upPower2(const MatN1& in,MatN<DIM,PixelType> &out){
        VecN<DIM,I32> x;
        for(int i=0;i<DIM;i++){
            x(i) = 1<<(int)(std::floor(std::log(in.getDomain()(i))/std::log(2))+1);
        }
        __resize(in,out,x,in.getDomain());
    }

    template<int DIM>
    static void scale(MatN<DIM,ComplexF32>& data){
        for (unsigned i=0;i<data.size();i++) {
            data[i] /= data.size();
        }
    }
    //@}

    template<typename MatN1,int DIM,typename PixelType>
    static void __resize(const MatN1& in, MatN<DIM,PixelType>& out,VecN<DIM,I32> x,VecN<DIM,I32> size){
        if(out.getDomain()!=x){
            out.resize(x);
        }
        if(DIM==2){
            for( int i=0;i<size(0);i++){
                std::copy(in.begin()+i*in.sizeJ(),in.begin()+(i*in.sizeJ()+size(1)),out.begin()+i*out.sizeJ());
            }
        }else if(DIM==1){
            std::copy(in.begin(),in.begin()+size(0),out.begin());
        }else{
            for( int k=0;k<size(2);k++){
                for( int i=0;i<size(0);i++){
                    std::copy(in.begin()+i*in.sizeJ()+k*in.sizeI()*in.sizeJ(),in.begin()+(i*in.sizeJ()+size(1))+k*in.sizeI()*in.sizeJ(),out.begin()+i*out.sizeJ()+k*out.sizeI()*out.sizeJ());
                }
            }
        }
    }
};
}
#endif // REPRESENTATION_H
