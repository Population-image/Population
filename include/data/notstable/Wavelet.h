#ifndef WAVELET_H
#define WAVELET_H
#include"data/vec/VecN.h"
#include"algorithm/Processing.h"
namespace pop{


/*! \ingroup Data
* \defgroup Wavelet Wavelet
* @{
*/
//Pure interface
template<int Dim,typename PixelType>
class POP_EXPORTS Wavelet
{
public:

    /*!
        \class pop::Wavelet
        \brief  Interface for the multi-dimensional wavelet classes
        \author Tariel Vincent

        The concrete classes must implement the copy constructor

    */

    typedef VecN<Dim,I32> Feature;

    /*!
     \brief scale the wavelet by the  \f$ \psi_\lambda\leftarrow (\psi,\lambda)  \f$
     \param lambda scale vector
     *
     */
    virtual void scale(const VecN<Dim,F32> & lambda)=0;

    /*!
     \brief iterative translation of the wavelet\f$ \psi_x\leftarrow (\psi,x)  \f$
     \param x translate vector

     * if you call it two time, the trranslation is the sum of the two vector
     */
    virtual void translate(const VecN<Dim,F32> & x)=0;

    /*!
     \brief wavelet coefficient as inner product between the wavelet and the input function \f$a(\psi_{\lambda,x},f) =\langle \psi_{\lambda,x} , f \rangle=\int_{R^n} \psi_{\lambda,x}(x')f(x')\,dx'.\f$
     \param x input VecN
     \return wavelet coefficient
     */
    virtual F32 operator ()(const VecN<Dim,I32> & x)=0;



    /*!
    \brief create a collection of wavelet from a single wavelet by scaling it
    \param wavelet_elementary elementatary wavelet
    \param minscale minimum scale parameter
    \param maxscale maximum scale parameter
    \param step incrementation step
    \return scaling collection of wavelets

    the collection of wavelet is the result of the scaling of the elementary wavelet between the range [minscale,maxscale] with a given step
    */
    template<typename WaveletConcrete>
    static std::vector<WaveletConcrete > scaleElementaryWavelet(const WaveletConcrete & wavelet_elementary,const VecN<Dim,F32>& minscale,const VecN<Dim,F32>& maxscale,const VecN<Dim,F32>& scalefactor){
        std::vector<WaveletConcrete > v_base;
        if(Dim==2){
            VecN<Dim,F32> scale;
            for(scale(0)=minscale(0);scale(0)<=maxscale(0);scale(0)*=scalefactor(0)){
                for(scale(1)=minscale(1);scale(1)<=maxscale(1);scale(1)*=scalefactor(1)){
                    WaveletConcrete wavelet(wavelet_elementary);
                    wavelet.scale(scale);
                    v_base.push_back(wavelet);
                }
            }
            return v_base;
        }else if(Dim==3){
            VecN<Dim,F32> scale;
            for(scale(0)=minscale(0);scale(0)<=maxscale(0);scale(0)*=scalefactor(0)){
                for(scale(1)=minscale(1);scale(1)<=maxscale(1);scale(1)*=scalefactor(1)){
                    for(scale(2)=minscale(2);scale(2)<=maxscale(2);scale(2)*=scalefactor(2)){
                        WaveletConcrete wavelet(wavelet_elementary);
                        wavelet.scale(scale);
                        v_base.push_back(wavelet);
                    }
                }
            }
            return v_base;
        }

        return v_base;
    }
    // the window size should be power of 2  work only in 2d
    template<typename WaveletConcrete>
    static std::vector<WaveletConcrete > scanWindow(const WaveletConcrete& wavelet_elementary,const VecN<Dim,F32>& radius=16,const VecN<Dim,F32>& scalefactor =2 ,const VecN<Dim,F32>& step =2, const VecN<Dim,F32>& border=2){
        std::vector<WaveletConcrete > v_wavelet;
        VecN<Dim,F32> minscale (1);
        VecN<Dim,F32> maxscale (radius);
        std::vector<WaveletConcrete > v_wavelet_scale = scaleElementaryWavelet(wavelet_elementary,minscale,maxscale,scalefactor);
        VecN<Dim,F32> x = -radius+border;
        for(x(0)=-radius(0)+border(0);x(0)<=radius(0)-border(0);x(0)+=step(0)){
            for(x(1)=-radius(1)+border(1);x(1)<=radius(1)-border(1);x(1)+=step(1)){
                typename std::vector<WaveletConcrete >::iterator it;
                for(it=v_wavelet_scale.begin();it!=v_wavelet_scale.end();it++){
                    WaveletConcrete wavelet=*it;
                    wavelet.translate(x);
                    if(wavelet.isValid(-radius,radius))
                        v_wavelet.push_back(wavelet);
                }

            }
        }
        return v_wavelet;
    }
};



template<int Dim,typename PixelType>
class POP_EXPORTS WaveletHaar :public Wavelet<Dim,PixelType >
{
    /*!
        \class pop::WaveletHaar
        \brief  pseudo Haar wavelet with integral matrixs to speed-up the calculation
        \author Tariel Vincent


        as explained in Viola and Jone article, we normalize the calculation by the standart deviation. The wavelet coefficient is \f$a(\phi,f) =\frac{<f>_{\sqcap_+}- <f>_{\sqcap_-}}{(<f^2>_\sqcap-<f>^2_\sqcap)}  \f$ such that \f$< >\f$ is the mean symbol,
        \f$\sqcap_- \f$ is the negative rectangular windows,  \f$\sqcap_+ \f$ is the possitive rectangular windows and \f$\sqcap =\sqcap_+ \cup \sqcap-   \f$ is the windows
    */
public:
    typedef VecN<Dim,I32> Feature;

    /*!
      \brief constructor by defining the rectangular windows \f$\sqcap\f$=[windows_min,windows_max] and the negative rectangular windows\f$\sqcap_-\f$=[windows_negative_min,windows_negative_max]
    such that the possitive rectangular windows is equal to \f$\sqcap\setminus\sqcap_- \f$
    *
    */
    WaveletHaar(const VecN<Dim,F32> & windows_min,const VecN<Dim,F32> & windows_max,const VecN<Dim,F32> & windows_negative_min,const VecN<Dim,F32> & windows_negative_max )
    {
        _x1_windows = windows_min;
        _x4_windows = windows_max;
        _x2_windows(0)=_x4_windows(0);_x2_windows(1)=_x1_windows(1);
        _x3_windows(0)=_x1_windows(0);_x3_windows(1)=_x4_windows(1);


        _x1_windows_negative = windows_negative_min;
        _x4_windows_negative = windows_negative_max;
        _x2_windows_negative(0)=_x4_windows_negative(0);_x2_windows_negative(1)=_x1_windows_negative(1);
        _x3_windows_negative(0)=_x1_windows_negative(0);_x3_windows_negative(1)=_x4_windows_negative(1);


        _area_windows = (_x4_windows-_x1_windows).multCoordinate();
        _area_windows_negative = (_x4_windows_negative-_x1_windows_negative).multCoordinate();
        _area_windows_possitive = _area_windows - _area_windows_negative;
    }
    void scale(const VecN<Dim,F32> & scalefactor){
        _x1_windows=_x1_windows*scalefactor;
        _x2_windows=_x2_windows*scalefactor;
        _x3_windows=_x3_windows*scalefactor;
        _x4_windows=_x4_windows*scalefactor;
        _x1_windows_negative=_x1_windows_negative*scalefactor;
        _x2_windows_negative=_x2_windows_negative*scalefactor;
        _x3_windows_negative=_x3_windows_negative*scalefactor;
        _x4_windows_negative=_x4_windows_negative*scalefactor;


        _area_windows = (_x4_windows-_x1_windows).multCoordinate();
        _area_windows_negative = (_x4_windows_negative-_x1_windows_negative).multCoordinate();
        _area_windows_possitive = _area_windows - _area_windows_negative;

    }
    virtual void translate(const VecN<Dim,F32> & x){
        _x1_windows+=x;
        _x2_windows+=x;
        _x3_windows+=x;
        _x4_windows+=x;


        _x1_windows_negative+=x;
        _x2_windows_negative+=x;
        _x3_windows_negative+=x;
        _x4_windows_negative+=x;
    }

    F32 operator ()(const VecN<Dim,I32> & x){
        VecN<Dim,F32> x1_windows_translate;
        VecN<Dim,F32> x2_windows_translate;
        VecN<Dim,F32> x3_windows_translate;
        VecN<Dim,F32> x4_windows_translate;

        VecN<Dim,F32> x1_windows_negative_translate;
        VecN<Dim,F32> x2_windows_negative_translate;
        VecN<Dim,F32> x3_windows_negative_translate;
        VecN<Dim,F32> x4_windows_negative_translate;
        x1_windows_translate = x + _x1_windows;
        x2_windows_translate = x + _x2_windows;
        x3_windows_translate = x + _x3_windows;
        x4_windows_translate = x + _x4_windows;


        x1_windows_negative_translate = x + _x1_windows_negative;
        x2_windows_negative_translate = x + _x2_windows_negative;
        x3_windows_negative_translate = x + _x3_windows_negative;
        x4_windows_negative_translate = x + _x4_windows_negative;

        F32 sumwindowspower2;
        sumwindowspower2 = _funtion_integral_power2(x1_windows_translate)+_funtion_integral_power2(x4_windows_translate)-(_funtion_integral_power2(x2_windows_translate)+_funtion_integral_power2(x3_windows_translate));

        F32 sumwindows;
        sumwindows =      _funtion_integral(x1_windows_translate)+_funtion_integral(x4_windows_translate)-(_funtion_integral(x2_windows_translate)+_funtion_integral(x3_windows_translate));
        F32 mean_windows      = sumwindows/_area_windows;
        F32 mean_deviation_windows =sumwindowspower2/_area_windows-mean_windows*mean_windows;
        if(mean_deviation_windows>0)
            mean_deviation_windows = std::sqrt( mean_deviation_windows);
        else
            return 0;

        F32 sumwindows_negative;

        sumwindows_negative =        _funtion_integral(x1_windows_negative_translate)+_funtion_integral(x4_windows_negative_translate)-(_funtion_integral(x2_windows_negative_translate)+_funtion_integral(x3_windows_negative_translate));
        F32 mean_negative = sumwindows_negative/_area_windows_negative;
        // use this relation xox = xxx -  x
        F32 sumwindows_possitive = sumwindows-sumwindows_negative;
        F32 mean_possitive = sumwindows_possitive/_area_windows_possitive;
        F32 result = (mean_possitive -  mean_negative)/  mean_deviation_windows;
        return result;

    }

    /*!
     \brief nice elementary base of Haar wavelet with these properties area(positive)=area(negative) and  area(positive)+area(negative)=2
     \return wavelet base
     */
    static Vec<WaveletHaar<Dim,PixelType > > baseWaveletHaar(){
        Vec<WaveletHaar<Dim,PixelType > > v_haar_feature;

        if(Dim==2){

            VecN<Dim,F32> windows_full_negative_corner,windows_full_positive_corner,windows_negative_negative_corner,windows_negative_positive_corner;

            //A
            // xo;
            windows_full_negative_corner(0)=-1;windows_full_negative_corner(1)=-1;
            windows_full_positive_corner(0)= 1;windows_full_positive_corner(1)= 1;
            windows_negative_negative_corner(0)= 0;  windows_negative_negative_corner(1)=-1;
            windows_negative_positive_corner(0)= 1;windows_negative_positive_corner(1)= 1;
            v_haar_feature.push_back(WaveletHaar<Dim,PixelType>(windows_full_negative_corner,windows_full_positive_corner,windows_negative_negative_corner,windows_negative_positive_corner));

            //B
            //  x
            //  o
            windows_full_negative_corner(0)=-1;windows_full_negative_corner(1)=-1;
            windows_full_positive_corner(0)= 1;windows_full_positive_corner(1)= 1;
            windows_negative_negative_corner(0)=-1;windows_negative_negative_corner(1)=0;
            windows_negative_positive_corner(0)= 1;windows_negative_positive_corner(1)=1;
            v_haar_feature.push_back(WaveletHaar<Dim,PixelType>(windows_full_negative_corner,windows_full_positive_corner,windows_negative_negative_corner,windows_negative_positive_corner));

            //C
            //  xox = xxx - 2*o
            windows_full_negative_corner(0)=-1;windows_full_negative_corner(1)=-1;
            windows_full_positive_corner(0)= 1;windows_full_positive_corner(1)= 1;
            windows_negative_negative_corner(0)=-0.5;windows_negative_negative_corner(1)=-1;
            windows_negative_positive_corner(0)= 0.5;windows_negative_positive_corner(1)= 1;
            v_haar_feature.push_back(WaveletHaar<Dim,PixelType>(windows_full_negative_corner,windows_full_positive_corner,windows_negative_negative_corner,windows_negative_positive_corner));

            //D
            //  x
            //  o
            //  x
            windows_full_negative_corner(0)=-1;windows_full_negative_corner(1)=-1;
            windows_full_positive_corner(0)= 1;windows_full_positive_corner(1)= 1;
            windows_negative_negative_corner(0)=-1;windows_negative_negative_corner(1)=-0.5;
            windows_negative_positive_corner(0)= 1;windows_negative_positive_corner(1)= 0.5;
            v_haar_feature.push_back(WaveletHaar<Dim,PixelType>(windows_full_negative_corner,windows_full_positive_corner,windows_negative_negative_corner,windows_negative_positive_corner));

            //E
            // xxx
            // x0x
            // xxx
            windows_full_negative_corner(0)=-1;windows_full_negative_corner(1)=-1;
            windows_full_positive_corner(0)= 1;windows_full_positive_corner(1)= 1;
            windows_negative_negative_corner(0)=-0.707106781;windows_negative_negative_corner(1)=-0.707106781;
            windows_negative_positive_corner(0)= 0.707106781;windows_negative_positive_corner(1)= 0.707106781;
            v_haar_feature.push_back(WaveletHaar<Dim,PixelType>(windows_full_negative_corner,windows_full_positive_corner,windows_negative_negative_corner,windows_negative_positive_corner));
        }
        return v_haar_feature;
    }
    bool isValid(const VecN<Dim,I32> & x ){
        VecN<Dim,F32> x1_windows_translate;
        VecN<Dim,F32> x2_windows_translate;
        VecN<Dim,F32> x3_windows_translate;
        VecN<Dim,F32> x4_windows_translate;
        x1_windows_translate = x + _x1_windows;
        x2_windows_translate = x + _x2_windows;
        x3_windows_translate = x + _x3_windows;
        x4_windows_translate = x + _x4_windows;
        if(_funtion_integral.isValid(x1_windows_translate)&&_funtion_integral.isValid(x2_windows_translate)&&_funtion_integral.isValid(x3_windows_translate)&&_funtion_integral.isValid(x4_windows_translate))
            return true;
        else
            return false;
    }
    bool isValid(VecN<Dim,I32> radius_min,VecN<Dim,I32> radius_max ){
        //TODO
//        if(VecN<Dim,I32>(_x1_windows).allSuperiorEqual(radius_min) &&VecN<Dim,I32>(_x1_windows).allInferiorEqual(radius_max)&&

//            return true;
//        else
//            return false;
        std::cerr << "WARNING: Calling Wavelet::isValid() which does nothing for now" << std::endl;
        return false;
    }

    void setMatrix(const MatN<Dim,PixelType> & f){
        _funtion = &f;
        _funtion_integral = Processing::integral(*_funtion);
        _funtion_integral_power2= Processing::integralPower2(*_funtion);
    }
    // we use this formula xox = xxx - o where xox is the Harr wavelet, xxx is the windows and  x is the negative part in the the Harr wavelet

    Mat2RGBUI8 display(const VecN<Dim,I32> & x){
        Mat2RGBUI8 out(_funtion->getDomain());
        Mat2RGBUI8::IteratorERectangle itrec(out.getIteratorERectangle(_x1_windows+ x,_x4_windows+x));
        while(itrec.next()){
            if(itrec.x()>=(_x1_windows_negative +x) && itrec.x()<=(_x4_windows_negative+x)){
                out(itrec.x())=RGBUI8( (*_funtion)(itrec.x()),0,0);
            }else
                out(itrec.x())=RGBUI8(0,0,(*_funtion)(itrec.x()));
        }
        return out;
    }

    void save(std::ostream& out)const{
        out<<_x1_windows<<"<Haar>";
        out<<_x2_windows<<"<Haar>";
        out<<_x3_windows<<"<Haar>";
        out<<_x4_windows<<"<Haar>";
        out<<_x1_windows_negative<<"<Haar>";
        out<<_x2_windows_negative<<"<Haar>";
        out<<_x3_windows_negative<<"<Haar>";
        out<<_x4_windows_negative<<"<Haar>";
    }
    void load(std::istream& in){
        std::string str;
        str = pop::BasicUtility::getline(in,"<Haar>");
        pop::BasicUtility::String2Any(str,_x1_windows);
        str = pop::BasicUtility::getline(in,"<Haar>");
        pop::BasicUtility::String2Any(str,_x2_windows );
        str = pop::BasicUtility::getline(in,"<Haar>");
        pop::BasicUtility::String2Any(str,_x3_windows );
        str = pop::BasicUtility::getline(in,"<Haar>");
        pop::BasicUtility::String2Any(str,_x4_windows );
        str = pop::BasicUtility::getline(in,"<Haar>");
        pop::BasicUtility::String2Any(str,_x1_windows_negative);
        str = pop::BasicUtility::getline(in,"<Haar>");
        pop::BasicUtility::String2Any(str,_x2_windows_negative );
        str = pop::BasicUtility::getline(in,"<Haar>");
        pop::BasicUtility::String2Any(str,_x3_windows_negative );
        str = pop::BasicUtility::getline(in,"<Haar>");
        pop::BasicUtility::String2Any(str,_x4_windows_negative );
    }

private:

    VecN<Dim,F32> _x1_windows;
    VecN<Dim,F32> _x2_windows;
    VecN<Dim,F32> _x3_windows;
    VecN<Dim,F32> _x4_windows;

    VecN<Dim,F32> _x1_windows_negative;
    VecN<Dim,F32> _x2_windows_negative;
    VecN<Dim,F32> _x3_windows_negative;
    VecN<Dim,F32> _x4_windows_negative;


    double _area_windows;
    double _area_windows_negative;
    double _area_windows_possitive;

    static const MatN<Dim,PixelType> * _funtion;
    static  MatN<Dim,PixelType> _funtion_integral;
    static  MatN<Dim,PixelType> _funtion_integral_power2;
};

/*!
@}
*/

template<int Dim,typename PixelType>
const MatN<Dim,PixelType> *  WaveletHaar<Dim,PixelType>::_funtion=NULL;
template<int Dim,typename PixelType>
MatN<Dim,PixelType>   WaveletHaar<Dim,PixelType>::_funtion_integral;
template<int Dim,typename PixelType>
MatN<Dim,PixelType>   WaveletHaar<Dim,PixelType>::_funtion_integral_power2;





}


#endif // WAVELET_H
