
#include"Population.h"//Single header
using namespace pop;//Population namespace
#include<complex>
Mat2UI8 gaussianField(const Mat2F32& correlation)
{
    int sizei=correlation.sizeI();
    int sizej=correlation.sizeI();

    std::string str_cummulative_gausssian = "1/((2*pi)^(0.5))*exp(-(x^2)*0.5)";
    DistributionExpression f_beta(str_cummulative_gausssian);
    F32 beta= Statistics::maxRangeIntegral(f_beta,correlation(0,1),-4,4,0.001);
    MatN<1,F32> autocorrelation;
    autocorrelation.resize(VecN<1,int>(std::max(sizei*1.f,correlation.sizeI()*1.f)));
    for(unsigned int i= 0; i<autocorrelation.sizeI();i++)
    {
        if(i<correlation.sizeI())
            autocorrelation(i)=correlation(i,1)-correlation(0,1)*correlation(0,1);
        else
            autocorrelation(i)=0;
    }
    //std::cout<<autocorrelation<<std::endl;
    std::string s = BasicUtility::Any2String(beta);
    std::string  equation= "1/(2*pi*(1-x^2)^0.5)*exp(-("+s+")^2/(1+x))";
    DistributionExpression f(equation.c_str());
    DistributionRegularStep fintegral = Statistics::integral(f,0,1.2,0.001f);
    MatN<1,F32> rho;
    rho.resize(VecN<1,int>(autocorrelation.size()));
    for(unsigned int i= 0; i<rho.size()/2;i++) {

        if(autocorrelation(i)>0){
            double value=std::min(0.999f,autocorrelation(i));
            rho(i)=Statistics::FminusOneOfYMonotonicallyIncreasingFunction(fintegral,value,0,1.2,0.001f);
        }else{
            rho(i)=0;
        }
        if(rho(i)<0)
            rho(i)=0;
        if(i>0)
            rho(rho.size()-i)=rho(i);
    }

    //    std::cout<<rho<<std::endl;

    MatN<1,ComplexF32> rhocomplex;
    Convertor::fromRealImaginary(rho,rhocomplex);
    MatN<1,ComplexF32> fft = Representation::FFT(rhocomplex,1);
//    for(unsigned int i=0;i<fft.size();i++){
//        std::complex<F32> c(fft(i).real(),fft(i).img());
//        fft(i).real()=std::real(c);
//        fft(i).img()=std::imag(c);
//    }
    MatN<1,ComplexF32> weigh_see = Representation::FFT(rhocomplex,0);
    weigh_see *=weigh_see.size();
    MatN<1,ComplexF32> fft_reversee = Representation::FFT(weigh_see,1);
    F32 initial =fft_reversee(0).real()/fft(0).real();
    for(unsigned int i=0;i<fft.size();i++){
        std::cout<<fft_reversee(i).real()/fft(i).real()/initial<<std::endl;
    }
//    F32 initial =fft_reversee(0).real()*fft_reversee(0).real()/fft(0).real();
//    for(unsigned int i=0;i<fft.size();i++){
//        std::cout<<fft_reversee(i).real()*fft_reversee(i).real()/fft(i).real()/initial<<std::endl;
//    }
    exit(0);


    MatN<1,F32> w;

    Convertor::toRealImaginary(weigh_see,w);
    std::cout<<"merde"<<std::endl;
    //std::cout<<w<<std::endl;
    int radius_kernel=5;
    Vec<F32> kernel(radius_kernel*2+1);
    kernel(radius_kernel)=w(0);
    for(unsigned int i=1;i<=radius_kernel;i++){
        kernel(radius_kernel-i)=w(i);
        kernel(radius_kernel+i)=w(i);
    }
    F32 sum = std::accumulate(kernel.begin(),kernel.end(),F32(0));
    kernel = kernel*(1./sum);
    std::cout<<kernel<<std::endl;
    exit(0);



    MatN<2,ComplexF32> weighcomplex_2d(sizei,sizej);

    ForEachDomain2D(xx,weighcomplex_2d){
        F32 dist= Vec2F32(xx-weighcomplex_2d.getDomain()/2).norm(2);
        weighcomplex_2d(xx)=fft(std::floor(dist))*(1-(dist-std::floor(dist)))+ fft(std::floor(dist)+1)*((dist-std::floor(dist)));

    }
    weighcomplex_2d = GeometricalTransformation::translate(weighcomplex_2d,-weighcomplex_2d.getDomain()/2);
    //    std::cout<<weighcomplex_2d<<std::endl;




    Mat2F32 m_U(sizei,sizej);
    DistributionNormal d_normal(0,1);
    ForEachDomain2D(x,m_U){
        m_U(x)=d_normal.randomVariable();
    }
    m_U = Processing::convolutionSeperable(m_U,kernel,0,MatNBoundaryConditionPeriodic());
    m_U = Processing::convolutionSeperable(m_U,kernel,1,MatNBoundaryConditionPeriodic());
    //    m_U.display();
    //    MatN<2,ComplexF32> m_U_complex;
    //    Convertor::fromRealImaginary(m_U,m_U_complex);
    //    MatN<2,ComplexF32> m_U_fft = Representation::FFT(m_U_complex,1);
    //    std::cout<<weighcomplex_2d.getDomain()<<std::endl;
    //    std::cout<<m_U_fft.getDomain()<<std::endl;
    //    for(unsigned int i=0;i<m_U_fft.size();i++){
    //        m_U_fft(i)=m_U_fft(i)*weighcomplex_2d(i).real();
    //    }
    //    Representation::FFTDisplay(m_U_fft).display();
    //    //m_U_fft= m_U_fft.multTermByTerm(weighcomplex_2d);

    //    m_U_complex = Representation::FFT(m_U_fft,-1);
    //    Convertor::toRealImaginary(m_U_complex,m_U);

    F32 mean_value = Analysis::meanValue(m_U);
    F32 standart_deviation = Analysis::standardDeviationValue(m_U);
    std::cout<<mean_value<<std::endl;
    std::cout<<standart_deviation<<std::endl;

    m_U = (m_U-mean_value)*(1/standart_deviation);
    Mat2UI8 m_U_bin(m_U.getDomain());
    m_U_bin = Processing::threshold(m_U,beta);
    m_U_bin.display();

    m_U_bin = Processing::greylevelRemoveEmptyValue(m_U_bin);
    Mat2F32 m_corr = Analysis::correlation(m_U_bin,100);

    //m_corr = m_corr.deleteCol(1);
    for(unsigned int i= 0; i<std::min(m_corr.sizeI(),correlation.sizeI());i++){
        std::cout<<i<<" "<<(m_corr(i,1)-m_corr(0,1)*m_corr(0,1))/(m_corr(0,1)-m_corr(0,1)*m_corr(0,1))<<" "<<(correlation(i,1)-correlation(0,1)*correlation(0,1))/(correlation(0,1)-correlation(0,1)*correlation(0,1)) <<std::endl;
    }

    return m_U_bin;
}

int main()
{
    //    {
    //    Mat2UI8 img;//2d grey-level image object
    //    img.load(POP_PROJECT_SOURCE_DIR+std::string("/image/Lena.bmp"));//load the image

    //    Mat2F32 img_float(img);

    //    Mat2ComplexF32 imgcomplex(img_float);
    //    Convertor::fromRealImaginary(img_float,imgcomplex);
    //    Mat2ComplexF32 fft = Representation::FFT(imgcomplex);
    //    fft = Representation::FFT(fft,-1);
    //    Convertor::toRealImaginary(fft,img_float);
    //    img = Processing::greylevelRange(img_float,0,255);
    //    img.display();
    //    }
    F32 f=0.4f;
    F32 c=0.01f;
    F32 n=1;
    std::string f_str=pop::BasicUtility::Any2String(f);
    std::string c_str=pop::BasicUtility::Any2String(c);
    std::string n_str=pop::BasicUtility::Any2String(n);
    std::string exp = f_str+"*"+f_str+"+"+f_str+"*(1-"+f_str+")*exp(-"+c_str+"*x^"+n_str+")";
    //std::cout<<exp<<std::endl;

    DistributionExpression Corson_model(exp);
    MatN<2,F32> m_corr(64,2);
    for(unsigned int i=0;i<m_corr.sizeI();i++){
        m_corr(i,0)=i;
        m_corr(i,1)=Corson_model(i);
    }
    //    std::cout<<m_corr<<std::endl;
    gaussianField(m_corr);
    return 1;

    Mat2UI8 img;//2d grey-level image object
    img.load(POP_PROJECT_SOURCE_DIR+std::string("/image/iex.png"));
    img = PDE::nonLinearAnisotropicDiffusion(img);//filtering
    int value;
    Mat2UI8 threshold = Processing::thresholdOtsuMethod(img,value);//threshold segmentation
    threshold.save("iexthreshold.pgm");
    Mat2RGBUI8 color = Visualization::labelForeground(threshold,img);//Visual validation
    color.display();
    return 0;
}
