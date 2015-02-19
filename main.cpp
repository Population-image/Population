#include"Population.h"//Single header
using namespace pop;//Population namespace

DistributionRegularStep generateProbabilitySpectralDensity(const Mat2F32& correlation)
{
    std::string str_cummulative_gausssian = "1/((2*pi)^(0.5))*exp(-(x^2)*0.5)";
    DistributionExpression f_beta(str_cummulative_gausssian);
    std::cout<<str_cummulative_gausssian<<std::endl;
    //TODO perhaps 1- correlation(0,1)
    F32 beta= Statistics::maxRangeIntegral(f_beta,1-correlation(0,1),-4,4,0.001);
//    std::cout<<Statistics::integral(f_beta,-4.,4,0.001f)(beta)<<std::endl;
//    std::cout<<1-correlation(0,1)<<std::endl;
//    exit(0);
    MatN<1,F32> autocorrelation;
    autocorrelation.resize(VecN<1,int>(correlation.sizeI()));
    for(unsigned int i= 0; i<correlation.sizeI();i++)
    {
        autocorrelation(i)=correlation(i,1)-correlation(0,1)*correlation(0,1);

    }
    std::cout<<autocorrelation<<std::endl;
    std::string s = BasicUtility::Any2String(beta);
    std::string  equation= "1/(2*pi*(1-x^2)^0.5)*exp(-"+s+"^2/(1+x))";

    std::cout<<equation<<std::endl;

    DistributionExpression f(equation.c_str());
    DistributionRegularStep fintegral = Statistics::integral(f,0,1.2,0.001f);
    MatN<1,F32> rho;
    rho.resize(VecN<1,int>(correlation.sizeI()));
    for(unsigned int i= 0; i<correlation.sizeI();i++) {
        if(autocorrelation(i)>=0){
            double value=std::min(0.999f,autocorrelation(i));
            rho(i)=Statistics::FminusOneOfYMonotonicallyIncreasingFunction(fintegral,value,0,1.2,0.001f);
        }
        else
            rho(i)=0;
    }


    std::cout<<rho<<std::endl;
    MatN<1,ComplexF32> rhocomplex;
    Convertor::fromRealImaginary(rho,rhocomplex);
    std::cout<<rhocomplex<<std::endl;
    MatN<1,ComplexF32> fft = Representation::FFT(rhocomplex,1);
    fft =pop::squareRoot(fft);
    MatN<1,ComplexF32> weighcomplex = Representation::FFT(rhocomplex,0);
    MatN<1,F32> weigh;
    Convertor::toRealImaginary(weighcomplex,weigh);
    F32 sum=0;
    for(unsigned int i=0;i<weigh.sizeI();i++){
        sum+=weigh(i);
    }
    std::cout<<weigh<<std::endl;
    for(unsigned int i=0;i<weighcomplex.sizeI();i++){
        weigh(i)/=sum;
    }

    Convertor::fromRealImaginary(weigh,weighcomplex);
    weighcomplex = Representation::FFT(weighcomplex,1);


    int sizei=256;
    int sizej=256;
    Mat2F32 m_U(sizei,sizej);
    DistributionNormal d_normal(0,1);
    ForEachDomain2D(x,m_U){
        m_U(x)=d_normal.randomVariable();
    }
    std::cout<<Analysis::histogram(Processing::threshold(m_U,beta))(255,1)<<std::endl;


    std::cout<<weighcomplex<<std::endl;

    MatN<2,ComplexF32>  m_weight(sizei,sizej);
    ForEachDomain2D(xx,m_weight){
        F32 dist= Vec2F32(xx-m_weight.getDomain()/2).norm(2);
        if(dist<weigh.sizeI()){
            m_weight(xx)=weighcomplex(std::floor(dist));
        }
        else{
            m_weight(xx)=0;
        }
    }
    m_weight = GeometricalTransformation::translate(m_weight,-m_weight.getDomain()/2);
     Representation::FFTDisplay(m_weight).display();
    MatN<2,ComplexF32> m_U_complex;
    Convertor::fromRealImaginary(m_U,m_U_complex);
    Representation::FFTDisplay(m_U_complex).display();
    MatN<2,ComplexF32> m_U_fft = Representation::FFT(m_U_complex,1);
    m_U_fft= m_U_fft.multTermByTerm(m_weight);




//    Representation::FFTDisplay(m_weight).display();
    m_U_complex = Representation::FFT(m_U_fft,-1);
    Convertor::toRealImaginary(m_U_complex,m_U);
//    m_U.display();
    Mat2UI8 m_U_bin(m_U.getDomain());
    m_U_bin = Processing::threshold(m_U,beta);
    m_U_bin.display();

    m_U_bin = Processing::greylevelRemoveEmptyValue(m_U_bin);
    Mat2F32 m_corr = Analysis::correlation(m_U_bin,100);
    std::cout<<m_corr<<std::endl;
    for(unsigned int i= 0; i<std::min(m_corr.sizeI(),correlation.sizeI());i++){
        std::cout<<i<<" "<<m_corr(i,1)<<" "<<correlation(i,1) <<std::endl;
    }
}

int main()
{
    F32 f=0.4;
    F32 c=0.1;
    F32 n=1;
    std::string f_str=pop::BasicUtility::Any2String(f);
    std::string c_str=pop::BasicUtility::Any2String(c);
    std::string n_str=pop::BasicUtility::Any2String(n);
    std::string exp = f_str+"*"+f_str+"+"+f_str+"*(1-"+f_str+")*exp(-"+c_str+"*x^"+n_str+")";
    std::cout<<exp<<std::endl;

    DistributionExpression Corson_model(exp);
    MatN<2,F32> m_corr(100,2);
    for(unsigned int i=0;i<m_corr.sizeI();i++){
        m_corr(i,0)=i;
        m_corr(i,1)=Corson_model(i);

    }
    std::cout<<m_corr<<std::endl;
    generateProbabilitySpectralDensity(m_corr);
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
