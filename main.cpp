#include"Population.h"//Single header
using namespace pop;//Population namespace
#include"chrono"
template<typename Type1,typename Type2,typename FunctorBinaryFunctionE>
void forEachFunctorBinaryFunctionE2(const MatN<2,Type1> & f, MatN<2,Type2> &  h,  FunctorBinaryFunctionE func, typename MatN<2,Type1>::IteratorERectangle it)
{
    int i,j;
#if defined(HAVE_OPENMP)
#pragma omp parallel shared(f,h) private(i,j) firstprivate(func)
#endif
    {
#if defined(HAVE_OPENMP)
#pragma omp for schedule (static)
#endif
        for(i=it.xMin()(0);i<it.xMax()(0);i++){
            for(j=it.xMin()(1);j<it.xMax()(1);j++){
                h(Vec2I32(i,j))=func( f, Vec2I32(i,j));
            }
        }
    }
}

struct __FunctorNiblackMethod
{
    const Mat2UI32* _integral;
    const Mat2UI32* _integral_power_2;
    F32 area_minus1;
    F32 _k;
    int _radius;
    F32 _offset_value;
    __FunctorNiblackMethod(const Mat2UI32 & integral,const Mat2UI32& integral_power_2,F32 _area_minus1,F32 k,int radius,F32 offset_value)
        :_integral(&integral),_integral_power_2(&integral_power_2),area_minus1(_area_minus1),_k(k),_radius(radius),_offset_value(offset_value){

    }

    template<typename PixelType>
    UI8 operator()(const MatN<2,PixelType > & f,const  typename MatN<2,PixelType>::E & x){
        Vec2I32 xadd1=x+Vec2I32(_radius);
        Vec2I32 xadd2=x+Vec2I32(-_radius);
        Vec2I32 xsub1=x-Vec2I32(_radius,-_radius);
        Vec2I32 xsub2=x-Vec2I32(-_radius,_radius);
        F32 mean =(F32) (*_integral)(xadd1)+(*_integral)(xadd2)-(*_integral)(xsub1)-(*_integral)(xsub2);
        mean*=area_minus1;

        F32 standartdeviation =(*_integral_power_2)(xadd1)+(*_integral_power_2)(xadd2)-(*_integral_power_2)(xsub1)-(*_integral_power_2)(xsub2);
        standartdeviation*=area_minus1;
        standartdeviation =standartdeviation-mean*mean;

        if(standartdeviation>0)
            standartdeviation = std::sqrt( standartdeviation);
        else
            standartdeviation =1;
        if(f(x-_radius)>ArithmeticsSaturation<PixelType,F32>::Range( mean+_k*standartdeviation)-_offset_value)
            return 255;
        else
            return  0;
    }
};
template<typename PixelType>
static MatN<2,UI8>  thresholdNiblackMethod(const MatN<2,PixelType> & f,F32 k=0.2,int radius=5,F32 offset_value=0  ){
    MatN<2,PixelType> fborder(f);
    Draw::addBorder(fborder,radius,typename MatN<2,PixelType>::F(0),MATN_BOUNDARY_CONDITION_MIRROR);
    MatN<2,UI32> f_F32(fborder);
    MatN<2,UI32> integral = Processing::integral(f_F32);
    MatN<2,UI32> integralpower2 = Processing::integralPower2(f_F32);
    typename MatN<2,UI32>::IteratorERectangle it(fborder.getIteratorERectangle(Vec2I32(radius),f_F32.getDomain()-1-Vec2I32(radius)));

    F32 area_minus1 = 1.f/((2*radius+1)*(2*radius+1));
    __FunctorNiblackMethod func(integral,integralpower2,area_minus1,k, radius, offset_value);
    forEachFunctorBinaryFunctionE2(f,fborder,func,it);
    return fborder( Vec2I32(radius) , fborder.getDomain()-Vec2I32(radius));
}

int main()
{

    omp_set_num_threads(6);
    pop::Mat2UI8 m(1200,1600);
    auto start_global = std::chrono::high_resolution_clock::now();

    m=thresholdNiblackMethod(m);
    auto end_global = std::chrono::high_resolution_clock::now();
    std::cout<<"processing nimblack1 : "<<std::chrono::duration<double, std::milli>(end_global-start_global).count()<<std::endl;

    int time1 = time(NULL);
     start_global = std::chrono::high_resolution_clock::now();

    m=Processing::thresholdNiblackMethod(m);
     end_global = std::chrono::high_resolution_clock::now();
    std::cout<<"processing nimblack2 : "<<std::chrono::duration<double, std::milli>(end_global-start_global).count()<<std::endl;

    return 1;

    //	m = m*m;
    int time2 = time(NULL);
    std::cout<<time2-time1<<std::endl;
    Mat2UI8 img;//2d grey-level image object
    img.load(POP_PROJECT_SOURCE_DIR+std::string("/image/iex.png"));//replace this path by those on your computer
    img = PDE::nonLinearAnisotropicDiffusion(img);//filtering
    int value;
    Mat2UI8 threshold = Processing::thresholdOtsuMethod(img,value);//threshold segmentation
    threshold.save("iexthreshold.pgm");
    Mat2RGBUI8 color = Visualization::labelForeground(threshold,img);//Visual validation
    color.display();
    return 0;
}
