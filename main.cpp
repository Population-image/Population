#include"Population.h"//Single header
using namespace pop;//Population namespace


int main()
{
    try{//Enclose this portion of code in a try block
        {
            {
//                std::cout<<createGaussianKernelOneDimension(2,5)<<std::endl;
//                std::cout<<createGaussianKernelMultiDimension<2>(1,2)<<std::endl;
//                return 1;

//                Mat2UI8 lena;
//                lena.load(POP_PROJECT_SOURCE_DIR+std::string("/image/iex.png"));
//                Mat2F64 d(7,7);
//                d(3,3)=1;
////                FunctorMatN::GaussianKernel<Mat2F64> g(d,2,3);
//                Mat2F64::IteratorEDomain it = d.getIteratorEDomain();
//                //std::cout<<FunctorMatN::convolutionSeperable(d,kernel_derivate,0,it,MatNBoundaryConditionMirror())<<std::endl;
//                Mat2F64 out2 = FunctorMatN::convolutionGaussianDerivate(d,0,2,4);

//                std::cout<<out2<<std::endl;
////                std::cout<<out2<<std::endl;
//                //                out = out*30000;
//                //                Scene3d scene;
//                //                Visualization::topography(scene,out);
//                //                scene.display();
//                return 1;
            }
            {

                Mat2UI8 img;
                img.load(POP_PROJECT_SOURCE_DIR+std::string("/image/iex.png"));
                Draw::addBorder(img,50,200,MATN_BOUNDARY_CONDITION_MIRROR);
                img.display();

                Draw::text(img,"TOTO",Vec2I32(20,20),255,5);
                img.display();
                double d[]=
                {
                    0.5, 1, 0.5,
                    1  , 2,   1,
                    0.5, 1, 0.5
                };
                Mat2F64 kernel(Vec2I32(3,3),d);
                std::cout<<normValue(kernel,1)<<std::endl;
                kernel = kernel/normValue(kernel,1);
                std::cout<<kernel<<std::endl;
                Mat2RGBUI8 lena;
                lena.load(POP_PROJECT_SOURCE_DIR+std::string("/image/lena.bmp"));
                MatNDisplay ddisp;
                MatNIteratorEDomain<Vec2I32> itlena = lena.getIteratorEDomain();
                clock_t start_global, end_global;
                start_global = clock();
                for(int i=0;i<10;i++){
                    itlena.init();
                    lena = Processing::convolution(lena,kernel,MatNBoundaryConditionMirror());
                    //ddisp.display(lena);
                }
                end_global = clock();
                std::cout<<"grad : "<<(double) (end_global - start_global) / CLOCKS_PER_SEC<<std::endl;
                VecF64 vv(10);
                vv(0)=0.25;vv(1)=0.5;vv(2)=0.25;
                //                start_global = clock();
                //                for(int i=0;i<10;i++){
                //                    //itlena.init();
                //                   lena = FunctorMatN::FunctorConvolution::seperableConvolution(lena,vv,0);
                //                   //ddisp.display(lena);
                //                }
                //                end_global = clock();
                std::cout<<"grad : "<<(double) (end_global - start_global) / CLOCKS_PER_SEC<<std::endl;
                start_global = clock();
                for(int i=0;i<10;i++){
                    //itlena.init();
                    lena = FunctorMatN::convolutionSeperable(lena,vv,0,MatNBoundaryConditionMirror());
                    //ddisp.display(lena);
                }
                end_global = clock();
                std::cout<<"grad : "<<(double) (end_global - start_global) / CLOCKS_PER_SEC<<std::endl;


                lena.display();
                return 1;

            }
            Mat2UI8 img;//2d grey-level image object


            clock_t start_global, end_global;

            img.load(POP_PROJECT_SOURCE_DIR+std::string("/image/iex.png"));//replace this path by those on your computer
            img = GeometricalTransformation::scale(img,Vec2F64(10,10));
            start_global = clock();
            Mat2UI8::IteratorEDomain it = img.getDomain();
//            Private::ConvolutionSeparableMirror<2> kernel;
            VecF64 v(7);
            v(0)=-1;
            v(1)=0;
            v(2)=1;
            //Mat2F64 m(3,3);
            //m(1,1)=1;
//            kernel.setSingleKernel(v);
            MatNDisplay disp;
            //
            //        while(1==1){
//            img = kernel.operator ()(0,img,Vec2I32(0,0),Vec2I32(0,0));

            //            img = convolution1D(img,v,0);
            //        disp.display(img);
            //                    }
            std::cout<<img.getDomain()<<std::endl;
            //            img = Processing::gradientMagnitudeSobel(img);
            end_global = clock();
            std::cout<<"grad : "<<(double) (end_global - start_global) / CLOCKS_PER_SEC<<std::endl;
            return 1;

            Mat2ComplexF64 imgcomplex;
            Convertor::fromRealImaginary(Mat2F64(img),imgcomplex);
            Mat2ComplexF64 fft = Representation::FFT(imgcomplex);
            Mat2UI8 filterlowpass(fft.getDomain());
            Vec2I32 x(0,0);
            Draw::disk(filterlowpass,x,20,UI8(255),MATN_BOUNDARY_CONDITION_PERIODIC);
            fft = Processing::mask(fft,filterlowpass);
            //Representation::FFTDisplay(fft).display();
            imgcomplex = Representation::FFT(fft,-1);
            Mat2F64 imgd;
            Convertor::toRealImaginary(imgcomplex,imgd);
            Mat2UI8 filter = Processing::greylevelRange(imgd,0.,255.);
            filter.save("/home/vincent/Population/doc/image/iexlowpass.jpg");
            double threshold_automatic;
            Mat2UI8 threshold = Processing::thresholdOtsuMethod(filter,threshold_automatic);
            Visualization::labelForegroundBoundary(threshold,img,2).save("/home/vincent/Population/doc/image/iexseglowpass.jpg");
        }

        Mat2UI8 img;
        img.load(POP_PROJECT_SOURCE_DIR+std::string("/image/iex.png"));//replace this path by those on your computer
        img.display("Initial image",false);
        img = PDE::nonLinearAnisotropicDiffusionDericheFast(img);//filtering
        double value;
        Mat2UI8 threshold = Processing::thresholdOtsuMethod(img,value);//threshold segmentation
        threshold.save("iexthreshold.png");
        Mat2RGBUI8 color = Visualization::labelForeground(threshold,img);//Visual validation
        color.display("Segmented image",true);
    }
    catch(const pexception &e){
        e.display();//Display the error in a window
    }
    return 0;
}
