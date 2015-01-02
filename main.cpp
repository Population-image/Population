#include"Population.h"//Single header
using namespace pop;//Population namespace
#include"dependency/tinythread.h"







int main()
{
    //    VecN<3,F64> v(2,3,4);
    //    Mat2x<F64,3,1> m_v(v.data());
    //    Mat2x<F64,3,3> J_0 = m_v*m_v.transpose();//structure tensor
    //    VecF64 eigen_value = LinearAlgebra::eigenValue(J_0,LinearAlgebra::Symmetric);
    //    LinearAlgebra::eigenVectorGaussianElimination(J_0,eigen_value);

    //    std::cout<<eigen_value<<std::endl;
    //    std::cout<<v.norm()*v.norm()<<std::endl;
    //    return 0;
    //Mat2x<1,3,F64> m_v_transpose(v);

    {
//        Mat3UI8 img3d(2000,1000);
//        img3d.load(POP_PROJECT_SOURCE_DIR+std::string("/image/rock3d.pgm"));
        Mat2RGBUI8 m;
        m.load(POP_PROJECT_SOURCE_DIR+std::string("/image/Lena.bmp"));
        PDE::nonLinearAnisotropicDiffusion(m,200,200).display();
//        img3d = GeometricalTransformation::scale(img3d,Vec3F64(4,4,4));


//        Mat2F32 img(img3d);
//        Mat2F32 img_temp(img.getDomain());

//        //            img(2,2)=150;img(2,3)=150;img(2,4)=150;
//        //            img(3,2)=150;img(3,3)=155;img(3,4)=150;
//        //            img(4,2)=150;img(4,3)=150;img(4,4)=150;
//
//        int time1=time(NULL);
//        int iter=500;
//        for(unsigned int i=0;i<iter;i++){
//            std::cout<<i<<std::endl;
//
//            img = img_temp;
//        }
//        int time2=time(NULL);
//        std::cout<<(time2-time1)/(1.0*iter)<<std::endl;
        return 0;
//        img.display();
//        //            Mat2F64::IteratorEDomain it = img.getDomain();
//        //            std::cout<<img<<std::endl;
//        //            while(it.next()){
//        ////                std::cout<<img<<std::endl;
//        //                img_temp(it.x())=img(it.x())+diff(img,it.x());
//        //            }

//        ////            img = PDE::nonLinearDiffusion(img,40,50,1);
//        std::cout<<img_temp<<std::endl;
        return 1;
        //            std::cout<<Processing::dilation(img,1,1)<<std::endl;

        //           Vec<KeyPoint<2> > v_harris = Feature::keyPointHarris(img);
        //            Feature::drawKeyPointsCircle(img,v_harris,3).display();
        //            return 1;
    }
    //    {
    //        Mat3UI8 img;
    //        img.load(POP_PROJECT_SOURCE_DIR+std::string("/image/rock3d.pgm"));
    //         Scene3d scene;
    //         Visualization::plane(scene,img,50,2);
    //        Visualization::plane(scene,img,50,1);
    //         Visualization::plane(scene,img,200,0);
    //         Visualization::lineCube(scene,img);
    //         scene.display();

    //    }
    //    {
    //        Mat2UI8 img;
    //        img.load(POP_PROJECT_SOURCE_DIR+std::string("/image/Lena.bmp"));
    //        img = Processing::threshold(img,125);
    ////        img.display();
    //        std::cout<<Analysis::eulerPoincare(img,POP_PROJECT_SOURCE_DIR+std::string("/file/eulertab.dat"))<<std::endl;
    ////        img.display();
    //    }

    {
        //        Mat2UI8 m(50,50);
        //        m.display();
        std::string path= "D:/Users/vtariel/Downloads/";
        Mat2RGBUI8 img3;
        img3.load("/home/vincent/Desktop/images.jpeg");
        Mat2RGBUI8 img4;
        img4.load("/home/vincent/Desktop/index.jpeg");
        Vec<Mat2RGBUI8> vv;
        vv.push_back(img3);
        vv.push_back(img4);
        Mat2RGBUI8 panoimg = Feature::panoramic(vv);
        panoimg.display();
    }
    {
        Mat2UI8 img;
        img.load(POP_PROJECT_SOURCE_DIR+std::string("/image/Lena.bmp"));
        double sigma = 1.6;
        Pyramid<2,F64> pyramid1 = Feature::pyramidGaussian(img,sigma);
        Vec<KeyPointPyramid<2> > keypoint1 = Feature::keyPointSIFT(pyramid1);
        std::cout<<keypoint1.size()<<std::endl;

        Vec<Descriptor<KeyPointPyramid<2> > > descriptors1 = Feature::descriptorPyramidPieChart(pyramid1,keypoint1,sigma);

        //Apply geometrical transformation
        MatN<2,UI8> imgt(img);
        imgt = GeometricalTransformation::scale(imgt,Vec2F64(0.5));
        imgt = GeometricalTransformation::rotate(imgt,PI/6);
        Pyramid<2,F64> pyramid2 = Feature::pyramidGaussian(imgt,sigma);
        Vec<KeyPointPyramid<2> > keypoint2 = Feature::keyPointSIFT(pyramid2);
        Vec<Descriptor<KeyPointPyramid<2> > > descriptors2 = Feature::descriptorPyramidPieChart(pyramid2,keypoint2,sigma);


        Vec<DescriptorMatch<Descriptor<KeyPointPyramid<2> > >   > match = Feature::descriptorMatch(descriptors1,descriptors2);
        int nbr_math_draw = std::min((int)match.size(),30);
        match.erase(match.begin()+nbr_math_draw,match.end());
        Feature::drawDescriptorMatch(img,imgt,match,1).display();
    }
    {
        Mat2UI8 m;
        m.load(POP_PROJECT_SOURCE_DIR+std::string("/image/iex.png"));
        m = GeometricalTransformation::scale(m,Vec2F64(4,4));
        m = Processing::threshold(m,150,255);

        std::cout<<m.getDomain()<<std::endl;
        clock_t start_global, end_global;
        start_global = clock();
        Mat2UI32 cluster =Processing::clusterToLabel(m);
        end_global = clock();
        std::cout<<"cluster : "<<(double) (end_global - start_global) / CLOCKS_PER_SEC<<std::endl;

        cluster =cluster*4;

        start_global = clock();
        cluster = Processing::greylevelRemoveEmptyValue(cluster);
        end_global = clock();
        std::cout<<"greylevel : "<<(double) (end_global - start_global) / CLOCKS_PER_SEC<<std::endl;
        return 1;
    }

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
                    //                    lena = FunctorMatN::convolutionSeperable(lena,vv,0,MatNBoundaryConditionMirror());
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
        //        img = PDE::nonLinearAnisotropicDiffusionDericheFast(img);//filtering
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
