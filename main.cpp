#include"Population.h"//Single header
using namespace pop;//Population namespace





template<typename PixelType>
Mat2F64 transformHough(MatN<2,PixelType> m,typename MatN<2,PixelType>::F threshold_value)
{
    double DEG2RAD=0.017453293f;
    //Create the accu
    double hough_h = ((sqrt(2.0) * (double)(m.sizeI()>m.sizeJ()?m.sizeI():m.sizeJ())) / 2.0);
    int heigh = hough_h * 2.0; // -r -> +r
    int width = 180;
    Mat2F64 accu (heigh,width);
    double center_x = m.sizeJ()/2;
    double center_y = m.sizeI()/2;
    for(int i=0;i<m.sizeI();i++)
    {
        for(int j=0;j<m.sizeJ();j++)
        {
            if( m(i,j) > threshold_value)
            {
                for(double t=0;t<180;t++)
                {
                    double r = ( (j- center_x) * cos((double)t * DEG2RAD)) + ((i - center_y) * sin((double)t * DEG2RAD));
                    VecN<4,std::pair<F64,Vec2I32> > v_hit = interpolationBilinearWeigh(accu.getDomain(),Vec2F64(r+hough_h,t));
                    for(unsigned int i=0;i<4;i++){
                        accu(v_hit(i).second)+=v_hit(i).first;
                    }

                }
            }
        }
    }
    accu = Processing::smoothGaussian(accu,1);
    return Processing::greylevelRange(accu,0,1);
}

std::vector< std::pair<Vec2I32, Vec2I32 > > GetLines(Mat2F64 accu,Mat2UI8 img,  double threshold)
{
    std::vector< std::pair<Vec2I32, Vec2I32 > > lines;
    double DEG2RAD=0.017453293f;
    Mat2F64::IteratorENeighborhood it=accu.getIteratorENeighborhood(4,0);
    ForEachDomain2D(x,accu){
        Vec2I32 xpoint =x;
        if(accu(x) >= threshold){
            it.init(x);
            double value=accu(x);
            bool max_local=true;
            while(it.next()){
                if(accu(it.x())>value){
                    max_local=false;
                    break;
                }
            }
            if(max_local==true){
                accu(x)=0;
                Vec2I32 x1,x2;
                double radius  = x(0);
                double angle   = x(1);
                double value1=img.sizeJ()/2;
                x1(0) = (-cos(angle* DEG2RAD)*value1+ radius-accu.sizeI()/2)/sin(angle* DEG2RAD)+img.sizeI()/2;
                x1(1) = value1 + img.sizeJ()/2;
                x2(0) = (-cos(angle* DEG2RAD)*(-value1)+ radius-accu.sizeI()/2)/sin(angle* DEG2RAD)+img.sizeI()/2;
                x2(1) = (-value1) + img.sizeJ()/2;
                lines.push_back(std::make_pair(x1,x2));
            }
        }
    }
//    accu.display();
    return lines;
}




template<int DIM>
void addValue(const MatN<DIM,UI8>&m){
   std::cout<<interpolationBilinearWeigh(m.getDomain(),VecN<DIM,F64>())<<std::endl;
}

int main(){
    {
        Mat2UI8 m;
        m.load(POP_PROJECT_SOURCE_DIR+std::string("/image/Lena.bmp"));
        std::cout<<(int)Analysis::maxValue(m)<<std::endl;
        m = Processing::greylevelRange(m,0,255);
        std::cout<<(int)Analysis::maxValue(m)<<std::endl;
    }
    {
//        Mat3UI8 m3d(3,2,4);
//        VecN<8,std::pair<double,Vec3I32 > > v= interpolationBilinearWeigh(Vec3F64(0.2,0.8,0.9),m3d.getDomain());
//        std::cout<<v<<std::endl;
//        addValue(m3d);
////        std::cout<<interpolationBilinearWeigh(Vec2F64(2.7,0.2),m.getDomain())<<std::endl;
//        return 0;
        Mat2UI8 m;
        addValue(m);
        m.load("/home/vincent/Desktop/Hough-example-result-en.png");



//        m.display();
//        m= GeometricalTransformation::scale(m,Vec2F64(0.1,0.1),1);
//        m.display();
        m.load("/home/vincent/Desktop/_.png");
        Mat2UI8 mm=m;
//        m.display();
//        m = Processing::erosion(m,1);
        //        m.display();
        //m
        m= Processing::edgeDetectorCanny(m,2,1,10);//.display("canny",true,false);
        Mat2F64 accu = transformHough(m,50);
//        accu.display();
        std::vector< std::pair<Vec2I32, Vec2I32 > > v_lines = GetLines(accu,m ,0.6);

        Mat2UI8 m_hough(mm);
        std::cout<<m.getDomain()<<std::endl;
        for(unsigned int i=0;i<v_lines.size();i++){
            Draw::line(m_hough,v_lines[i].first,v_lines[i].second,  255,2);
        }
        m_hough.display();


        m.display();

        //        hough.Transform(m.data(),m.sizeJ(),m.sizeI());
        //        for(unsigned int i=100;i<100;i++){
        //            std::vector< std::pair< std::pair<int, int>, std::pair<int, int> > > lines = hough.GetLines(i*2);
        //            std::vector< std::pair< std::pair<int, int>, std::pair<int, int> > >::iterator it;
        //
        //            for(it=lines.begin();it!=lines.end();it++)
        //            {
        //
        //            }
        //            m_hough.display();
        //            std::cout<<i<<" "<<lines.size()<<std::endl;
        //        }
        return 1;

    }
    //hough.Transform()
    Mat2RGBUI8 m(2,2);
    m(1,1)=RGBUI8(255,10,10);
    Mat2UI8 m1;
    m1 = m;


    return 1;

    NeuralNetworkFeedForward n;
    TrainingNeuralNetwork::neuralNetworkForRecognitionForHandwrittenDigits(n,"/home/vincent/train-images.idx3-ubyte",
                                                                           "/home/vincent/train-labels.idx1-ubyte",
                                                                           "/home/vincent/t10k-images.idx3-ubyte",
                                                                           "/home/vincent/t10k-labels.idx1-ubyte",1);
    Mat2UI8 img;
    img.load(POP_PROJECT_SOURCE_DIR+std::string("/image/iex.png"));//replace this path by those on your computer
    img.display("Initial image",false);
    img = PDE::nonLinearAnisotropicDiffusion(img);//filtering
    img.display();
    double value;
    Mat2UI8 threshold = Processing::thresholdOtsuMethod(img,value);//threshold segmentation
    threshold.save("iexthreshold.png");
    Mat2RGBUI8 color = Visualization::labelForeground(threshold,img);//Visual validation
    color.display("Segmented image",true);
    return 0;
}
