﻿#include"Population.h"//Single header
using namespace pop;//Population namespace


Mat2UI16 Transform(Mat2UI8 m,UI8 threshold_value);
std::vector<std::pair<Vec2I32, Vec2I32> > GetLines(Mat2UI16 accu,int img_w,int img_h,  int threshold);





Mat2UI16 Transform(Mat2UI8 m,UI8 threshold_value)
{
    double DEG2RAD=0.017453293f;
    //Create the accu
    double hough_h = ((sqrt(2.0) * (double)(m.sizeI()>m.sizeJ()?m.sizeI():m.sizeJ())) / 2.0);
    double heigh = hough_h * 2.0; // -r -> +r
    double width = 180;
    Mat2UI16 accu (heigh,width);
    double center_x = m.sizeJ()/2;
    double center_y = m.sizeI()/2;
    for(int i=0;i<m.sizeI();i++)
    {
        for(int j=0;j<m.sizeJ();j++)
        {
            if( m(i,j) > threshold_value)
            {
                for(int t=0;t<180;t++)
                {
                    double r = ( (j- center_x) * cos((double)t * DEG2RAD)) + ((i - center_y) * sin((double)t * DEG2RAD));
                    accu(round(r + hough_h), t)++;
                }
            }
        }
    }
    return accu;
}

std::vector< std::pair<Vec2I32, Vec2I32 > > GetLines(Mat2UI16 accu,int img_w,int img_h,  int threshold)
{
    std::vector< std::pair<Vec2I32, Vec2I32 > > lines;
    double DEG2RAD=0.017453293f;
    Mat2UI16::IteratorENeighborhood it=accu.getIteratorENeighborhood(4,0);

    ForEachDomain2D(x,accu){
        if(accu(x) >= threshold){
            it.init(x);
            double value=accu(it.x());
            bool max_local=true;
            while(it.next()){
                if(accu(it.x())>value){
                    max_local=false;
                    break;
                }
            }

            if(max_local==true){
                std::cout<<"line "<<x<<std::endl;
                Vec2I32 x1,x2;
                double radius  = x(0);
                double angle   = x(1);
                //                int accu.sizeI();
//                if(x(1) >= 45 && x(1) <= 135){//y = (r - x cos(t)) / sin(t)

//                    x1(1) = 0;
//                    x1(0) = ((double)(x(0)-(accu.sizeI()/2)) - ((x1(1) - (img_w/2) ) * cos(x(1) * DEG2RAD))) / sin(x(1) * DEG2RAD) + (img_h / 2);
//                    x2(1) = img_w - 0;
//                    x2(0) = ((double)(x(0)-(accu.sizeI()/2)) - ((x2(1) - (img_w/2) ) * cos(x(1) * DEG2RAD))) / sin(x(1) * DEG2RAD) + (img_h / 2);
//                }
//                else{//x = (r - y sin(t)) / cos(t);

                    double value1=80;
                    x1(0) = (-cos(angle* DEG2RAD)*value1+ radius-accu.sizeI()/2)/sin(angle* DEG2RAD)+img_h/2;
                    x1(1) = value1 + img_w/2;
                    x2(0) = (-cos(angle* DEG2RAD)*(-value1)+ radius-accu.sizeI()/2)/sin(angle* DEG2RAD)+img_h/2;
                    x2(1) = (-value1) + img_w/2;
//                }
//                std::cout<<x1<<std::endl;
//                std::cout<<x2<<std::endl;
                    lines.push_back(std::make_pair(x1,x2));
//                        return lines;
            }
        }
    }
    return lines;
}

//const unsigned int* Hough::GetAccu(int *w, int *h)
//{
//    *w = _accu_w;
//    *h = _accu_h;
//    return _accu;
//}




int main(){
    {
        Mat2UI8 m;
        m.load("/home/vincent/Desktop/Hough-example-result-en.png");
        m = Processing::erosion(m,1);
//        m.display();
        //m= Processing::edgeDetectorCanny(m,1,2,20);//.display("canny",true,false);
        Mat2UI32 accu = Transform(m,50);
//       accu.display();
        std::vector< std::pair<Vec2I32, Vec2I32 > > v_lines = GetLines(accu,m.sizeJ(),m.sizeI() ,100);

        Mat2UI8 m_hough(m);
        std::cout<<m.getDomain()<<std::endl;
        for(unsigned int i=0;i<v_lines.size();i++){
            Draw::line(m_hough,v_lines[i].first,v_lines[i].second,  100,2);
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
