#include"Population.h"

using namespace pop;





int main(){
    CollectorExecutionInformationSingleton::getInstance()->setActivate(true);
    try{
        //    Mat3UI8 m("_.pgm");
        //    double slice =5;
        //    m = m(Vec3I32(0,0,0),Vec3I32(150,150,25));
        //    m.getPlane(2,slice).display("initial",false);
        //    MatNIteratorENeighborhoodAmoebas2<Mat3UI8> it_local2(m,4,0.01);
        //    Mat3UI8::IteratorENeighborhoodAmoebas  it_local = m.getIteratorENeighborhoodAmoebas(4,0.01);
        //    Mat3UI8::IteratorEDomain it_global = m.getIteratorEDomain();
        //    Mat3UI8 m_median = ProcessingAdvanced::median(m,it_global,it_local);
        //    m_median.getPlane(2,slice).display("median anoaba",false);

        //    it_global.init();
        //    Mat3UI8 m_median_amoeba = ProcessingAdvanced::median(m,it_global,it_local2);
        //    m_median_amoeba.getPlane(2,slice).display("median anoaba real",false);


        ////    Mat3UI8 m_median2 = Processing::median(m,2);
        ////    m_median2.getPlane(2,slice).display("median2",false);

        ////    Mat3UI8 m_non_linear = PDE::nonLinearAnisotropicDiffusionDericheFast(m,10,10,1.5);
        ////    m_non_linear.getPlane(2,slice).display("non linear",false);
        //    double value;
        //    Processing::thresholdOtsuMethod(m_median,value).getPlane(2,slice).display("threshold anoaba",false);

        //    Processing::thresholdOtsuMethod(m_median_amoeba,value).getPlane(2,slice).display("threshold anoaba real");
        ////    Processing::thresholdOtsuMethod(m_non_linear,value).getPlane(2,slice).display("non linear",false);
        ////    Processing::thresholdOtsuMethod(m_median2,value).getPlane(2,slice).display("threshold fixed kernel");


        Mat3UI8 m2("_.pgm");
        double slice =5;
        Mat2UI8 mm =  m2.getPlane(2,slice);
        int i=2;
        MatNDisplay disp, disp1,disp2;
        while(1==1)
        {
            Mat2UI8 m(mm.getDomain());
            DistributionNormal d(0,i);
            i+=2;
            i=i%100;
            std::cout<<i<<std::endl;
            ForEachDomain(x,m){
                m(x)=ArithmeticsSaturation<UI8,F64>::Range(d.randomVariable()+mm(x));
            }


            disp.display((GeometricalTransformation::scale(m,Vec2F64(4,4))));


            MatNIteratorENeighborhoodAmoebas2<Mat2UI8> it_local2(m,5,0.02);
            Mat2UI8::IteratorENeighborhoodAmoebas  it_local = m.getIteratorENeighborhoodAmoebas(5,0.01);
            i+=0.1;

            Mat2UI8::IteratorEDomain it_global = m.getIteratorEDomain();
            Mat2UI8 m_median = ProcessingAdvanced::median(m,it_global,it_local);
            //        m_median.display("median anoaba");

            it_global.init();
            Mat2UI8 m_median_amoeba = ProcessingAdvanced::median(m,it_global,it_local2);
            disp2.set_title("Classical");
            disp1.display(GeometricalTransformation::scale(m_median,Vec2F64(4,4)));
            disp2.display(GeometricalTransformation::scale(m_median_amoeba,Vec2F64(4,4)));
            //        disp2.display(m_median);
            //        m_median_amoeba.display("median anoaba real",false);
            //        Mat2UI8 m_median2 = Processing::median(m,2);
            //        m_median2.getPlane(2,slice).display("median2",false);

            //        Mat2UI8 m_non_linear = PDE::nonLinearAnisotropicDiffusionDericheFast(m,10,10,1.5);
            //        m_non_linear.getPlane(2,slice).display("non linear",false);
            //        double value;
            //        Processing::thresholdOtsuMethod(m_median,value).display("threshold anoaba",false);

            //        Processing::thresholdOtsuMethod(m_median_amoeba,value).display("threshold anoaba real");
            //        Processing::thresholdOtsuMethod(m_non_linear,value).getPlane(2,slice).display("non linear",false);
            //        Processing::thresholdOtsuMethod(m_median2,value).getPlane(2,slice).display("threshold fixed kernel");
        }
        return 1;
        return 1;
        //    Scene3d scene;
        //    Mat3UI8 extruded(img.getDomain());
        //    int radius=img.getDomain()(0)/2;
        //    Vec3I32 x1(0,0,0);
        //    Vec3I32 x2(img.getDomain());
        //    ForEachDomain3D(x,extruded){
        //        if((x-x1).norm(2)<radius||(x-x2).norm(2)<radius)
        //            extruded(x)=0;
        //        else
        //            extruded(x)=255;
        //    }
        //    Visualization::cubeExtruded(scene,img,extruded);//add the cube surfaces to the scene
        //    Visualization::lineCube(scene,img);//add the border red lines to the scene to the scene
        //    Visualization::axis(scene,40);//add axis
        //    scene.display();//display the scene



        //    MatNDisplay disp;
        //    //while(disp.is)
        //    ForEachDomain(x,m){
        //        Mat2UI8 m_temp(m);
        //        it.init(x);
        //        while(it.next()){
        //            m_temp(it.x())=255;
        //        }
        //        disp.display();

        //    }
        //    FunctorGradient
        //    PopulationInformation
        //    Processing::thresholdOtsuMethod(m,value).display();
        //    m.display();
        //    Processing::m

        //    Population<pop::Mat2UI16> pop;

        //    return 1;

    }
    catch(const pexception &e){
        e.display();//Display the error in a window
    }
}
