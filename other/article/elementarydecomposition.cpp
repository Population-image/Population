#include"Population.h"

using namespace pop;

Mat2UI32 poreDecompositionGrainBoundarySharpVariationGreyLevel2D(Mat2UI8 m ){

    m.save("in2d_sharpvariation.bmp");
    //GRAIN PARTITION OF THE BINARY IMAGE
    m = m.opposite();
    m.save("in2d_sharpvariation_opposite.bmp");
    //horizontal filter
    Mat2UI8 filter = Processing::smoothDeriche(m,1);
    filter.save("in2d_sharpvariation_filter_horizontal.bmp");
    filter = Processing::dynamic(filter,10);
    filter.save("in2d_sharpvariation_filter_vertical.bmp");


    Mat2UI32 markers_inside_grains = Processing::minimaRegional(filter,0);
    Visualization::labelForeground(markers_inside_grains,filter,0).save("in2d_sharpvariation_minima.bmp");


    //    Visualization::labelForeground(minima,filter,0.5).display();

    Mat2UI32 marker_outside_grains =  Processing::watershedBoundary(markers_inside_grains,filter,0);

    marker_outside_grains = Processing::threshold(marker_outside_grains,0,0);
    Visualization::labelForeground(marker_outside_grains,filter,0).save("in2d_sharpvariation_ouside.bmp");



    Mat2UI32 markers = Processing::labelMerge(marker_outside_grains,markers_inside_grains);
    Visualization::labelForeground(markers,filter,0).save("in2d_sharpvariation_inside_ouside.bmp");


    Mat2UI8 gradient =Processing::gradientMagnitudeDeriche(m,1);
    gradient.save("in2d_sharpvariation_gradient.bmp");
    Mat2UI32 watershed =Processing::watershed(markers,gradient);
    watershed = watershed -1;
    Visualization::labelForegroundBoundary(watershed,m,2).save("in2d_sharpvariation_grain.bmp");
    return watershed;
}


Mat2UI32 poreDecompositionGrainContactNarrowContact2D(Mat2UI8 m ){

    m.save("in2d_graincontact.bmp");
    //SEGMENTATION
    Mat2UI8 m_filter = PDE::nonLinearAnisotropicDiffusionDericheFast(m,20,10,2);
    m_filter.save("in2d_graincontact_filter.bmp");
    //    Draw::mergeTwoMatrixHorizontal(m, m_filter).display();
    double value;
    Mat2UI8 m_grain_binary =  Processing::thresholdOtsuMethod(m_filter,value);
    m_grain_binary.save("in2d_graincontact_threshold.bmp");

    //    Draw::mergeTwoMatrixHorizontal(m,m_grain_binary).display();
    //GRAIN PARTITION OF THE BINARY IMAGE
    //create the distunce function
    Mat2UI8 m_pore = m_grain_binary.opposite();
    Mat2UI16 m_dist = Processing::distanceEuclidean(m_pore)*4;
    m_dist = m_dist.opposite()-20;

    Mat2UI16 m_dist_disp(m_dist);
    ForEachDomain2D(x,m_dist_disp){
        if(m_grain_binary(x)==0)
            m_dist_disp(x)=0;
    }
    Visualization::labelToRandomRGB( m_dist_disp).save("in2d_graincontact_distance.bmp");


    //dynamic filter to avoid over-partition
    m_dist = Processing::dynamic(m_dist,4,0);

    m_dist_disp = m_dist;
    ForEachDomain2D(xx,m_dist_disp){
        if(m_grain_binary(xx)==0)
            m_dist_disp(xx)=0;
    }
    Visualization::labelToRandomRGB(m_dist_disp).save("in2d_graincontact_dynamic.bmp");
    //regional minima with with the norm-0 (the norm is here important with norm-1 ->over-partition)
    Mat2UI32 m_seed = Processing::minimaRegional(m_dist,0);

    Visualization::labelForeground(Processing::dilation(m_seed,1,0),m_grain_binary,0).save("in2d_graincontact_minima.bmp");


    //watershed ytansformation with the seeds as minima of the distunce function, with the topographic surface as the distunce function, and the growing region is restricted by a mask function as the granular phase
    Mat2UI32 m_grain_labelled = Processing::watershed(m_seed,m_dist,m_grain_binary,0);


    //foreground the grain label on the initial image for a visual checking
    Visualization::labelForegroundBoundary(m_grain_labelled,m,2).save("in2d_graincontact_grain.bmp");


    return m_grain_labelled;

}
Mat2UI32 poreDecompositionMixedMethod2D(Mat2UI8 m,double alpha=1 ){

    //SEGMENTATION
    Mat2UI8 m_filter = PDE::nonLinearAnisotropicDiffusionDericheFast(m,20,10,2);
    double value;
    Mat2UI8 m_grain_binary =  Processing::thresholdOtsuMethod(m_filter,value);
    //GRAIN PARTITION OF THE BINARY IMAGE
    //create the distunce function
    Mat2UI8 m_pore = m_grain_binary.opposite();

    Mat2F64 m_dist = Processing::distanceEuclidean(m_pore);
    //normalization
    m_dist =Processing::greylevelRange(m_dist,0,1);

    //
    Mat2F64 m_filter_normalization =Processing::greylevelRange(Mat2F64(m_filter.opposite()),0,1);

    m_dist = m_dist + 0.5*m_filter_normalization;

    Mat2UI8 m_dist_int = Processing::greylevelRange(m_dist,0,255);

    ForEachDomain2D(x,m_dist_int){
        if(m_grain_binary(x)==0)
            m_dist_int(x)=0;
    }
    m_dist_int = m_dist_int.opposite();
//    m_dist_int.display();
//            save("in2d_mixed_distance.bmp");

    //dynamic filter to avoid over-partition
    m_dist_int = Processing::dynamic(m_dist_int,30,0);

//    m_dist_int.display();
    //regional minima with with the norm-0 (the norm is here important with norm-1 ->over-partition)
    Mat2UI32 m_seed = Processing::minimaRegional(m_dist_int,0);
    Visualization::labelForeground(m_seed,m_dist_int,0).display("seed",false);

    //watershed ytansformation with the seeds as minima of the distunce function, with the topographic surface as the distunce function, and the growing region is restricted by a mask function as the granular phase
    Mat2UI32 m_grain_labelled = Processing::watershed(m_seed,m_dist,m_grain_binary,0);
    Visualization::labelForegroundBoundary(m_grain_labelled,m,1).display();
    //save("in2d_mixed_grain.bmp");
    return m_grain_labelled;
}


Mat3UI32 poreDecompositionGrainBoundarySharpVariationGreyLevel3D(Mat3UI8 m ){

    //GRAIN PARTITION OF THE BINARY IMAGE
    m = m.opposite();

    //horizontal filter
    Mat3UI8 filter = Processing::smoothDeriche(m,0.5);
    filter = Processing::dynamic(filter,5);
    //filter.display();
    Mat3UI32 markers_inside_grains = Processing::minimaRegional(filter,0);

    //    Visualization::labelForeground(minima,filter,0.5).display();

    Mat3UI32 marker_outside_grains =  Processing::watershedBoundary(markers_inside_grains,filter,0);


    marker_outside_grains = Processing::threshold(marker_outside_grains,0,0);

    //    Visualization::labelForeground(marker_outside_grains,filter,0.5).display();

    Mat3UI32 markers = Processing::labelMerge(marker_outside_grains,markers_inside_grains);

    Mat3UI8 gradient =Processing::gradientMagnitudeDeriche(m,2);

    Mat3UI32 watershed =Processing::watershed(markers,gradient);
    watershed = watershed-1;
    //Visualization::labelForeground(watershed,filter,0.5).display();
    return watershed;
}





Mat3UI32 poreDecompositionGrainContactNarrowContact3D(Mat3UI8 m ){


    //SEGMENTATION
    Mat3UI8 m_filter = PDE::nonLinearAnisotropicDiffusionDericheFast(m,20,10,2);
    //    Draw::mergeTwoMatrixHorizontal(m, m_filter).display();
    double value;
    Mat3UI8 m_grain_binary =  Processing::thresholdOtsuMethod(m_filter,value);

    //GRAIN PARTITION OF THE BINARY IMAGE

    //create the distunce function
    Mat3UI8 m_pore = m_grain_binary.opposite();
    Mat3UI16 m_dist = Processing::distanceEuclidean(m_pore)*4;
    m_dist = m_dist.opposite()-20;

    //dynamic filter to avoid over-partition
    m_dist = Processing::dynamic(m_dist,3,0);

    //regional minima with with the norm-0 (the norm is here important with norm-1 ->over-partition)
    Mat3UI32 m_seed = Processing::minimaRegional(m_dist,0);

    //watershed ytansformation with the seeds as minima of the distunce function, with the topographic surface as the distunce function, and the growing region is restricted by a mask function as the granular phase
    Mat3UI32 m_grain_labelled = Processing::watershed(m_seed,m_dist,m_grain_binary,0);

    return m_grain_labelled;
}
Mat3UI32 poreDecompositionMixedMethod3D(Mat3UI8 m ){


    //SEGMENTATION
    Mat3UI8 m_filter = PDE::nonLinearAnisotropicDiffusionDericheFast(m,20,10,2);
    double value;
    Mat3UI8 m_grain_binary =  Processing::thresholdOtsuMethod(m_filter,value);
    //GRAIN PARTITION OF THE BINARY IMAGE
    //create the distunce function
    Mat3UI8 m_pore = m_grain_binary.opposite();
    Mat3UI16 m_dist = Processing::distanceEuclidean(m_pore)*2;


    m_dist = (m_dist.opposite()-300) + Mat3UI16(m_filter.opposite() );
    Visualization::labelToRGBGradation(m_dist).display();

    //dynamic filter to avoid over-partition
    m_dist = Processing::dynamic(m_dist,18,0);

    //regional minima with with the norm-0 (the norm is here important with norm-1 ->over-partition)
    Mat3UI32 m_seed = Processing::minimaRegional(m_dist,0);


    //watershed ytansformation with the seeds as minima of the distunce function, with the topographic surface as the distunce function, and the growing region is restricted by a mask function as the granular phase
    Mat3UI32 m_grain_labelled = Processing::watershed(m_seed,m_dist,m_grain_binary,0);

    return m_grain_labelled;
}
int main(){
    CollectorExecutionInformationSingleton::getInstance()->setActivate(true);
    //m.getPlane(2,160);
    {
        Mat2UI8 m(2,2);
        m.display("init",false);
        m.display();
    }
    try{
        Mat3UI8 m;
#if Pop_OS==2
        std::string dir = "C:/Users/tariel/Dropbox/MyArticle/GranularSegmentation/image/SableHostun_png/";
#else
        std::string dir = "/home/vincent/Dropbox/MyArticle/GranularSegmentation/image/SableHostun_png/";
#endif
        m.loadFromDirectory(dir.c_str());
        //        Scene3d scene;
        //                Visualization::cubeExtruded(scene,m);
        //                Visualization::lineCube(scene,m);
        //                Visualization::axis(scene);
        //                scene.display();

        //        {
        //            Mat2UI8 plane = m.getPlane(0,120);
        //            Mat2UI32 grain = poreDecompositionGrainBoundarySharpVariationGreyLevel2D(plane);

        //            Visualization::labelForegroundBoundary(grain,plane).display();
        //        }
        //        {
        //            Scene3d scene;
        //            Mat3UI32 grain = poreDecompositionGrainBoundarySharpVariationGreyLevel3D(m);
        //            Mat3RGBUI8 m_grain_labelled_color = Visualization::labelToRandomRGB(grain);
        //            ForEachDomain3D(x,m_grain_labelled_color){
        //                if(x(0)<m_grain_labelled_color.getDomain()(0)/2&&x(1)<m_grain_labelled_color.getDomain()(1)/2&&x(2)>=m_grain_labelled_color.getDomain()(2)/2)
        //                    m_grain_labelled_color(x) = 0;

        //            }
        //            Visualization::marchingCube(scene,m_grain_labelled_color);
        //            Visualization::lineCube(scene,m_grain_labelled_color);
        //            Visualization::axis(scene);
        //            scene.display();
        //        }

        {
//            Mat2UI8 plane = m.getPlane(0,120);
//            Mat2UI32 grain = poreDecompositionGrainContactNarrowContact2D(plane);
//            Visualization::labelForegroundBoundary(grain,plane).display();
        }

        //        {

        //            Mat3UI32 grain = poreDecompositionGrainContactNarrowContact3D(m);
        //            Mat3RGBUI8 m_grain_labelled_color = Visualization::labelToRandomRGB(grain);
        //            ForEachDomain3D(x,m_grain_labelled_color){
        //                if(x(0)<m_grain_labelled_color.getDomain()(0)/2&&x(1)<m_grain_labelled_color.getDomain()(1)/2&&x(2)>=m_grain_labelled_color.getDomain()(2)/2)
        //                    m_grain_labelled_color(x) = 0;

        //            }
        //            Scene3d scene;
        //            Visualization::marchingCube(scene,m_grain_labelled_color);
        //            Visualization::lineCube(scene,m_grain_labelled_color);
        //            Visualization::axis(scene);
        //            scene.display();
        //        }
        {

            Mat2UI8 plane = GeometricalTransformation::plane(m,120);
            Mat2UI32 grain = poreDecompositionMixedMethod2D(plane,1);
        }
        //        {
        //            Mat3UI32 grain = poreDecompositionMixedMethod3D(m);
        //            Mat3RGBUI8 m_grain_labelled_color = Visualization::labelToRandomRGB(grain);
        //            ForEachDomain3D(x,m_grain_labelled_color){
        //                if(x(0)<m_grain_labelled_color.getDomain()(0)/2&&x(1)<m_grain_labelled_color.getDomain()(1)/2&&x(2)>=m_grain_labelled_color.getDomain()(2)/2)
        //                    m_grain_labelled_color(x) = 0;

        //            }
        //            Scene3d scene;
        //            Visualization::marchingCube(scene,m_grain_labelled_color);
        //            Visualization::lineCube(scene,m_grain_labelled_color);
        //            Visualization::axis(scene);
        //            scene.display();
        //        }




        //        poreDecompositionGrainBoundary3D(m);
        //         poreDecompositionGrainBoundary(m.getPlane(2,0));

        //Crop a small cube for fast prototyping
        //        Vec3I32 x1,x2;
        //        x1(0)=0;x1(1)=0;x1(2)=0;
        //        x2(0)=150;x2(1)=150;x2(2)=150;
        //        //comment this line to process the full 3d image
        //        m = m(x1,x2);


        //        m = m.opposite();
        //        Mat3UI8 filter = Processing::smoothDeriche(m,2);
        //        filter = Processing::dynamic(filter,20);
        //        Mat3UI16 minima = Processing::minimaRegional(filter);
        ////        Visualization::labelForeground(minima,m).display();
        //        Mat3UI16 water =Processing::watershedBoundary(minima,filter,1);
        //        Mat3UI16 boundary = Processing::threshold(water,0,0);//the boundary label is 0
        ////        boundary.display("boundary",true,false);
        //        minima = Processing::labelMerge(boundary,minima);
        //        Mat3UI8 gradient = Processing::gradientMagnitudeDeriche(m,1);
        //        water = Processing::watershed(minima,gradient);
        //        water = water -1;
        //        Visualization::labelForeground(water,m,0.2).display();



    }
    catch(const pexception &e){
        e.display();//Display the error in a window
    }
}
