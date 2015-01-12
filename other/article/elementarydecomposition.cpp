#include"Population.h"

using namespace pop;


template<int DIM>
MatN<DIM,UI32> poreDecompositionMixedMethod(MatN<DIM,UI8> m,F32 alpha=1 ){

    //###SEGMENTATION###
    MatN<DIM,UI8> m_filter = PDE::nonLinearAnisotropicDiffusion(m,20,10);
    double value;
    MatN<DIM,UI8> m_grain_binary =  Processing::thresholdOtsuMethod(m_filter,value);

    //###GRAIN PARTITION###
    //euclidean distunce map
    MatN<DIM,UI8> m_pore = m_grain_binary.opposite();
    MatN<DIM,F32> m_dist = Processing::distanceEuclidean(m_pore);

    //normalization in [0,1]
    m_dist =Processing::greylevelRange(m_dist,0,1);
    MatN<DIM,F32> m_filter_normalization =Processing::greylevelRange(MatN<DIM,F32>(m_filter),0,1);

    //linear combination
    m_dist = m_dist + alpha*m_filter_normalization;

    //convert float to integer needed by the watershed and dynamic algorithms
    MatN<DIM,UI8> m_dist_int = Processing::greylevelRange(m_dist,0,255);

    //set at 0 the points outside the grain space
    typename MatN<DIM,UI8>::IteratorEDomain it = m_dist_int.getIteratorEDomain();
    while(it.next()){
        if(m_grain_binary(it.x())==0)
            m_dist_int(it.x())=0;
    }

    //opposite to have the watershed surface on the grain boundaries
    m_dist_int = m_dist_int.opposite();

    //filter the over-seeding with vertical filter
    m_dist_int = Processing::dynamic(m_dist_int,10,0);

    //regional minima with with the norm-0 (the norm is here important with norm-1 ->over-partition)
    MatN<DIM,UI32> m_seed = Processing::minimaRegional(m_dist_int,0);

    //watershed ytansformation with the seeds as minima of the distunce function, with the topographic surface as the distunce function, and the growing region is restricted by a mask function as the granular phase
    MatN<DIM,UI32> m_grain_labelled = Processing::watershed(m_seed,m_dist_int,m_grain_binary,0);
    return m_grain_labelled;
}

template<int DIM>
MatN<DIM,UI32> poreDecompositionGrainBoundarySharpVariationGreyLevel(MatN<DIM,UI8> m ){
    //GRAIN PARTITION OF THE BINARY IMAGE
    m = m.opposite();
    //horizontal filter
    MatN<DIM,UI8> filter = Processing::smoothDeriche(m,1);
    filter = Processing::dynamic(filter,10);
    MatN<DIM,UI32> markers_inside_grains = Processing::minimaRegional(filter,0);
    MatN<DIM,UI32> marker_outside_grains =  Processing::watershedBoundary(markers_inside_grains,filter,0);
    marker_outside_grains = Processing::threshold(marker_outside_grains,0,0);
    MatN<DIM,UI32> markers = Processing::labelMerge(marker_outside_grains,markers_inside_grains);
    MatN<DIM,UI8> gradient =Processing::gradientMagnitudeDeriche(m,1);
    MatN<DIM,UI32> watershed =Processing::watershed(markers,gradient);
    watershed = watershed -1;
    return watershed;
}

template<int DIM>
MatN<DIM,UI32> poreDecompositionGrainContactNarrowContact(MatN<DIM,UI8> m ){

    //SEGMENTATION
    MatN<DIM,UI8> m_filter = PDE::nonLinearAnisotropicDiffusion(m,20,10);
    double value;
    MatN<DIM,UI8> m_grain_binary =  Processing::thresholdOtsuMethod(m_filter,value);

    //GRAIN PARTITION OF THE BINARY IMAGE
    //create the distunce function
    MatN<DIM,UI8> m_pore = m_grain_binary.opposite();
    MatN<DIM,UI8> m_dist = Processing::greylevelRange(Processing::distanceEuclidean(m_pore),0,255);
    m_dist = m_dist.opposite();
    //dynamic filter to avoid over-partition
    m_dist = Processing::dynamic(m_dist,20,0);

    //regional minima with with the norm-0 (the norm is here important with norm-1 ->over-partition)
    MatN<DIM,UI8> m_seed = Processing::minimaRegional(m_dist,0);

    //watershed ytansformation with the seeds as minima of the distunce function, with the topographic surface as the distunce function, and the growing region is restricted by a mask function as the granular phase
    MatN<DIM,UI32> m_grain_labelled = Processing::watershed(m_seed,m_dist,m_grain_binary,0);
    return m_grain_labelled;
}

int main(){
    Mat3UI8 m;
#if Pop_OS==2
    std::string dir = "C:/Users/tariel/Dropbox/MyArticle/GranularSegmentation/image/SableHostun_png/";
#else
    std::string dir = "/home/vincent/Dropbox/MyArticle/GranularSegmentation/image/SableHostun_png/";
#endif
    m.loadFromDirectory(dir.c_str());
    {
        Mat2UI8 plane = GeometricalTransformation::plane(m,120);

        m = m(Vec3I32(0,0,0),Vec3I32(50,50,50));
        //               Mat3UI32 grain3d = poreDecompositionGrainBoundarySharpVariationGreyLevel(m);
        Mat3UI32 grain3d = poreDecompositionGrainContactNarrowContact(m);
        //            Mat3UI32 grain3d = poreDecompositionMixedMethod(m,2);

        Visualization::labelForegroundBoundary(grain3d,m,2).display();
        return 1;
    }

}
