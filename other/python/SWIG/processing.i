//###Processing###
%include "../../../include/algorithm/Processing.h"
typedef pop::Processing Processing;

//Generator
ALL_IMAGE(Processing,randomField)
ALL_IMAGE(Processing,fill)


//Point
ALL_IMAGE_DIM_TYPE(Processing,_threshold,threshold);
%template(thresholdColorInRange) pop::Processing::thresholdColorInRange<2>;
%template(thresholdColorInRange) pop::Processing::thresholdColorInRange<3>;
ALL_IMAGE_DIM_TYPE(Processing,_thresholdKMeansVariation,thresholdKMeansVariation);
ALL_IMAGE_DIM_TYPE(Processing,_thresholdOtsuMethod,thresholdOtsuMethod);
ALL_IMAGE_DIM_TYPE(Processing,_thresholdToggleMappingMorphological,thresholdToggleMappingMorphological);
ALL_IMAGE_DIM_TYPE(Processing,_thresholdToggleMappingMorphologicalFabrizio,thresholdToggleMappingMorphologicalFabrizio);
ALL_IMAGE_DIM_TYPE(Processing,_thresholdNiblackMethod,thresholdNiblackMethod);
ALL_IMAGE_DIM_TYPE(Processing,_thresholdMultiValley,thresholdMultiValley);
ALL_IMAGE_DIM_TYPE(Processing,_edgeDetectorCanny,edgeDetectorCanny);
ALL_IMAGE(Processing,fofx)
ALL_IMAGE(Processing,greylevelScaleContrast)
ALL_IMAGE(Processing,greylevelRange)
ALL_IMAGE(Processing,greylevelTranslateMeanValue)
ALL_IMAGE(Processing,greylevelRemoveEmptyValue)
ALL_IMAGE(Processing,integral)
ALL_IMAGE(Processing,integralPower2)
ALL_IMAGE_BINARY(Processing,mask)


 //Neighborhood
ALL_IMAGE(Processing,minimaLocal)
ALL_IMAGE_DIM_TYPE(Processing,_minimaLocalMap,minimaLocalMap);
ALL_IMAGE(Processing,maximaLocal)
ALL_IMAGE_DIM_TYPE(Processing,_maximaLocalMap,maximaLocalMap);
ALL_IMAGE(Processing,extremaLocal)
ALL_IMAGE_DIM_TYPE(Processing,_extremaLocalMap,extremaLocalMap);
ALL_IMAGE(Processing,erosion)
ALL_IMAGE_BINARY(Processing,erosionStructuralElement)
ALL_IMAGE(Processing,dilation)
ALL_IMAGE_BINARY(Processing,dilationStructuralElement)
ALL_IMAGE(Processing,median)
ALL_IMAGE_BINARY(Processing,medianStructuralElement)
ALL_IMAGE(Processing,mean)
ALL_IMAGE_BINARY(Processing,meanStructuralElement)
ALL_IMAGE(Processing,closing)
ALL_IMAGE_BINARY(Processing,closingStructuralElement)
ALL_IMAGE(Processing,opening)
ALL_IMAGE_BINARY(Processing,openingStructuralElement)
ALL_IMAGE(Processing,alternateSequentialCO)
ALL_IMAGE_BINARY(Processing,alternateSequentialCOStructuralElement)
ALL_IMAGE(Processing,alternateSequentialOC)
ALL_IMAGE_BINARY(Processing,alternateSequentialOCStructuralElement)
ALL_IMAGE_BINARY(Processing,hitOrMiss)
ALL_IMAGE(Processing,meanShiftFilter)
ALL_IMAGE_FLOAT(Processing,convolution)
ALL_IMAGE(Processing,gradientMagnitudeSobel)
ALL_IMAGE_DIM_TYPE(Processing,_gradientSobel,gradientSobel)
ALL_IMAGE_DIM_TYPE(Processing,_gradientVecSobel,gradientVecSobel)

ALL_IMAGE(Processing,gradientMagnitudeGaussian)
ALL_IMAGE_DIM_TYPE(Processing,_gradientGaussian,gradientGaussian)
ALL_IMAGE_DIM_TYPE(Processing,_gradientVecGaussian,gradientVecGaussian)
ALL_IMAGE(Processing,smoothGaussian)

ALL_IMAGE(Processing,gradientMagnitudeDeriche)
ALL_IMAGE_DIM_TYPE(Processing,_gradientDeriche,gradientDeriche)
ALL_IMAGE_DIM_TYPE(Processing,_gradientVecDeriche,gradientVecDeriche)
ALL_IMAGE(Processing,smoothDeriche)


 //Seeds
ALL_IMAGE_UNINT(Processing,labelMerge)
ALL_IMAGE_UNINT_BINARY(Processing,labelFromSingleSeed)



 //Region growing
%template(holeFilling) pop::Processing::holeFilling<pop::Mat2UI8>;
%template(holeFilling) pop::Processing::holeFilling<pop::Mat3UI8>;
ALL_IMAGE_LABEL(Processing,regionGrowingAdamsBischofMeanOverStandardDeviation)
ALL_IMAGE_LABEL(Processing,regionGrowingAdamsBischofMean)
%template(clusterToLabel) pop::Processing::_clusterToLabel<2,pop::UI8>;
%template(clusterToLabel) pop::Processing::_clusterToLabel<3,pop::UI8>;
%template(clusterMax) pop::Processing::clusterMax<pop::Mat2UI8>;
%template(clusterMax) pop::Processing::clusterMax<pop::Mat3UI8>;
ALL_IMAGE_DIM_TYPE(Processing,_minimaRegional,minimaRegional)
ALL_IMAGE_LABEL(Processing,watershed)
ALL_IMAGE_LABEL(Processing,watershedBoundary)
ALL_IMAGE_LABEL_MASK(Processing,watershed)
ALL_IMAGE_LABEL_MASK(Processing,watershedBoundary)
ALL_IMAGE_UNINT(Processing,geodesicReconstruction)
ALL_IMAGE_UNINT(Processing,dynamic)
ALL_IMAGE_UNINT(Processing,voronoiTesselation)
ALL_IMAGE_UNINT_BINARY(Processing,voronoiTesselation)
ALL_IMAGE_UNINT(Processing,voronoiTesselationEuclidean)
ALL_IMAGE_UINT_TYPE(Processing,_distance,distance)
ALL_IMAGE_UINT_TYPE(Processing,_distanceMask,distance)
ALL_IMAGE_UNINT(Processing,distanceEuclidean)
ALL_IMAGE_UNINT(Processing,dilationRegionGrowing)
ALL_IMAGE_UNINT(Processing,erosionRegionGrowing)
ALL_IMAGE_UNINT(Processing,openingRegionGrowing)
ALL_IMAGE_UNINT(Processing,closingRegionGrowing)
ALL_IMAGE_UNINT_BINARY(Processing,erosionRegionGrowingStructuralElement)
ALL_IMAGE_UNINT_BINARY(Processing,dilationRegionGrowingStructuralElement)
ALL_IMAGE_UNINT_BINARY(Processing,openingRegionGrowingStructuralElement)
ALL_IMAGE_UNINT_BINARY(Processing,closingRegionGrowingStructuralElement)
ALL_IMAGE(Processing,rotateAtHorizontal)


