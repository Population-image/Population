%include "../../../include/algorithm/Analysis.h"
typedef pop::Analysis Analysis;

ALL_IMAGE_SCALAR(Analysis,REVHistogram)
ALL_IMAGE_SCALAR(Analysis,REVPorosity)

ALL_IMAGE(Analysis,maxValue)
ALL_IMAGE(Analysis,minValue)
ALL_IMAGE_SCALAR_SWIG(Analysis,_meanValue,meanValue)
ALL_IMAGE_SCALAR_SWIG(Analysis,_standardDeviationValue,standardDeviationValue)

ALL_IMAGE_UNINT(Analysis,histogram)
ALL_IMAGE_UNINT(Analysis,area)
ALL_IMAGE_UNINT(Analysis,perimeter)
ALL_IMAGE_UNINT(Analysis,fractalBox)
ALL_IMAGE_UNINT(Analysis,correlation)
ALL_IMAGE_UNINT(Analysis,autoCorrelationFunctionGreyLevel)

ALL_IMAGE_UINT_TYPE(Analysis,_correlationDirectionByFFT,correlationDirectionByFFT)
ALL_IMAGE_UNINT(Analysis,chord)
ALL_IMAGE_UNINT(Analysis,ldistance)
ALL_IMAGE_UNINT(Analysis,granulometryMatheron)
ALL_IMAGE_UNINT(Analysis,geometricalTortuosity)
ALL_IMAGE_UNINT(Analysis,medialAxis)
ALL_IMAGE_UNINT(Analysis,percolation)
ALL_IMAGE_UNINT(Analysis,percolationErosion)
ALL_IMAGE_UNINT(Analysis,percolationOpening)
ALL_IMAGE_UNINT(Analysis,eulerPoincare)

%template(thinningAtConstantTopology3d) pop::Analysis::thinningAtConstantTopology3d<pop::Mat3UI8>;
%template(thinningAtConstantTopology2d) pop::Analysis::thinningAtConstantTopology2d<pop::Mat2UI8>;
%template(thinningAtConstantTopology2dWire) pop::Analysis::thinningAtConstantTopology2dWire<pop::Mat2UI8>;


ALL_IMAGE_UNINT(Analysis,areaByLabel)
ALL_IMAGE_UNINT(Analysis,perimeterByLabel)
ALL_IMAGE_UNINT(Analysis,perimeterContactBetweenLabel)
ALL_IMAGE_UNINT(Analysis,feretDiameterByLabel)
