%include "../../include/algorithm/Analysis.h"
typedef pop::Analysis Analysis;

ALL_IMAGE_SCALAR_TYPE(Analysis,REVHistogram)


%template(REVPorosity) pop::Analysis::REVPorosity<2>;
%template(REVPorosity) pop::Analysis::REVPorosity<3>;

ALL_IMAGE2_DIM_TYPE(Analysis,maxValue)
ALL_IMAGE2_DIM_TYPE(Analysis,minValue)
ALL_IMAGE_SCALAR_TYPE(Analysis,meanValue)
ALL_IMAGE_SCALAR_TYPE(Analysis,standardDeviationValue)

ALL_IMAGE2_UINT_TYPE(Analysis,histogram)
ALL_IMAGE2_UINT_TYPE(Analysis,area)
ALL_IMAGE2_UINT_TYPE(Analysis,perimeter)
%template(fractalBox) pop::Analysis::fractalBox<2>;
%template(fractalBox) pop::Analysis::fractalBox<3>;
ALL_IMAGE2_UINT_TYPE(Analysis,correlation)
ALL_IMAGE2_UINT_TYPE(Analysis,autoCorrelationFunctionGreyLevel)
ALL_IMAGE2_UINT_TYPE(Analysis,chord)
%template(ldistance) pop::Analysis::ldistance<2>;
%template(ldistance) pop::Analysis::ldistance<3>;
%template(granulometryMatheron) pop::Analysis::granulometryMatheron<2>;
%template(granulometryMatheron) pop::Analysis::granulometryMatheron<3>;
%template(geometricalTortuosity) pop::Analysis::geometricalTortuosity<2>;
%template(geometricalTortuosity) pop::Analysis::geometricalTortuosity<3>;
%template(medialAxis) pop::Analysis::medialAxis<2>;
%template(medialAxis) pop::Analysis::medialAxis<3>;
%template(percolation) pop::Analysis::percolation<2>;
%template(percolation) pop::Analysis::percolation<3>;
%template(percolationErosion) pop::Analysis::percolationErosion<2>;
%template(percolationErosion) pop::Analysis::percolationErosion<3>;
%template(percolationOpening) pop::Analysis::percolationOpening<2>;
%template(percolationOpening) pop::Analysis::percolationOpening<3>;
%template(eulerPoincare) pop::Analysis::eulerPoincare<2>;
%template(eulerPoincare) pop::Analysis::eulerPoincare<3>;

%template(thinningAtConstantTopology) pop::Analysis::thinningAtConstantTopology<3>;
%template(thinningAtConstantTopology) pop::Analysis::thinningAtConstantTopology<2>;
%template(thinningAtConstantTopologyWire) pop::Analysis::thinningAtConstantTopologyWire<2>;


ALL_IMAGE2_UINT_TYPE(Analysis,areaByLabel)
ALL_IMAGE2_UINT_TYPE(Analysis,perimeterByLabel)
ALL_IMAGE2_UINT_TYPE(Analysis,perimeterContactBetweenLabel)
ALL_IMAGE2_UINT_TYPE(Analysis,feretDiameterByLabel)
