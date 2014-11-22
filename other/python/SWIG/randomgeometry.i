%include "../../../include/algorithm/RandomGeometry.h"
typedef pop::RandomGeometry RandomGeometry;
%template(poissonPointProcess) pop::RandomGeometry::poissonPointProcess<2>;
%template(poissonPointProcess) pop::RandomGeometry::poissonPointProcess<3>;

%template(poissonPointProcessNonUniform) pop::RandomGeometry::poissonPointProcessNonUniform<2,pop::F64>;
%template(poissonPointProcessNonUniform) pop::RandomGeometry::poissonPointProcessNonUniform<3,pop::F64>;


%template(hardCoreFilter) pop::RandomGeometry::hardCoreFilter<2>;
%template(hardCoreFilter) pop::RandomGeometry::hardCoreFilter<3>;

%template(minOverlapFilter) pop::RandomGeometry::minOverlapFilter<2>;
%template(minOverlapFilter) pop::RandomGeometry::minOverlapFilter<3>;

%template(intersectionGrainToMask) pop::RandomGeometry::intersectionGrainToMask<2>;
%template(intersectionGrainToMask) pop::RandomGeometry::intersectionGrainToMask<3>;

%template(sphere) pop::RandomGeometry::sphere<2>;
%template(sphere) pop::RandomGeometry::sphere<3>;

%template(box) pop::RandomGeometry::box<2>;
%template(box) pop::RandomGeometry::box<3>;

%template(polyhedra) pop::RandomGeometry::polyhedra<2>;
%template(polyhedra) pop::RandomGeometry::polyhedra<3>;

%template(ellipsoid) pop::RandomGeometry::ellipsoid<2>;
%template(ellipsoid) pop::RandomGeometry::ellipsoid<3>;


%template(matrixBinary) pop::RandomGeometry::matrixBinary<2>;
%template(matrixBinary) pop::RandomGeometry::matrixBinary<3>;


%template(addition) pop::RandomGeometry::addition<2>;
%template(addition) pop::RandomGeometry::addition<3>;

%template(RGBRandomBlackOrWhite) pop::RandomGeometry::RGBRandomBlackOrWhite<2>;
%template(RGBRandomBlackOrWhite) pop::RandomGeometry::RGBRandomBlackOrWhite<3>;


%template(RGBRandom) pop::RandomGeometry::RGBRandom<2>;
%template(RGBRandom) pop::RandomGeometry::RGBRandom<3>;

%template(RGBFromMatrix) pop::RandomGeometry::RGBFromMatrix<2>;
%template(RGBFromMatrix) pop::RandomGeometry::RGBFromMatrix<3>;

%template(germToMatrix) pop::RandomGeometry::germToMatrix<2>;
%template(germToMatrix) pop::RandomGeometry::germToMatrix<3>;

%template(continuousToDiscrete) pop::RandomGeometry::continuousToDiscrete<2>;
%template(continuousToDiscrete) pop::RandomGeometry::continuousToDiscrete<3>;

%template(gaussianThesholdedRandomField) pop::RandomGeometry::gaussianThesholdedRandomField<2>;
%template(gaussianThesholdedRandomField) pop::RandomGeometry::gaussianThesholdedRandomField<3>;

%template(randomStructure) pop::RandomGeometry::randomStructure<2>;
%template(randomStructure) pop::RandomGeometry::randomStructure<3>;

%template(annealingSimutated) pop::RandomGeometry::annealingSimutated<2,2>;
%template(annealingSimutated) pop::RandomGeometry::annealingSimutated<2,3>;
%template(annealingSimutated) pop::RandomGeometry::annealingSimutated<3,2>;
%template(annealingSimutated) pop::RandomGeometry::annealingSimutated<3,3>;

