%include "../../../include/algorithm/GeometricalTransformation.h"
typedef pop::GeometricalTransformation GeometricalTransformation;


ALL_IMAGE(GeometricalTransformation,translate)
ALL_IMAGE(GeometricalTransformation,scale)
ALL_IMAGE(GeometricalTransformation,rotateMultPi_2)

ALL_IMAGE(GeometricalTransformation,rotate)
ALL_IMAGE(GeometricalTransformation,mirror)

%template(transformAffine2D) pop::GeometricalTransformation::transformAffine2D<pop::UI8>;
%template(transformAffine2D) pop::GeometricalTransformation::transformAffine2D<pop::F64>;
%template(transformAffine2D) pop::GeometricalTransformation::transformAffine2D<pop::RGBUI8>;
%template(transformAffine2D) pop::GeometricalTransformation::transformAffine2D<pop::UI16>;
%template(transformAffine2D) pop::GeometricalTransformation::transformAffine2D<pop::UI32>;

%template(transformHomogeneous2D) pop::GeometricalTransformation::transformHomogeneous2D<pop::UI8>;
%template(transformHomogeneous2D) pop::GeometricalTransformation::transformHomogeneous2D<pop::F64>;
%template(transformHomogeneous2D) pop::GeometricalTransformation::transformHomogeneous2D<pop::RGBUI8>;
%template(transformHomogeneous2D) pop::GeometricalTransformation::transformHomogeneous2D<pop::UI16>;
%template(transformHomogeneous2D) pop::GeometricalTransformation::transformHomogeneous2D<pop::UI32>;
