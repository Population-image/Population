//###Visualization###
%include "../../../include/algorithm/Visualization.h"
typedef pop::Visualization Visualization;

%template(labelToRandomRGB) pop::Visualization::__labelToRandomRGB<2,pop::UI8>;
%template(labelToRandomRGB) pop::Visualization::__labelToRandomRGB<3,pop::UI8>;
%template(labelToRandomRGB) pop::Visualization::__labelToRandomRGB<2,pop::UI32>;
%template(labelToRandomRGB) pop::Visualization::__labelToRandomRGB<3,pop::UI32>;
%template(labelToRandomRGB) pop::Visualization::__labelToRandomRGB<2,pop::UI16>;
%template(labelToRandomRGB) pop::Visualization::__labelToRandomRGB<3,pop::UI16>;

%template(labelToRGBGradation) pop::Visualization::__labelToRGBGradation<2,pop::UI8>;
%template(labelToRGBGradation) pop::Visualization::__labelToRGBGradation<3,pop::UI8>;
%template(labelToRGBGradation) pop::Visualization::__labelToRGBGradation<2,pop::UI32>;
%template(labelToRGBGradation) pop::Visualization::__labelToRGBGradation<3,pop::UI32>;
%template(labelToRGBGradation) pop::Visualization::__labelToRGBGradation<2,pop::UI16>;
%template(labelToRGBGradation) pop::Visualization::__labelToRGBGradation<3,pop::UI16>;

%template(labelAverageRGB) pop::Visualization::__labelAverageRGB<2,pop::UI8,pop::UI8>;
%template(labelAverageRGB) pop::Visualization::__labelAverageRGB<3,pop::UI8,pop::UI8>;
%template(labelAverageRGB) pop::Visualization::__labelAverageRGB<2,pop::UI32,pop::UI8>;
%template(labelAverageRGB) pop::Visualization::__labelAverageRGB<3,pop::UI32,pop::UI8>;
%template(labelAverageRGB) pop::Visualization::__labelAverageRGB<2,pop::UI16,pop::UI8>;
%template(labelAverageRGB) pop::Visualization::__labelAverageRGB<3,pop::UI16,pop::UI8>;

%template(labelAverageRGB) pop::Visualization::__labelAverageRGB<2,pop::UI8,pop::RGBUI8>;
%template(labelAverageRGB) pop::Visualization::__labelAverageRGB<3,pop::UI8,pop::RGBUI8>;
%template(labelAverageRGB) pop::Visualization::__labelAverageRGB<2,pop::UI32,pop::RGBUI8>;
%template(labelAverageRGB) pop::Visualization::__labelAverageRGB<3,pop::UI32,pop::RGBUI8>;
%template(labelAverageRGB) pop::Visualization::__labelAverageRGB<2,pop::UI16,pop::RGBUI8>;
%template(labelAverageRGB) pop::Visualization::__labelAverageRGB<3,pop::UI16,pop::RGBUI8>;


%template(labelForegroundBoundary) pop::Visualization::__labelForegroundBoundary<2,pop::UI8,pop::UI8>;
%template(labelForegroundBoundary) pop::Visualization::__labelForegroundBoundary<3,pop::UI8,pop::UI8>;
%template(labelForegroundBoundary) pop::Visualization::__labelForegroundBoundary<2,pop::UI32,pop::UI8>;
%template(labelForegroundBoundary) pop::Visualization::__labelForegroundBoundary<3,pop::UI32,pop::UI8>;
%template(labelForegroundBoundary) pop::Visualization::__labelForegroundBoundary<2,pop::UI16,pop::UI8>;
%template(labelForegroundBoundary) pop::Visualization::__labelForegroundBoundary<3,pop::UI16,pop::UI8>;

%template(labelForegroundBoundary) pop::Visualization::__labelForegroundBoundary<2,pop::UI8,pop::RGBUI8>;
%template(labelForegroundBoundary) pop::Visualization::__labelForegroundBoundary<3,pop::UI8,pop::RGBUI8>;
%template(labelForegroundBoundary) pop::Visualization::__labelForegroundBoundary<2,pop::UI32,pop::RGBUI8>;
%template(labelForegroundBoundary) pop::Visualization::__labelForegroundBoundary<3,pop::UI32,pop::RGBUI8>;
%template(labelForegroundBoundary) pop::Visualization::__labelForegroundBoundary<2,pop::UI16,pop::RGBUI8>;
%template(labelForegroundBoundary) pop::Visualization::__labelForegroundBoundary<3,pop::UI16,pop::RGBUI8>;


%template(labelForeground) pop::Visualization::__labelForeground<2,pop::UI8,pop::UI8>;
%template(labelForeground) pop::Visualization::__labelForeground<3,pop::UI8,pop::UI8>;
%template(labelForeground) pop::Visualization::__labelForeground<2,pop::UI32,pop::UI8>;
%template(labelForeground) pop::Visualization::__labelForeground<3,pop::UI32,pop::UI8>;
%template(labelForeground) pop::Visualization::__labelForeground<2,pop::UI16,pop::UI8>;
%template(labelForeground) pop::Visualization::__labelForeground<3,pop::UI16,pop::UI8>;

%template(labelForeground) pop::Visualization::__labelForeground<2,pop::UI8,pop::RGBUI8>;
%template(labelForeground) pop::Visualization::__labelForeground<3,pop::UI8,pop::RGBUI8>;
%template(labelForeground) pop::Visualization::__labelForeground<2,pop::UI32,pop::RGBUI8>;
%template(labelForeground) pop::Visualization::__labelForeground<3,pop::UI32,pop::RGBUI8>;
%template(labelForeground) pop::Visualization::__labelForeground<2,pop::UI16,pop::RGBUI8>;
%template(labelForeground) pop::Visualization::__labelForeground<3,pop::UI16,pop::RGBUI8>;

%template(cube) pop::Visualization::cube<pop::Mat3UI8>;
%template(cube) pop::Visualization::cube<pop::Mat3RGBUI8>;
%template(cubeExtruded) pop::Visualization::cubeExtruded<pop::UI8>;
%template(cubeExtruded) pop::Visualization::cubeExtruded<pop::RGBUI8>;

%template(lineCube) pop::Visualization::lineCube<pop::Mat3UI8>;
%template(lineCube) pop::Visualization::lineCube<pop::Mat3RGBUI8>;
%template(marchingCube) pop::Visualization::marchingCube<pop::Mat3UI8>;
%template(marchingCube) pop::Visualization::marchingCube<pop::Mat3RGBUI8>;


%template(marchingCubeLevelSet) pop::Visualization::marchingCubeLevelSet<pop::Mat3F64>;
%template(marchingCubeLevelSet) pop::Visualization::marchingCubeLevelSet<pop::Mat3F64,pop::Mat3UI8>;
%template(marchingCubeLevelSet) pop::Visualization::marchingCubeLevelSet<pop::Mat3F64,pop::Mat3RGBUI8>;

%template(voxelSurface) pop::Visualization::voxelSurface<pop::Mat3UI8>;
%template(voxelSurface) pop::Visualization::voxelSurface<pop::Mat3RGBUI8>;

%template(plane) pop::Visualization::plane<pop::Mat3UI8>;
%template(plane) pop::Visualization::plane<pop::Mat3RGBUI8>;
