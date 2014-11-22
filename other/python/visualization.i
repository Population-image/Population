//###Visualization###
%include "../../include/algorithm/Visualization.h"
typedef pop::Visualization Visualization;

%template(labelToRandomRGB) pop::Visualization::labelToRandomRGB<2,pop::UI8>;
%template(labelToRandomRGB) pop::Visualization::labelToRandomRGB<3,pop::UI8>;
%template(labelToRandomRGB) pop::Visualization::labelToRandomRGB<2,pop::UI32>;
%template(labelToRandomRGB) pop::Visualization::labelToRandomRGB<3,pop::UI32>;
%template(labelToRandomRGB) pop::Visualization::labelToRandomRGB<2,pop::UI16>;
%template(labelToRandomRGB) pop::Visualization::labelToRandomRGB<3,pop::UI16>;

%template(labelToRGBGradation) pop::Visualization::labelToRGBGradation<2,pop::UI8>;
%template(labelToRGBGradation) pop::Visualization::labelToRGBGradation<3,pop::UI8>;
%template(labelToRGBGradation) pop::Visualization::labelToRGBGradation<2,pop::UI32>;
%template(labelToRGBGradation) pop::Visualization::labelToRGBGradation<3,pop::UI32>;
%template(labelToRGBGradation) pop::Visualization::labelToRGBGradation<2,pop::UI16>;
%template(labelToRGBGradation) pop::Visualization::labelToRGBGradation<3,pop::UI16>;

%template(labelAverageRGB) pop::Visualization::labelAverageRGB<2,pop::UI8,pop::UI8>;
%template(labelAverageRGB) pop::Visualization::labelAverageRGB<3,pop::UI8,pop::UI8>;
%template(labelAverageRGB) pop::Visualization::labelAverageRGB<2,pop::UI32,pop::UI8>;
%template(labelAverageRGB) pop::Visualization::labelAverageRGB<3,pop::UI32,pop::UI8>;
%template(labelAverageRGB) pop::Visualization::labelAverageRGB<2,pop::UI16,pop::UI8>;
%template(labelAverageRGB) pop::Visualization::labelAverageRGB<3,pop::UI16,pop::UI8>;

%template(labelAverageRGB) pop::Visualization::labelAverageRGB<2,pop::UI8,pop::RGBUI8>;
%template(labelAverageRGB) pop::Visualization::labelAverageRGB<3,pop::UI8,pop::RGBUI8>;
%template(labelAverageRGB) pop::Visualization::labelAverageRGB<2,pop::UI32,pop::RGBUI8>;
%template(labelAverageRGB) pop::Visualization::labelAverageRGB<3,pop::UI32,pop::RGBUI8>;
%template(labelAverageRGB) pop::Visualization::labelAverageRGB<2,pop::UI16,pop::RGBUI8>;
%template(labelAverageRGB) pop::Visualization::labelAverageRGB<3,pop::UI16,pop::RGBUI8>;


%template(labelForegroundBoundary) pop::Visualization::labelForegroundBoundary<2,pop::UI8,pop::UI8>;
%template(labelForegroundBoundary) pop::Visualization::labelForegroundBoundary<3,pop::UI8,pop::UI8>;
%template(labelForegroundBoundary) pop::Visualization::labelForegroundBoundary<2,pop::UI32,pop::UI8>;
%template(labelForegroundBoundary) pop::Visualization::labelForegroundBoundary<3,pop::UI32,pop::UI8>;
%template(labelForegroundBoundary) pop::Visualization::labelForegroundBoundary<2,pop::UI16,pop::UI8>;
%template(labelForegroundBoundary) pop::Visualization::labelForegroundBoundary<3,pop::UI16,pop::UI8>;

%template(labelForegroundBoundary) pop::Visualization::labelForegroundBoundary<2,pop::UI8,pop::RGBUI8>;
%template(labelForegroundBoundary) pop::Visualization::labelForegroundBoundary<3,pop::UI8,pop::RGBUI8>;
%template(labelForegroundBoundary) pop::Visualization::labelForegroundBoundary<2,pop::UI32,pop::RGBUI8>;
%template(labelForegroundBoundary) pop::Visualization::labelForegroundBoundary<3,pop::UI32,pop::RGBUI8>;
%template(labelForegroundBoundary) pop::Visualization::labelForegroundBoundary<2,pop::UI16,pop::RGBUI8>;
%template(labelForegroundBoundary) pop::Visualization::labelForegroundBoundary<3,pop::UI16,pop::RGBUI8>;


%template(labelForeground) pop::Visualization::labelForeground<2,pop::UI8,pop::UI8>;
%template(labelForeground) pop::Visualization::labelForeground<3,pop::UI8,pop::UI8>;
%template(labelForeground) pop::Visualization::labelForeground<2,pop::UI32,pop::UI8>;
%template(labelForeground) pop::Visualization::labelForeground<3,pop::UI32,pop::UI8>;
%template(labelForeground) pop::Visualization::labelForeground<2,pop::UI16,pop::UI8>;
%template(labelForeground) pop::Visualization::labelForeground<3,pop::UI16,pop::UI8>;

%template(labelForeground) pop::Visualization::labelForeground<2,pop::UI8,pop::RGBUI8>;
%template(labelForeground) pop::Visualization::labelForeground<3,pop::UI8,pop::RGBUI8>;
%template(labelForeground) pop::Visualization::labelForeground<2,pop::UI32,pop::RGBUI8>;
%template(labelForeground) pop::Visualization::labelForeground<3,pop::UI32,pop::RGBUI8>;
%template(labelForeground) pop::Visualization::labelForeground<2,pop::UI16,pop::RGBUI8>;
%template(labelForeground) pop::Visualization::labelForeground<3,pop::UI16,pop::RGBUI8>;

%template(cube) pop::Visualization::cube<pop::UI8>;
%template(cube) pop::Visualization::cube<pop::RGBUI8>;
%template(cubeExtruded) pop::Visualization::cubeExtruded<pop::UI8>;
%template(cubeExtruded) pop::Visualization::cubeExtruded<pop::RGBUI8>;

%template(lineCube) pop::Visualization::lineCube<pop::UI8>;
%template(lineCube) pop::Visualization::lineCube<pop::RGBUI8>;
%template(marchingCube) pop::Visualization::marchingCube<pop::UI8>;
%template(marchingCube) pop::Visualization::marchingCube<pop::RGBUI8>;


%template(voxelSurface) pop::Visualization::voxelSurface<pop::UI8>;
%template(voxelSurface) pop::Visualization::voxelSurface<pop::RGBUI8>;

%template(plane) pop::Visualization::plane<pop::UI8>;
%template(plane) pop::Visualization::plane<pop::RGBUI8>;
