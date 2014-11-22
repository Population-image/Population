%include "../../../core/algorithm/Vision.h"
typedef pop::Vision Vision;

%template(TrackingVideoYangMat2UI8) pop::Vision::TrackingVideoYang<pop::Mat2UI8>;
typedef pop::Vision::TrackingVideoYang<pop::Mat2UI8> TrackingVideoYangMat2UI8;
%template(Vision) pop::Vision::harrisDetector<pop::Mat2UI8>;



//TODO Vision
