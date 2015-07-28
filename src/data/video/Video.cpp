#include"data/video/Video.h"
#include"3rdparty/VideoVLC.h"
#include "3rdparty/VideoVLCDeprecated.h"
#include"3rdparty/VideoFFMPEG.h"

namespace pop {


Video * Video::create(VideoImpl impl){
    if(impl==Video::VLC){
#if defined(HAVE_VLC)
        return new VideoVLC();
#else
        std::cerr<<"[ERROR][Video::create] VLV library is not included. For qtcreator, in populationconfig.pri, you uncomment this line  CONFIG += HAVE_VLC . For CMake, in CMakeList.txt, you set WITH_VLC at ON.";
        return NULL;
#endif
    } else if(impl==Video::VLCDEPRECATED){
#if defined(HAVE_VLC)
        return new VideoVLCDeprecated();
#else
        return NULL;
#endif
    } else if (impl==Video::FFMPEG){
#if defined(HAVE_FFMPEG)
        return new VideoFFMPEG();
#else
        return NULL;
#endif
    } else {
        return NULL;
    }
}
}
