#include"data/video/Video.h"
#include"3rdparty/VideoVLC.h"
#include "3rdparty/VideoVLCDeprecated.h"
#include"3rdparty/VideoFFMPEG.h"

namespace pop {

Video * Video::create(VideoImpl impl, bool debug) {
    if(impl==Video::VLC){
#if defined(HAVE_VLC)
        return new VideoVLC(debug);
#else
        std::cerr<<"[ERROR][Video::create] VLC library is not included. For qtcreator, in populationconfig.pri, you uncomment this line  CONFIG += HAVE_VLC . For CMake, in CMakeList.txt, you set WITH_VLC at ON."<<std::endl;
        return NULL;
#endif
    } else if(impl==Video::VLCDEPRECATED){
#if defined(HAVE_VLC)
        return new VideoVLCDeprecated(debug);
#else
        std::cerr<<"[ERROR][Video::create] VLC library is not included. For qtcreator, in populationconfig.pri, you uncomment this line  CONFIG += HAVE_VLC . For CMake, in CMakeList.txt, you set WITH_VLC at ON."<<std::endl;
        return NULL;
#endif
    } else if (impl==Video::FFMPEG){
#if defined(HAVE_FFMPEG)
        return new VideoFFMPEG();
#else
        std::cerr<<"[ERROR][Video::create] FFMPEG library is not included. For qtcreator, in populationconfig.pri, you uncomment this line  CONFIG += HAVE_FFMPEG . For CMake, in CMakeList.txt, you set WITH_FFMPEG at ON."<<std::endl;
        return NULL;
#endif
    } else {
        std::cerr<<"[ERROR][Video::create] Unknown VideoImpl"<<std::endl;
        return NULL;
    }
}

}
