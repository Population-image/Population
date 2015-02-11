#include"data/video/Video.h"
#include"3rdparty/VideoVLC.h"
#include "3rdparty/VideoVLCDeprecated.h"
#include"3rdparty/VideoFFMPEG.h"

namespace pop {
bool ConvertRV32ToGrey::init =false;
UI8 ConvertRV32ToGrey::_look_up_table[256][256][256];

Video * Video::create(VideoImpl impl){
    if(impl==Video::VLC){
#if defined(HAVE_VLC)
        return new VideoVLC();
#else
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
