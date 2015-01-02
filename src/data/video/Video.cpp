#include"data/video/Video.h"
#include"dependency/VideoVLC.h"
#include"dependency/VideoFFMPEG.h"

namespace pop {
Video * Video::create(VideoImpl impl){
    if(impl==Video::VLC){
#if defined(HAVE_VLC)
        return new VideoVLC();
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
