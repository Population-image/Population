#ifndef VIDEOVLC_H
#define VIDEOVLC_H

#include"PopulationConfig.h"

#if defined(HAVE_VLC)

#include"string"
#include"data/mat/MatN.h"
#include "data/video/Video.h"

#include <vlc/vlc.h>

namespace pop
{


struct ctx;
class POP_EXPORTS VideoVLC: public Video
{
private:
    libvlc_instance_t* instance;
    libvlc_media_player_t* mediaPlayer;
    libvlc_event_manager_t* eventManager;
    std::string file_playing;
    bool isplaying;
    int my_index;
    ctx* context;
    pop::Mat2UI8 imggrey;
    pop::Mat2RGBUI8 imgcolor;
        bool _isfile;
public:
    VideoVLC();
    VideoVLC(const VideoVLC& vlc);
    virtual ~VideoVLC();
    bool open(const std::string &filename);
    bool grabMatrixGrey();
    Mat2UI8 &retrieveMatrixGrey();
    bool grabMatrixRGB();
    Mat2RGBUI8 &retrieveMatrixRGB();
    bool isFile()const;
    bool isPlaying() const;

private:
    void release();
};

}

#endif

#endif // VIDEOVLC_H
