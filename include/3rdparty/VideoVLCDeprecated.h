#ifndef OLDVIDEOVLC_H
#define OLDVIDEOVLC_H

#include"PopulationConfig.h"
#if defined(HAVE_VLC)
#include"string"
#include"data/mat/MatN.h"
#include "data/video/Video.h"

class libvlc_instance_t;
class libvlc_media_player_t;

namespace pop
{


struct ctx;
class POP_EXPORTS VideoVLCDeprecated: public Video
{
private:
    libvlc_instance_t* instance;
    libvlc_media_player_t* mediaPlayer;
    std::string file_playing;
    bool isplaying;
    int my_index;
    ctx* context;
    pop::Mat2UI8 imggrey;
    pop::Mat2RGBUI8 imgcolor;
        bool _isfile;
public:
    VideoVLCDeprecated(bool vlc_debug=false);
    VideoVLCDeprecated(const VideoVLCDeprecated& vlc);
    virtual ~VideoVLCDeprecated();
    bool open(const std::string & filename);
    bool grabMatrixGrey();
    Mat2UI8 &retrieveMatrixGrey();
    bool grabMatrixRGB();
    Mat2RGBUI8 &retrieveMatrixRGB();
    bool tryOpen(const std::string & filename);
    bool isFile() const;

private:
    void release();
};
}
#endif
#endif // OLDVIDEOVLC_H
