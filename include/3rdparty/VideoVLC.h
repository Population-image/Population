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

/*!
 * \brief The ConvertRV32ToGrey struct is used as a rainbow table to speed up the call to lumi()
 */
class ConvertRV32ToGrey{
private:
    static bool init;
    static pop::UI8 _look_up_table[256][256][256];
    static pop::UI8 lumi(const pop::VecN<4,pop::UI8> &rgb);
};

/*!
 * \brief The ConvertRV32ToRGBUI8 struct is used to speed up the call to lumi()
 */
struct ConvertRV32ToRGBUI8
{
    static pop::RGBUI8 lumi(const pop::VecN<4,pop::UI8> &rgb);
};


#endif

#endif // VIDEOVLC_H
