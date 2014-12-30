#ifndef VIDEOFFMPEG_H
#define VIDEOFFMPEG_H
#include"PopulationConfig.h"
#if defined(HAVE_FFMPEG)
#include"string"
#include"data/mat/MatN.h"
#include "data/video/Video.h"

class AVFormatContext;
class  AVPacket;
class  AVStream;
class  AVCodecContext;
class  AVCodec;
class AVFrame;
struct SwsContext;


namespace pop
{
class POP_EXPORTS VideoFFMPEG: public Video
{
private:
    static bool init_global;
    bool init_local;
    AVFormatContext* context;
    AVCodecContext* ccontext;
    AVFormatContext* oc;
    AVPacket * packet;
    AVStream* stream;
    AVCodecContext * pCodecCtx;
    AVCodec *codec;
    int video_stream_index;

    AVFrame* picture ;
    AVFrame* picture_grey ;
    UI8 * picture_grey_buf ;
    AVFrame* picture_RGB ;
    UI8 * picture_RGB_buf ;
    struct SwsContext *ctx;
    bool rgb;
    pop::Mat2RGBUI8 imgcolor;
    pop::Mat2UI8 imggrey;

public:
    std::string file_playing;
    bool isplaying;
    void release();
public:
    VideoFFMPEG();
    VideoFFMPEG(const VideoFFMPEG& vlc);
    virtual ~VideoFFMPEG();
    bool open(const std::string & filename)throw(pexception);
    bool grabMatrixGrey();
    Mat2UI8 &retrieveMatrixGrey();
    bool grabMatrixRGB();
    Mat2RGBUI8 &retrieveMatrixRGB();
        bool isFile()const;
};
}
#endif
#endif // VIDEOFFMPEG_H
