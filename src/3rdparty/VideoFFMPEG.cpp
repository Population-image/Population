#include "3rdparty/VideoFFMPEG.h"
#if defined(HAVE_FFMPEG)
extern "C"
{
#ifdef __cplusplus
#define __STDC_CONSTANT_MACROS
#ifdef _STDINT_H
#undef _STDINT_H
#endif
#include <stdint.h>
#endif

#ifndef INT64_C
#define INT64_C(c) (c ## LL)
#define UINT64_C(c) (c ## ULL)
#endif
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
}
namespace pop
{

VideoFFMPEG::VideoFFMPEG()
{

    if(init_global==false){
        av_register_all();
        avformat_network_init();
        init_global =true;
    }
    isplaying =false;
    init_local=false;
    context=NULL;
    ccontext=NULL;
    oc=NULL;
    stream=NULL;
    pCodecCtx=NULL;
    codec=NULL;
    video_stream_index=0;

    picture=NULL ;
    picture_grey=NULL ;
    picture_grey_buf=NULL ;
    picture_RGB=NULL ;
    picture_RGB_buf=NULL ;
    ctx =NULL;
    packet = new    AVPacket;
    packet->data=NULL;

}

VideoFFMPEG::VideoFFMPEG(const VideoFFMPEG & video)
{
    if(video.isplaying==true){
        this->open(video.file_playing);
    }
}

bool VideoFFMPEG::open(const std::string & str)
{
    if(init_local==true)
        release();
    init_local=true;
    context = avformat_alloc_context();
    ccontext = avcodec_alloc_context3(NULL);
    //TODO multithreading if no god rtsp infinite loop
    if(avformat_open_input(&context,str.c_str() ,NULL,NULL) != 0){
        std::cerr<<"Open failure: "+str;
    }
    if(avformat_find_stream_info(context,NULL) < 0){
        std::cerr<<"Open failure: "+str;

    }
    //search video stream
    for(unsigned int i =0;i<context->nb_streams;i++){
        if(context->streams[i]->codec->codec_type == AVMEDIA_TYPE_VIDEO)
            video_stream_index = i;
    }


    //open output file
    oc = avformat_alloc_context();

    //start reading packets from stream and write them to file
    av_read_play(context);//play RTSP


    // Get a pointer to the codec context for the video stream
    pCodecCtx =context->streams[video_stream_index]->codec;

    // Find the decoder for the video stream

    codec = avcodec_find_decoder(pCodecCtx->codec_id);
    if (!codec) {
        std::cerr<<"No codec";
        return false;
    }

    avcodec_get_context_defaults3(ccontext, codec);
    avcodec_copy_context(ccontext,context->streams[video_stream_index]->codec);

    if(codec->capabilities&CODEC_CAP_TRUNCATED)
        ccontext->flags|= CODEC_FLAG_TRUNCATED; /* we do not send complete frames */

    if (avcodec_open2(ccontext, codec,NULL) < 0) {
        std::cerr<<"Cannot find the codec";
        return false;
    }

    isplaying =true;
    file_playing = str;
    return true;
}
bool VideoFFMPEG::grabMatrixGrey(){

    if(isplaying==true){
        int check=0;
        int result=-1;
        int try_read=0;
        do{
            if(packet->data!=NULL)
                av_free_packet(packet);

            av_init_packet(packet);
            //packet->size=0;
            if(av_read_frame(context,packet)>=0){
                check =0;
                result=-1;
                if(packet->stream_index == video_stream_index){//packet is video

                    if(picture_grey == NULL)
                    {//create stream in file
                        if(picture!=NULL)
                            av_free(picture);
                        stream = avformat_new_stream(oc,const_cast<AVCodec *>(context->streams[video_stream_index]->codec->codec));
                        avcodec_copy_context(stream->codec,context->streams[video_stream_index]->codec);
                        stream->sample_aspect_ratio = context->streams[video_stream_index]->codec->sample_aspect_ratio;
                        picture = avcodec_alloc_frame();
                        picture_grey = avcodec_alloc_frame();
                        int size2 = avpicture_get_size(PIX_FMT_GRAY8, ccontext->width, ccontext->height);
                        picture_grey_buf = (uint8_t*)(av_malloc(size2));
                        avpicture_fill((AVPicture *) picture_grey, picture_grey_buf, PIX_FMT_GRAY8, ccontext->width, ccontext->height);

                    }

                    packet->stream_index = stream->id;
                    result = avcodec_decode_video2(ccontext, picture, &check, packet);
                }
            }
            else{
                try_read++;
                if(try_read>100)
                    check=1;
            }
        }while((check==0&&result<0));

        rgb =false;
        if(try_read>100)
            return false;
        else
            return true;
    }else{
        return false;
    }
}
bool VideoFFMPEG::grabMatrixRGB(){
    if(isplaying==true){
        int check=0;
        int try_read=0;
        int result=-1;
        do{
            if(packet->data!=NULL)
                av_free_packet(packet);
            av_init_packet(packet);

            if(av_read_frame(context,packet)>=0){
                check =0;
                result=-1;
                if(packet->stream_index == video_stream_index){//packet is video
                    if(picture_RGB == NULL)
                    {//create stream in file
                        if(picture!=NULL)
                            av_free(picture);
                        stream = avformat_new_stream(oc,const_cast<AVCodec *>(context->streams[video_stream_index]->codec->codec));
                        avcodec_copy_context(stream->codec,context->streams[video_stream_index]->codec);
                        stream->sample_aspect_ratio = context->streams[video_stream_index]->codec->sample_aspect_ratio;
                        picture = avcodec_alloc_frame();
                        picture_RGB = avcodec_alloc_frame();
                        int size2 = avpicture_get_size(PIX_FMT_RGB24, ccontext->width, ccontext->height);
                        picture_RGB_buf = (uint8_t*)(av_malloc(size2));
                        result=avpicture_fill((AVPicture *) picture_RGB, picture_RGB_buf, PIX_FMT_RGB24, ccontext->width, ccontext->height);

                    }
                    packet->stream_index = stream->id;
                    result=avcodec_decode_video2(ccontext, picture, &check, packet);
                }
            }
            else{
                pop::BasicUtility::sleep_ms(10);
                try_read++;
                if(try_read>100)
                    check=1;
            }
        }while(check==0&&result<0);
        rgb =true;
        if(try_read>100)
            return false;
        else
            return true;
    }else{
        return false;
    }
}
Mat2RGBUI8  &VideoFFMPEG::retrieveMatrixRGB(){
    if(isplaying==true){
        imgcolor.resize(ccontext->height,ccontext->width);
        if(ctx==NULL)
            ctx= sws_getContext(pCodecCtx->width, pCodecCtx->height, pCodecCtx->pix_fmt, pCodecCtx->width,
                                pCodecCtx->height, PIX_FMT_RGB24, SWS_BICUBIC, NULL, NULL, NULL);
        sws_scale(ctx, picture->data, picture->linesize, 0, pCodecCtx->height,
                  picture_RGB->data, picture_RGB->linesize);

        unsigned char * data =reinterpret_cast<unsigned char *> (imgcolor.data());
        std::copy(picture_RGB->data[0],picture_RGB->data[0]+3*ccontext->height*ccontext->width,data);

        return imgcolor;
    }else{
        return imgcolor;
    }
}

Mat2UI8  &VideoFFMPEG::retrieveMatrixGrey(){
    if(isplaying==true){
        imggrey.resize(ccontext->height,ccontext->width);
        if(ctx==NULL)
            ctx=  sws_getContext(pCodecCtx->width, pCodecCtx->height, pCodecCtx->pix_fmt, pCodecCtx->width,
                                 pCodecCtx->height, PIX_FMT_GRAY8, SWS_BICUBIC, NULL, NULL, NULL);
        sws_scale(ctx, picture->data, picture->linesize, 0, pCodecCtx->height,
                  picture_grey->data, picture_grey->linesize);
        std::copy(picture_grey->data[0],picture_grey->data[0]+ccontext->height*ccontext->width,imggrey.begin());

        return imggrey;
    }else{
        return imggrey;
    }
}
VideoFFMPEG::~VideoFFMPEG(){
    if(init_local==true)
        release();
}
void VideoFFMPEG::release(){

    isplaying =false;
    if(packet->data!=NULL)
        av_free_packet(packet);
    if(picture!=NULL)
        av_free(picture);
    if(picture_grey!=NULL)
        av_free(picture_grey);
    if(picture_grey_buf!=NULL)
        av_free(picture_grey_buf);
    if(picture_RGB!=NULL)
        av_free(picture_RGB);
    if(picture_RGB_buf!=NULL)
        av_free(picture_RGB_buf);
    if(oc!=NULL)
        avformat_free_context(oc);
    // Close the codec
    if(ccontext!=NULL)
        avcodec_close(ccontext);
    // Close the video file
    if(context!=NULL)
        avformat_close_input(&context);
    if(ctx!=NULL)
        sws_freeContext(ctx);
    video_stream_index=0;
    init_local=false;
    context=NULL;
    ccontext=NULL;
    oc=NULL;
    stream=NULL;
    pCodecCtx=NULL;
    codec=NULL;
    picture_grey=NULL ;
    picture_grey_buf=NULL ;
    picture_RGB=NULL ;
    picture_RGB_buf=NULL ;
    picture=NULL ;
    packet->data=NULL;
    free(packet);
    ctx =NULL;
}

bool VideoFFMPEG::init_global = false;



bool VideoFFMPEG::isFile()const{
    if(file_playing.find("rtsp")||file_playing.find("http"))
        return false;
    else
        return true;
}
}
#endif
