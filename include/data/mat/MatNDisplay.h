/******************************************************************************\
|*                   Population library for C++ X.X.X                         *|
|*----------------------------------------------------------------------------*|
The Population License is similar to the MIT license in adding this clause:
for any writing public or private that has resulted from the use of the
software population, the reference of this book "Population library, 2012,
Vincent Tariel" shall be included in it.

So, the terms of the Population License are:

Copyright © 2012-2015, Tariel Vincent

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to
deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software and for any writing
public or private that has resulted from the use of the software population,
the reference of this book "Population library, 2012, Vincent Tariel" shall
be included in it.

The Software is provided "as is", without warranty of any kind, express or
implied, including but not limited to the warranties of merchantability,
fitness for a particular purpose and noninfringement. In no event shall the
authors or copyright holders be liable for any claim, damages or other
liability, whether in an action of contract, tort or otherwise, arising
from, out of or in connection with the software or the use or other dealings
in the Software.
\***************************************************************************/

#ifndef MatNDISPLAY_H
#define MatNDISPLAY_H

#define UNICODE 1


#include"data/mat/MatN.h"
#include"data/mat/Mat2x.h"

#include"algorithm/Convertor.h"
#include"algorithm/GeometricalTransformation.h"
#include"algorithm/Processing.h"
namespace pop{
class MatNDisplayInterface{

public:

    virtual ~MatNDisplayInterface ();
    virtual MatNDisplayInterface & display(const MatN<2, RGBUI8 > &img)=0;
//    virtual MatNDisplayInterface & display(const MatN<2, RGBAUI8 > &img)=0;
    virtual MatNDisplayInterface & display(const MatN<2, UI8 > &img)=0;
    virtual operator  bool () const=0;
    virtual bool 	is_empty () const=0;
    virtual bool 	is_closed () const=0;
    virtual bool 	is_resized () const=0;
    virtual bool 	is_moved () const=0;
    virtual bool 	is_event () const=0;
    virtual bool 	is_fullscreen () const=0;
    virtual bool 	is_key () const=0;
    virtual bool 	is_key (const unsigned int keycode) const=0;
    virtual bool 	is_key (const char *const keycode) const=0;
    virtual bool 	is_key_sequence (const unsigned int *const keycodes_sequence, const unsigned int length, const  bool remove_sequence=false)=0;
    virtual bool 	is_keyESC () const=0;
    virtual bool 	is_keyF1 () const=0;
    virtual bool 	is_keyF2 () const=0;
    virtual bool 	is_keyF3 () const=0;
    virtual bool 	is_keyF4 () const=0;
    virtual bool 	is_keyF5 () const=0;
    virtual bool 	is_keyF6 () const=0;
    virtual bool 	is_keyF7 () const=0;
    virtual bool 	is_keyF8 () const=0;
    virtual bool 	is_keyF9 () const=0;
    virtual bool 	is_keyF10 () const=0;
    virtual bool 	is_keyF11 () const=0;
    virtual bool 	is_keyF12 () const=0;
    virtual bool 	is_keyPAUSE () const=0;
    virtual bool 	is_key1 () const=0;
    virtual bool 	is_key2 () const=0;
    virtual bool 	is_key3 () const=0;
    virtual bool 	is_key4 () const=0;
    virtual bool 	is_key5 () const=0;
    virtual bool 	is_key6 () const=0;
    virtual bool 	is_key7 () const=0;
    virtual bool 	is_key8 () const=0;
    virtual bool 	is_key9 () const=0;
    virtual bool 	is_key0 () const=0;
    virtual bool 	is_keyBACKSPACE () const=0;
    virtual bool 	is_keyINSERT () const=0;
    virtual bool 	is_keyHOME () const=0;
    virtual bool 	is_keyPAGEUP () const=0;
    virtual bool 	is_keyTAB () const=0;
    virtual bool 	is_keyQ () const=0;
    virtual bool 	is_keyW () const=0;
    virtual bool 	is_keyE () const=0;
    virtual bool 	is_keyR () const=0;
    virtual bool 	is_keyT () const=0;
    virtual bool 	is_keyY () const=0;
    virtual bool 	is_keyU () const=0;
    virtual bool 	is_keyI () const=0;
    virtual bool 	is_keyO () const=0;
    virtual bool 	is_keyP () const=0;
    virtual bool 	is_keyDELETE () const=0;
    virtual bool 	is_keyEND () const=0;
    virtual bool 	is_keyPAGEDOWN () const=0;
    virtual bool 	is_keyCAPSLOCK () const=0;
    virtual bool 	is_keyA () const=0;
    virtual bool 	is_keyS () const=0;
    virtual bool 	is_keyD () const=0;
    virtual bool 	is_keyF () const=0;
    virtual bool 	is_keyG () const=0;
    virtual bool 	is_keyH () const=0;
    virtual bool 	is_keyJ () const=0;
    virtual bool 	is_keyK () const=0;
    virtual bool 	is_keyL () const=0;
    virtual bool 	is_keyENTER () const=0;
    virtual bool 	is_keySHIFTLEFT () const=0;
    virtual bool 	is_keyZ () const=0;
    virtual bool 	is_keyX () const=0;
    virtual bool 	is_keyC () const=0;
    virtual bool 	is_keyV () const=0;
    virtual bool 	is_keyB () const=0;
    virtual bool 	is_keyN () const=0;
    virtual bool 	is_keyM () const=0;
    virtual bool 	is_keySHIFTRIGHT () const=0;
    virtual bool 	is_keyARROWUP () const=0;
    virtual bool 	is_keyCTRLLEFT () const=0;
    virtual bool 	is_keyAPPLEFT () const=0;
    virtual bool 	is_keyALT () const=0;
    virtual bool 	is_keySPACE () const=0;
    virtual bool 	is_keyALTGR () const=0;
    virtual bool 	is_keyAPPRIGHT () const=0;
    virtual bool 	is_keyMENU () const=0;
    virtual bool 	is_keyCTRLRIGHT () const=0;
    virtual bool 	is_keyARROWLEFT () const=0;
    virtual bool 	is_keyARROWDOWN () const=0;
    virtual bool 	is_keyARROWRIGHT () const=0;
    virtual bool 	is_keyPAD0 () const=0;
    virtual bool 	is_keyPAD1 () const=0;
    virtual bool 	is_keyPAD2 () const=0;
    virtual bool 	is_keyPAD3 () const=0;
    virtual bool 	is_keyPAD4 () const=0;
    virtual bool 	is_keyPAD5 () const=0;
    virtual bool 	is_keyPAD6 () const=0;
    virtual bool 	is_keyPAD7 () const=0;
    virtual bool 	is_keyPAD8 () const=0;
    virtual bool 	is_keyPAD9 () const=0;
    virtual bool 	is_keyPADADD () const=0;
    virtual bool 	is_keyPADSUB () const=0;
    virtual bool 	is_keyPADMUL () const=0;
    virtual bool 	is_keyPADDIV () const=0;
    virtual int 	width () const=0;
    virtual int 	height () const=0;
    virtual unsigned int 	normalization () const=0;
    virtual const char * 	title () const=0;
    virtual int 	window_width () const=0;
    virtual int 	window_height () const=0;
    virtual int 	window_x () const=0;
    virtual int 	window_y () const=0;
    virtual int 	mouse_x () const=0;
    virtual int 	mouse_y () const=0;
    virtual unsigned int 	button () const=0;
    virtual int 	wheel () const=0;
    virtual unsigned int 	key (const unsigned int pos=0) const=0;
    virtual unsigned int 	released_key (const unsigned int pos=0) const=0;
    virtual float 	frames_per_second ()=0;
    virtual int 	screen_width ()=0;
    virtual int 	screen_height ()=0;
    virtual unsigned int 	keycode (const char *const keycode)=0;
    virtual MatNDisplayInterface & 	show ()=0;
    virtual MatNDisplayInterface & 	close ()=0;
    virtual MatNDisplayInterface & 	move (const int pos_x, const int pos_y)=0;
    virtual MatNDisplayInterface & 	resize (const  bool force_redraw=true)=0;
    virtual MatNDisplayInterface & 	resize (const int width, const int height, const bool force_redraw=true)=0;
    virtual MatNDisplayInterface & 	set_normalization (const unsigned int normalization)=0;
    virtual MatNDisplayInterface & 	set_title (const char *const format,...)=0;
    virtual MatNDisplayInterface & 	set_fullscreen (const bool is_fullscreen, const bool force_redraw=true)=0;
    virtual MatNDisplayInterface & 	toggle_fullscreen (const bool force_redraw=true)=0;
    virtual MatNDisplayInterface & 	show_mouse ()=0;
    virtual MatNDisplayInterface & 	hide_mouse ()=0;
    virtual MatNDisplayInterface & 	set_mouse (const int pos_x, const int pos_y)=0;
    virtual MatNDisplayInterface & 	set_button ()=0;
    virtual MatNDisplayInterface & 	set_button (const unsigned int button, const bool is_pressed=true)=0;
    virtual MatNDisplayInterface & 	set_wheel ()=0;
    virtual MatNDisplayInterface & 	set_wheel (const int amplitude)=0;
    virtual MatNDisplayInterface & 	set_key ()=0;
    virtual MatNDisplayInterface & 	set_key (const unsigned int keycode, const bool is_pressed=true)=0;
    virtual MatNDisplayInterface & 	flush ()=0;

    virtual MatNDisplayInterface & 	waitTime ()=0;
    virtual MatNDisplayInterface & 	waitTime (const unsigned int milliseconds)=0;
    virtual MatNDisplayInterface & 	paint ()=0;
};
POP_EXPORTS void waitKey( std::string text="");
class POP_EXPORTS MatNDisplay
{
private:
    MatNDisplayInterface * _impl;
public:
    MatNDisplay();
    MatNDisplay(const MatNDisplay& impl);
    MatNDisplay & 	operator= (const MatNDisplay &disp);
    virtual ~MatNDisplay ();
    virtual MatNDisplay & display(const MatN<2, RGBUI8 > &img);
//    virtual MatNDisplay & display(const MatN<2, RGBAUI8 > &img);
    virtual MatNDisplay & display(const MatN<2, UI8 > &img);
    template<typename PixelType>
    MatNDisplay & display(const MatN<3, PixelType > &m){
        MatN<2, PixelType > m_2d(m.sizeI(),m.sizeJ());
        std::copy(m.begin(),m.begin()+m.sizeI()*m.sizeJ(),m_2d.begin());
        this->display(m_2d);
        return *this;
    }
    template<int DIM,typename PixelType>
    MatNDisplay & display(const MatN<DIM, PixelType > &){return *this;}

    template<typename PixelType>
    MatNDisplay & display(const MatN<2, PixelType > &img){
        this->display(   MatN<2, UI8>  ( Processing::greylevelRange(img,0,255)));
        return *this;
    }

    virtual operator  bool () const;
    virtual bool 	is_empty () const;
    virtual bool 	is_closed () const;
    virtual bool 	is_resized () const;
    virtual bool 	is_moved () const;
    virtual bool 	is_event () const;
    virtual bool 	is_fullscreen () const;
    virtual bool 	is_key () const;
    virtual bool 	is_key (const unsigned int keycode_value) const;
    virtual bool 	is_key (const char *const keycode_value) const;
    virtual bool 	is_key_sequence (const unsigned int *const keycodes_sequence, const unsigned int length, const  bool remove_sequence=false);
    virtual bool 	is_keyESC () const;
    virtual bool 	is_keyF1 () const;
    virtual bool 	is_keyF2 () const;
    virtual bool 	is_keyF3 () const;
    virtual bool 	is_keyF4 () const;
    virtual bool 	is_keyF5 () const;
    virtual bool 	is_keyF6 () const;
    virtual bool 	is_keyF7 () const;
    virtual bool 	is_keyF8 () const;
    virtual bool 	is_keyF9 () const;
    virtual bool 	is_keyF10 () const;
    virtual bool 	is_keyF11 () const;
    virtual bool 	is_keyF12 () const;
    virtual bool 	is_keyPAUSE () const;
    virtual bool 	is_key1 () const;
    virtual bool 	is_key2 () const;
    virtual bool 	is_key3 () const;
    virtual bool 	is_key4 () const;
    virtual bool 	is_key5 () const;
    virtual bool 	is_key6 () const;
    virtual bool 	is_key7 () const;
    virtual bool 	is_key8 () const;
    virtual bool 	is_key9 () const;
    virtual bool 	is_key0 () const;
    virtual bool 	is_keyBACKSPACE () const;
    virtual bool 	is_keyINSERT () const;
    virtual bool 	is_keyHOME () const;
    virtual bool 	is_keyPAGEUP () const;
    virtual bool 	is_keyTAB () const;
    virtual bool 	is_keyQ () const;
    virtual bool 	is_keyW () const;
    virtual bool 	is_keyE () const;
    virtual bool 	is_keyR () const;
    virtual bool 	is_keyT () const;
    virtual bool 	is_keyY () const;
    virtual bool 	is_keyU () const;
    virtual bool 	is_keyI () const;
    virtual bool 	is_keyO () const;
    virtual bool 	is_keyP () const;
    virtual bool 	is_keyDELETE () const;
    virtual bool 	is_keyEND () const;
    virtual bool 	is_keyPAGEDOWN () const;
    virtual bool 	is_keyCAPSLOCK () const;
    virtual bool 	is_keyA () const;
    virtual bool 	is_keyS () const;
    virtual bool 	is_keyD () const;
    virtual bool 	is_keyF () const;
    virtual bool 	is_keyG () const;
    virtual bool 	is_keyH () const;
    virtual bool 	is_keyJ () const;
    virtual bool 	is_keyK () const;
    virtual bool 	is_keyL () const;
    virtual bool 	is_keyENTER () const;
    virtual bool 	is_keySHIFTLEFT () const;
    virtual bool 	is_keyZ () const;
    virtual bool 	is_keyX () const;
    virtual bool 	is_keyC () const;
    virtual bool 	is_keyV () const;
    virtual bool 	is_keyB () const;
    virtual bool 	is_keyN () const;
    virtual bool 	is_keyM () const;
    virtual bool 	is_keySHIFTRIGHT () const;
    virtual bool 	is_keyARROWUP () const;
    virtual bool 	is_keyCTRLLEFT () const;
    virtual bool 	is_keyAPPLEFT () const;
    virtual bool 	is_keyALT () const;
    virtual bool 	is_keySPACE () const;
    virtual bool 	is_keyALTGR () const;
    virtual bool 	is_keyAPPRIGHT () const;
    virtual bool 	is_keyMENU () const;
    virtual bool 	is_keyCTRLRIGHT () const;
    virtual bool 	is_keyARROWLEFT () const;
    virtual bool 	is_keyARROWDOWN () const;
    virtual bool 	is_keyARROWRIGHT () const;
    virtual bool 	is_keyPAD0 () const;
    virtual bool 	is_keyPAD1 () const;
    virtual bool 	is_keyPAD2 () const;
    virtual bool 	is_keyPAD3 () const;
    virtual bool 	is_keyPAD4 () const;
    virtual bool 	is_keyPAD5 () const;
    virtual bool 	is_keyPAD6 () const;
    virtual bool 	is_keyPAD7 () const;
    virtual bool 	is_keyPAD8 () const;
    virtual bool 	is_keyPAD9 () const;
    virtual bool 	is_keyPADADD () const;
    virtual bool 	is_keyPADSUB () const;
    virtual bool 	is_keyPADMUL () const;
    virtual bool 	is_keyPADDIV () const;
    virtual int 	width () const;
    virtual int 	height () const;
    virtual unsigned int 	normalization () const;
    virtual const char * 	title () const;
    virtual int 	window_width () const;
    virtual int 	window_height () const;
    virtual int 	window_x () const;
    virtual int 	window_y () const;
    virtual int 	mouse_x () const;
    virtual int 	mouse_y () const;
    virtual unsigned int 	button () const;
    virtual int 	wheel () const;
    virtual unsigned int 	key (const unsigned int pos) const;
    virtual unsigned int 	released_key (const unsigned int pos) const;
    virtual float 	frames_per_second ();
    virtual int 	screen_width ();
    virtual int 	screen_height ();
    virtual unsigned int 	keycode (const char *const keycode);
    virtual MatNDisplay & 	show ();
    virtual MatNDisplay & 	close ();
    virtual MatNDisplay & 	move (const int pos_x, const int pos_y);
    virtual MatNDisplay & 	resize (const  bool force_redraw=true);
    virtual MatNDisplay & 	resize (const int width_value, const int height_value, const bool force_redraw=true);
    virtual MatNDisplay & 	set_normalization (const unsigned int normalization_value);
    virtual MatNDisplay & 	set_title (const char *const format,...);
    virtual MatNDisplay & 	set_fullscreen (const bool is_fullscreen_value, const bool force_redraw=true);
    virtual MatNDisplay & 	toggle_fullscreen (const bool force_redraw=true);
    virtual MatNDisplay & 	show_mouse ();
    virtual MatNDisplay & 	hide_mouse ();
    virtual MatNDisplay & 	set_mouse (const int pos_x, const int pos_y);
    virtual MatNDisplay & 	set_button ();
    virtual MatNDisplay & 	set_button (const unsigned int button_value, const bool is_pressed=true);
    virtual MatNDisplay & 	set_wheel ();
    virtual MatNDisplay & 	set_wheel (const int amplitude);
    virtual MatNDisplay & 	set_key ();
    virtual MatNDisplay & 	set_key (const unsigned int keycode, const bool is_pressed=true);
    virtual MatNDisplay & 	flush ();

    virtual MatNDisplay & 	waitTime ();
    virtual MatNDisplay & 	waitTime (const unsigned int milliseconds);
    virtual MatNDisplay & 	paint ();

};


namespace Private {
template<int Dim, typename PixelType>
struct Display
{
    static std::vector<MatNDisplay> v_display;
};

template<int Dim, typename PixelType>
std::vector<MatNDisplay> Display< Dim,  PixelType>::v_display;
template<typename PixelType>
struct DisplayOutputPixel{inline static std::string   print( PixelType v){ return BasicUtility::Any2String(v); }};
template<>
struct DisplayOutputPixel<UI8>{ inline static  std::string   print( UI8 v){ return BasicUtility::Any2String((int)v); }};
template<>
struct DisplayOutputPixel<RGBUI8>{ inline static std::string   print( RGBUI8 v){ return BasicUtility::Any2String((int)v(0))+","+BasicUtility::Any2String((int)v(1))+","+BasicUtility::Any2String((int)v(2)); }};
}
template<int Dim, typename PixelType>
void MatN<Dim,PixelType>::display(const char * title,bool stop_process, bool automaticresize)const {

    MatN<Dim,PixelType>  img(*this);
    VecN<DIM,pop::F32> scale;
    scale =1;
    if(automaticresize ==true&&Dim==2){
        scale = scale*(600.f/img.getDomain()(0));
        img= GeometricalTransformation::scale(img,scale);
    }
    if(Dim==2){
        Private::Display< Dim, PixelType>::v_display.push_back(MatNDisplay());

        Private::Display< Dim, PixelType>::v_display.rbegin()->display(img);
        Private::Display< Dim, PixelType>::v_display.rbegin()->set_title(title);
        if(stop_process==true){
            while (!Private::Display< Dim, PixelType>::v_display.rbegin()->is_closed()) {
                Private::Display< Dim, PixelType>::v_display.rbegin()->waitTime();
                int iimg =(int)1.*Private::Display< Dim, PixelType>::v_display.rbegin()->mouse_y()/Private::Display< Dim, PixelType>::v_display.rbegin()->height()*img.getDomain()(0);
                int jimg =(int)1.*Private::Display< Dim, PixelType>::v_display.rbegin()->mouse_x()/Private::Display< Dim, PixelType>::v_display.rbegin()->width()*img.getDomain()(1);
                int i =(int)(Private::Display< Dim, PixelType>::v_display.rbegin()->mouse_y()*this->getDomain()(0)*1.0/img.getDomain()(0));
                int j =(int)(Private::Display< Dim, PixelType>::v_display.rbegin()->mouse_x()*this->getDomain()(1)*1.0/img.getDomain()(1));
                if (Private::Display< Dim, PixelType>::v_display.rbegin()->button()) {
                    if(img.isValid(iimg,jimg)){
                        std::string t ="i="+ BasicUtility::Any2String(i)+", j="+BasicUtility::Any2String(j)+", f(i,j)="+Private::DisplayOutputPixel<PixelType>::print(img(iimg,jimg));
                        Private::Display< Dim, PixelType>::v_display.rbegin()->set_title(t.c_str());
                    }
                }
            }
        }
    }else if(DIM==3){
        Vec2I32 d;
        d(0)=img.getDomain()(0);
        d(1)=img.getDomain()(1);
        MatN<2, PixelType> plane(d);
        typename MatN<2, PixelType>::IteratorEDomain it(plane.getIteratorEDomain());
        VecN<DIM,int> x;
        x(2)=img.getDomain()(2)/2;
        while(it.next()){
            x(0) = it.x()(0);
            x(1) = it.x()(1);
            plane(it.x())=img(x);
        }

        int index = x(2);
        std::string t =title;


        MatNDisplay main_disp;
        main_disp.display(plane);
        main_disp.set_title(t.c_str());
        main_disp.set_normalization(0);
        while (!main_disp.is_closed() ) {
            if(main_disp.is_keyARROWDOWN())
                index--;
            if(main_disp.is_keyARROWUP())
                index++;
            Vec3I32 xx(main_disp.mouse_y(),main_disp.mouse_x(),index);
            t = "i="+BasicUtility::Any2String((int)(xx(0)))+
                    ", j="+BasicUtility::Any2String((int)(xx(1)))+
                    ", k="+BasicUtility::Any2String((int)(xx(2)));
            if(img.isValid(xx)){
                t+=", f(i,j,k)="+Private::DisplayOutputPixel<PixelType>::print(img(xx));
            }
            t+=" and press  down(up)-arrow to move in the z axis";
            main_disp.set_title(t.c_str());
            if(index>=img.getDomain()[2])
                index = img.getDomain()[2]-1;
            if(index<0)
                index = 0;
            x(2)=index;
            it.init();
            while(it.next()){
                x(0) = it.x()(0);
                x(1) = it.x()(1);
                plane(it.x())=img(x);
            }
            main_disp.display(plane).waitTime(20);
        }
    }
}
}

#endif// MatNDISPLAY_H
