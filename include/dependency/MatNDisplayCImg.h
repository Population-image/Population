#ifndef MATNDISPLAYCIMG_H
#define MATNDISPLAYCIMG_H
#include"data/mat/MatNDisplay.h"
#include"PopulationConfig.h"
#if defined(HAVE_CIMG)
#include"dependency/CImg.h"
namespace pop{
/*! \ingroup Matrix
* \defgroup MatNDisplayCImg  MatNDisplayCImg
* \brief class to display matrices in windows handling mouse and keyboard events
*/

class MatNDisplayCImg : public MatNDisplayInterface
{
    class impl;
    impl * _pImpl;
public:
    /*!
     * \class pop::MatNDisplayCImg
     * \brief a window which can display pop::MatN matrices and handles mouse and keyboard events
     * \author Tariel Vincent
     * \ingroup MatNDisplay
     *
     * As the implementation is similar to the class CImgDisplay,   for the documentation see http://cimg.sourceforge.net/reference/structcimg__library_1_1CImgDisplay.html .
     * \code
    Mat2UI8 img;
    img.load("../image/Lena.bmp");
    MatNDisplay disp;
    while(1==1){
        img = Processing::erosion(img,1);
        disp.display(img);//display the current image
    }
     * \endcode

*/
    ~MatNDisplayCImg ();
    MatNDisplayCImg ();
   // MatNDisplay (const unsigned int width, const unsigned int height, const char *const title=0, const unsigned int normalization=3, const bool is_fullscreen=false, const bool is_closed=false);



    MatNDisplayCImg (const MatNDisplayCImg &disp);
    MatNDisplayCImg & 	operator= (const MatNDisplayCImg &disp);
    operator bool () const;
    bool 	is_empty () const;
    bool 	is_closed () const;
    bool 	is_resized () const;
    bool 	is_moved () const;
    bool 	is_event () const;
    bool 	is_fullscreen () const;
    bool 	is_key () const;
    bool 	is_key (const unsigned int keycode) const;
    bool 	is_key (const char *const keycode) const;
    bool 	is_key_sequence (const unsigned int *const keycodes_sequence, const unsigned int length, const bool remove_sequence=false);
    bool 	is_keyESC () const;
    bool 	is_keyF1 () const;
    bool 	is_keyF2 () const;
    bool 	is_keyF3 () const;
    bool 	is_keyF4 () const;
    bool 	is_keyF5 () const;
    bool 	is_keyF6 () const;
    bool 	is_keyF7 () const;
    bool 	is_keyF8 () const;
    bool 	is_keyF9 () const;
    bool 	is_keyF10 () const;
    bool 	is_keyF11 () const;
    bool 	is_keyF12 () const;
    bool 	is_keyPAUSE () const;
    bool 	is_key1 () const;
    bool 	is_key2 () const;
    bool 	is_key3 () const;
    bool 	is_key4 () const;
    bool 	is_key5 () const;
    bool 	is_key6 () const;
    bool 	is_key7 () const;
    bool 	is_key8 () const;
    bool 	is_key9 () const;
    bool 	is_key0 () const;
    bool 	is_keyBACKSPACE () const;
    bool 	is_keyINSERT () const;
    bool 	is_keyHOME () const;
    bool 	is_keyPAGEUP () const;
    bool 	is_keyTAB () const;
    bool 	is_keyQ () const;
    bool 	is_keyW () const;
    bool 	is_keyE () const;
    bool 	is_keyR () const;
    bool 	is_keyT () const;
    bool 	is_keyY () const;
    bool 	is_keyU () const;
    bool 	is_keyI () const;
    bool 	is_keyO () const;
    bool 	is_keyP () const;
    bool 	is_keyDELETE () const;
    bool 	is_keyEND () const;
    bool 	is_keyPAGEDOWN () const;
    bool 	is_keyCAPSLOCK () const;
    bool 	is_keyA () const;
    bool 	is_keyS () const;
    bool 	is_keyD () const;
    bool 	is_keyF () const;
    bool 	is_keyG () const;
    bool 	is_keyH () const;
    bool 	is_keyJ () const;
    bool 	is_keyK () const;
    bool 	is_keyL () const;
    bool 	is_keyENTER () const;
    bool 	is_keySHIFTLEFT () const;
    bool 	is_keyZ () const;
    bool 	is_keyX () const;
    bool 	is_keyC () const;
    bool 	is_keyV () const;
    bool 	is_keyB () const;
    bool 	is_keyN () const;
    bool 	is_keyM () const;
    bool 	is_keySHIFTRIGHT () const;
    bool 	is_keyARROWUP () const;
    bool 	is_keyCTRLLEFT () const;
    bool 	is_keyAPPLEFT () const;
    bool 	is_keyALT () const;
    bool 	is_keySPACE () const;
    bool 	is_keyALTGR () const;
    bool 	is_keyAPPRIGHT () const;
    bool 	is_keyMENU () const;
    bool 	is_keyCTRLRIGHT () const;
    bool 	is_keyARROWLEFT () const;
    bool 	is_keyARROWDOWN () const;
    bool 	is_keyARROWRIGHT () const;
    bool 	is_keyPAD0 () const;
    bool 	is_keyPAD1 () const;
    bool 	is_keyPAD2 () const;
    bool 	is_keyPAD3 () const;
    bool 	is_keyPAD4 () const;
    bool 	is_keyPAD5 () const;
    bool 	is_keyPAD6 () const;
    bool 	is_keyPAD7 () const;
    bool 	is_keyPAD8 () const;
    bool 	is_keyPAD9 () const;
    bool 	is_keyPADADD () const;
    bool 	is_keyPADSUB () const;
    bool 	is_keyPADMUL () const;
    bool 	is_keyPADDIV () const;
    int 	width () const;
    int 	height () const;
    unsigned int 	normalization () const;
    const char * 	title () const;
    int 	window_width () const;
    int 	window_height () const;
    int 	window_x () const;
    int 	window_y () const;
    int 	mouse_x () const;
    int 	mouse_y () const;
    unsigned int 	button () const;
    int 	wheel () const;
    unsigned int 	key (const unsigned int pos=0) const;
    unsigned int 	released_key (const unsigned int pos=0) const;
    float 	frames_per_second ();
    int 	screen_width ();
    int 	screen_height ();
    unsigned int 	keycode (const char *const keycode);

    MatNDisplayCImg & display(const MatN<2, RGBUI8 > &img);
    MatNDisplayCImg & display(const MatN<2, RGBAUI8 > &img);
    MatNDisplayCImg & display(const MatN<2, UI8 > &img);
    MatNDisplayCImg & display(const MatN<2, UI16 > &img);
    MatNDisplayCImg & display(const MatN<2, UI32 > &img);
    MatNDisplayCImg & display(const MatN<2, F64 > &img);

   template<int DIM,typename Type>
    MatNDisplayCImg & display(const MatN<DIM,Type >&  ){
        std::cerr<<"Cannot display this image with this pixel type. For 2D, you have to convert it in UI8 or F64 pixel type before to display it";
        return *this;
    }

    MatNDisplayCImg & 	show ();
    MatNDisplayCImg & 	close ();
    MatNDisplayCImg & 	move (const int pos_x, const int pos_y);
    MatNDisplayCImg & 	resize (const bool force_redraw=true);
    MatNDisplayCImg & 	resize (const int width, const int height, const bool force_redraw=true);
    MatNDisplayCImg & 	resize (const MatNDisplayCImg &disp, const bool force_redraw=true);
    MatNDisplayCImg & 	set_normalization (const unsigned int normalization);
    MatNDisplayCImg & 	set_title (const char *const format,...);
    MatNDisplayCImg & 	set_fullscreen (const bool is_fullscreen, const bool force_redraw=true);
    MatNDisplayCImg & 	toggle_fullscreen (const bool force_redraw=true);
    MatNDisplayCImg & 	show_mouse ();
    MatNDisplayCImg & 	hide_mouse ();
    MatNDisplayCImg & 	set_mouse (const int pos_x, const int pos_y);
    MatNDisplayCImg & 	set_button ();
    MatNDisplayCImg & 	set_button (const unsigned int button, const bool is_pressed=true);
    MatNDisplayCImg & 	set_wheel ();
    MatNDisplayCImg & 	set_wheel (const int amplitude);
    MatNDisplayCImg & 	set_key ();
    MatNDisplayCImg & 	set_key (const unsigned int keycode, const bool is_pressed=true);
    MatNDisplayCImg & 	flush ();

    MatNDisplayCImg & 	waitTime ();
    MatNDisplayCImg & 	waitTime (const unsigned int milliseconds);
    MatNDisplayCImg & 	paint ();
    void 	waitTime (MatNDisplayCImg &disp1);
    void 	waitTime (MatNDisplayCImg &disp1, MatNDisplayCImg &disp2);
    void 	waitTime (MatNDisplayCImg &disp1, MatNDisplayCImg &disp2, MatNDisplayCImg &disp3);
    void 	waitTime (MatNDisplayCImg &disp1, MatNDisplayCImg &disp2, MatNDisplayCImg &disp3, MatNDisplayCImg &disp4);
    void 	waitTime (MatNDisplayCImg &disp1, MatNDisplayCImg &disp2, MatNDisplayCImg &disp3, MatNDisplayCImg &disp4, MatNDisplayCImg &disp5);
    void 	wait_all ();
};
}
#endif
#endif // MATNDISPLAYCIMG_H
