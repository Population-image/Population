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
#ifndef DRAW_H
#define DRAW_H
#include<vector>
#include<cmath>

#include"data/distribution/Distribution.h"
#include"data/mat/MatN.h"
#include"data/mat/Mat2x.h"

#include"data/distribution/DistributionFromDataStructure.h"
#include"algorithm/GeometricalTransformation.h"
#include"algorithm/Statistics.h"
#include"algorithm/ProcessingAdvanced.h"
namespace pop
{

template<int DIM>
struct _caractere{
    static F32 _A_[63],_B_[63], _C_[63],_D_[63],_E_[63],_F_[63],_G_[63],_H_[63],_I_[63],_J_[63],_K_[63],_L_[63],_M_[63],_N_[63],_O_[63],_P_[63],_Q_[63],_R_[63],_S_[63],_T_[63],_U_[63],_V_[63],_W_[63],_X_[63],_Y_[63],_Z_[63],_0_[54],_1_[54],_2_[54],_3_[54],_4_[54],_5_[54],_6_[54],_7_[54],_8_[54],_9_[54],_virgule_[63],_point_[63],_dash_[63],_apostrope_[63],_colons_[63],_space_[63],_parenthese_open_[63],_parenthese_close_[63],_div_[63],_mult_[63],_sub_[63],_plus_[63],_exclamation_[63],_interrogation_[63];
    static Mat2F32 LetterA,LetterB,LetterC,LetterD,LetterE,LetterF,LetterG,LetterH,LetterI,LetterJ,LetterK,LetterL,LetterM,LetterN,LetterO,LetterP,LetterQ,LetterR,LetterS,LetterT,LetterU,LetterV,LetterW,LetterX,LetterY,LetterZ,Letter0,Letter1,Letter2,Letter3,Letter4,Letter5,Letter6,Letter7,Letter8,Letter9,Lettervirgule,Letterpoint,Letterdash,Letterapostrope,Lettercolons,Letterspace,Letterparenthese_open,Letterparenthese_close,Letterdiv,Lettermult,Lettersub,Letterplus,Letterexclamation,Letterinterrogation;

    static F32 _a_[63];
    static F32 _b_[63],_c_[63],_d_[63],_e_[63],_f_[63],_g_[63],_h_[63],_i_[63],_j_[63],_k_[63],_l_[63],_m_[63],_n_[63],_o_[63],_p_[63],_q_[63],_r_[63],_s_[63],_t_[63],_u_[63],_v_[63],_w_[63],_x_[63],_y_[63],_z_[63];
    static Mat2F32 Lettera,Letterb,Letterc,Letterd,Lettere,Letterf,Letterg,Letterh,Letteri,Letterj,Letterk,Letterl,Letterm,Lettern,Lettero,Letterp,Letterq,Letterr,Letters,Lettert,Letteru,Letterv,Letterw,Letterx,Lettery,Letterz;
};




/*!
\defgroup Draw Draw
\ingroup Algorithm
\brief Matrix In -> Matrix Out (drawing objects as line circle)

In this example, I draw a circle for each key point
\code
Mat2RGBUI8 lena;
lena.load((std::string(POP_PROJECT_SOURCE_DIR)+"/image/Lena.bmp").c_str());
Vec<KeyPoint<2> > keypoints = Feature::keyPointHarris(lena);
for(int i=0;i<keypoints.size();i++){
    Draw::circle(lena,keypoints[i].x(),4,RGBUI8::randomRGB(),2);
}
lena.display();
\endcode
\image html lenaharris.jpg
*/
struct POP_EXPORTS Draw
{

    /*!
        \class pop::Draw
        \ingroup Draw
        \brief Draw objects on the matrix
        \author Tariel Vincent
    */
    //-------------------------------------
    //
    //! \name Some geometrical figures
    //@{
    //-------------------------------------

    /*!
     * \brief draw a circle on a matrix
     * \param img  input/output function
     * \param x center of the circle
     * \param radius radius of the circle
     * \param value value for the circle
     * \param widthline with of the circle
     *
     * \code
     * Mat2RGBUI8 lena;
     * lena.load((std::string(POP_PROJECT_SOURCE_DIR)+"/image/Lena.bmp").c_str());
     * Vec<KeyPoint<2> > keypoints = Feature::keyPointHarris(lena);
     * for(int i=0;i<keypoints.size();i++){
     *     Draw::circle(lena,keypoints[i].x(),4,RGBUI8::randomRGB(),2);
     * }
     * lena.display();
     * \endcode
     * \image html lenaharris.jpg
    */
    template<int DIM, typename PixelType>
    static void circle(MatN<DIM,PixelType> &img,const typename MatN<DIM,PixelType>::E & x,int radius,typename MatN<DIM,PixelType>::F value,int widthline=1)
    {

        if(DIM==2){
            F32 anglestep = 1./(radius*3.14);
            F32 anglesum=0;

            Vec2F32 xzero(1,0);
            Mat2x22F32 mrot = GeometricalTransformation::rotation2D(anglestep);
            Mat2x22F32 mrotsum;
            mrotsum(0,0)=1;mrotsum(1,1)=1;

            while(anglesum<2*3.14){
                Vec2F32 xx;
                xx=(mrotsum*xzero);
                for(int r=0;r<widthline;r++){
                    Vec2F32 xxx;
                    xxx= xx*(radius+r);
                    xxx+=x;
                    xxx=round(xxx);
                    if(img.isValid(xxx(0),xxx(1)))
                        img(xxx)=value;
                }
                anglesum+=anglestep;
                mrotsum=mrotsum*mrot;
            }
        }else{
            F32 anglestep = 1./(radius*3.14);
            F32 anglesum=0;

            Vec3F32 xzero(1,0,0);
            Mat2x33F32 mrot = GeometricalTransformation::rotation3D(anglestep,2);
            Mat2x33F32 mrotsum;
            mrotsum(0,0)=1;mrotsum(1,1)=1;mrotsum(2,2)=1;

            while(anglesum<2*3.14){
                Vec3F32 xx;
                xx=(mrotsum*xzero);
                for(int r=0;r<widthline;r++){
                    Vec3F32 xxx;
                    xxx= xx*(radius+r);
                    xxx+=x;
                    xxx=round(xxx);
                    if(img.isValid(xxx(0),xxx(1),xxx(2)))
                        img(xxx)=value;
                }
                anglesum+=anglestep;
                mrotsum=mrotsum*mrot;
            }

        }
    }
    /*!
     * \brief draw a disk (circle filled) on a matrix
     * \param img  input/output function
     * \param x center of the circle
     * \param radius radius of the circle
     * \param value pixel value for the circle
     * \param boundarycondition boundary condition (MATN_BOUNDARY_CONDITION_BOUNDED=bounded, MATN_BOUNDARY_CONDITION_PERIODIC=Periodic)
     *
     * \code
     * Mat2RGBUI8 lena;
     * lena.load((std::string(POP_PROJECT_SOURCE_DIR)+"/image/Lena.bmp").c_str());
     * Vec<KeyPoint<2> > keypoints = Feature::keyPointHarris(lena);
     * for(int i=0;i<keypoints.size();i++){
     *     Draw::circle(lena,keypoints[i].x(),4,RGBUI8::randomRGB(),2);
     * }
     * lena.display();
     * \endcode
     * \image html lenaharris2.jpg

    */
    template< int DIM,typename PixelType>
    static void disk(MatN<DIM,PixelType> &img,const typename MatN<DIM,PixelType>::E & x,int radius,typename MatN<DIM,PixelType>::F value,MatNBoundaryConditionType boundarycondition=MATN_BOUNDARY_CONDITION_PERIODIC)
    {
        int radiuspower2=radius*radius;
        typename MatN<DIM,PixelType>::E xmin = x-radius;
        typename MatN<DIM,PixelType>::E xmax = x+radius;
        typename MatN<DIM,PixelType>::IteratorERectangle it = img.getIteratorERectangle(xmin,xmax);
        MatNBoundaryCondition condition(boundarycondition);
        while(it.next())
        {
            typename MatN<DIM,PixelType>::E xx = it.x();
            if((xx-x).normPower()<=radiuspower2){
                if(condition.isValid(img.getDomain(),xx)){
                    condition.apply(img.getDomain(),xx);
                    img(xx)=value;
                }
            }
        }
    }

    /*!
     * \brief draw a rectangle
     * \param img  input/output matrix
     * \param center center of the rectangle
     * \param size size of the rectangle
     * \param value pixel/voxel value
     * \param widthline with of the line
     * \param angle_radian rotate the rectangle in trigonometric direction using radian
     *
     * \code
     * Mat2RGBUI8 lena;
     * lena.load((std::string(POP_PROJECT_SOURCE_DIR)+"/image/Lena.bmp").c_str());
     * Draw::rectangle(lena,Vec2I32(150,100),Vec2I32(60,20),RGBUI8(255,0,0),2,PI/8);
     * lena.display();
     * \endcode
     * \image html lenarec.jpg
    */
    template<int DIM,typename PixelType>
    static void rectangle(MatN<DIM,PixelType> &img,const typename  MatN<DIM,PixelType>::E & center,const typename  MatN<DIM,PixelType>::E & size,typename MatN<DIM,PixelType>::F value,int widthline=1,F32 angle_radian=0 )
    {
        pop::Mat2F32 mrot = GeometricalTransformation::rotation2D(-angle_radian);

        VecN<DIM,F32> x1,x2,x3,x4;
        x1(0)=-1.f*size(0)/2;x1(1)=-1.f*size(1)/2;
        x2(0)= 1.f*size(0)/2;x2(1)=-1.f*size(1)/2;
        x3(0)= 1.f*size(0)/2;x3(1)= 1.f*size(1)/2;
        x4(0)=-1.f*size(0)/2;x4(1)= 1.f*size(1)/2;
        x1 = mrot*x1;x2 = mrot*x2; x3 = mrot*x3; x4 = mrot*x4;
        x1 +=center; x2 +=center;  x3 +=center;  x4 +=center;

        Draw::line(img,x1,x2,value,widthline);
        Draw::line(img,x2,x3,value,widthline);
        Draw::line(img,x3,x4,value,widthline);
        Draw::line(img,x4,x1,value,widthline);

    }
    /*!
     * \brief draw a polygon
     * \param img  input/output matrix
     * \param pos list of vertices x_0,x_1,...x_n
     * \param value size of the rectangle
     * \param value value of the line
     * \param widthline with of the line
     * \param bullet add bullet in vertices
     *
     * \code
     * Mat2RGBUI8 lena;
     * lena.load((std::string(POP_PROJECT_SOURCE_DIR)+"/image/Lena.bmp").c_str());
     * Vec<Vec2I32> points;
     * points.push_back(Vec2I32(20,40));
     * points.push_back(Vec2I32(180,20));
     * points.push_back(Vec2I32(140,200));
     * points.push_back(Vec2I32(70,200));
     * Draw::polygone(lena,points,RGBUI8(255,0,0),2,0);
     * \endcode
     * \image html lenapoly.jpg
    */
    template<int DIM,typename PixelType>
    static void polygone(MatN<DIM,PixelType> &img,const std::vector<typename  MatN<DIM,PixelType>::E> & pos,typename MatN<DIM,PixelType>::F value,int widthline=1,bool bullet=false )
    {
        for(unsigned int i =0;i<pos.size();i++){
            typename  MatN<DIM,PixelType>::E x1,x2;
            x1 = pos[i];
            if(i<pos.size()-1)
                x2 = pos[i+1];
            else
                x2 = pos[0];
            Draw::line(img,x1,x2,value,widthline);
            if(bullet==true)
                Draw::circle(img,x1,widthline+2,value);
        }

    }
    /*!
     * \brief draw a line between two points
     * \param img  input/output function
     * \param x1 first point
     * \param x2 second point
     * \param value value of the line
     * \param width with of the line
     *
     * \code
     * Mat2UI8 lena;
     * lena.load((std::string(POP_PROJECT_SOURCE_DIR)+"/image/Lena.bmp").c_str());
     * Vec2I32 x1(10,40);
     * Vec2I32 x2(152,200);
     * Draw::line(lena,x1,x2,255,1);
     * img.display();
     * \endcode
     * \image html lenatext.jpg
    */
    template<int DIM,typename PixelType>
    static void line(MatN<DIM,PixelType> &img,typename MatN<DIM,PixelType>::E x1, typename MatN<DIM,PixelType>::E x2,typename MatN<DIM,PixelType>::F value,int width=1)
    {
        if(width<1)
            width=1;
        typedef VecN<DIM,F32> Point_Float;
        F32 distance = normValue(x2-x1);
        Point_Float d = Point_Float(x2-x1)/normValue(x2-x1);
        for(int r =0;r<=distance;r++){
            Point_Float x = Point_Float(x1)+ d*(F32)r;
            typename MatN<DIM,PixelType>::IteratorENeighborhood itn(img.getIteratorENeighborhood(width-1,2));
            itn.init(x);
            while(itn.next())
            {
                img(itn.x())=value;
            }
        }
    }
    /*!
     * \brief draw an arrow between two points
     * \param img  input/output function
     * \param x1 first point
     * \param x2 second point
     * \param value value of the arrow
     * \param widthline with of the line
     * \param linelength length of the arrow at the end of the line
     * \param angleradian arrow angle
     *
     * \code
     * Mat2UI8 lena;
     * lena.load((std::string(POP_PROJECT_SOURCE_DIR)+"/image/Lena.bmp").c_str());
     * Vec2I32 x1(10,40);
     * Vec2I32 x2(152,200);
     * Draw::arrow(lena,x1,x2,255,1);
     * img.display();
     * \endcode
     * \image html lenaarrow.jpg
    */
    template<int DIM,typename PixelType>
    static void arrow(MatN<DIM,PixelType> &img,typename MatN<DIM,PixelType>::E x1,typename MatN<DIM,PixelType>::E x2,typename MatN<DIM,PixelType>::F value,int widthline=1,int linelength=6,F32 angleradian=0.5)
    {
        line(img, x1,  x2, value, widthline);

        VecF32 v1(x1);
        VecF32 v2(x2);

        VecF32 d = v1-v2;
        d/=d.norm();
        Mat2F32 Mangle = GeometricalTransformation::rotation2D(angleradian);
        VecF32 d1 = Mangle*d ;
        for(int i=0;i<2*linelength;i++){
            VecF32 v = v2+ d1*(i*0.5);
            if(img.isValid(v(0),v(1)))
                img(v(0),v(1))=value;
        }
        Mangle = GeometricalTransformation::rotation2D(-angleradian);
        d1 = Mangle*d ;
        for(int i=0;i<2*linelength;i++){
            VecF32 v = v2+ d1*(i*0.5);
            if(img.isValid(v(0),v(1)))
                img(v(0),v(1))=value;
        }
    }
    /*!
     * \brief draw text on the matrix (very simple draw :)
     * \param img  in/out matrix
     * \param text  text to write
     * \param pos position to start the text writing
     * \param value_text pixel value of the text
     * \param scale_factor scale factor  for the fint size
     * \param angle_radian rotate the text from the start position in trigonometric direction using radian (scale_factor should be superior to 2)
     *
     *
     * \code
     * Mat2RGBUI8 lena;
     * lena.load((std::string(POP_PROJECT_SOURCE_DIR)+"/image/Lena.bmp").c_str());;
     * string str ="Join the wonderful Population community !!!";
     * Draw::text(lena,str,Vec2I32(50,200),RGBUI8(255,0,0),1.5);
     * \endcode
     * \image html lenatext.jpg
    */

    template<typename PixelType>
    static void text( MatN<2,PixelType> & img, std::string text,const typename MatN<2,PixelType>::E&  pos,const typename MatN<2,PixelType>::F & value_text,int scale_factor=1,F32 angle_radian=0){
        Vec2F32 x = pos;
        for(unsigned int i =0;i<text.size();i++){
            char letter = text[i];
            Mat2F32 mask_letter;
            switch (letter)
            {
            case '?': mask_letter =pop::_caractere<0>::Letterinterrogation ;
                break;
            case '!': mask_letter =pop::_caractere<0>::Letterexclamation ;
                break;
            case ',': mask_letter =pop::_caractere<0>::Lettervirgule ;
                break;
            case '.': mask_letter =pop::_caractere<0>::Letterpoint ;
                break;
            case '_': mask_letter =pop::_caractere<0>::Letterdash ;
                break;
            case '\'': mask_letter =pop::_caractere<0>::Letterapostrope ;
                break;
            case ':': mask_letter =pop::_caractere<0>::Lettercolons ;
                break;
            case ' ': mask_letter =pop::_caractere<0>::Letterspace ;
                break;
            case '(': mask_letter =pop::_caractere<0>::Letterparenthese_open ;
                break;
            case ')': mask_letter =pop::_caractere<0>::Letterparenthese_close ;
                break;
            case '/': mask_letter =pop::_caractere<0>::Letterdiv ;
                break;
            case '*': mask_letter =pop::_caractere<0>::Lettermult ;
                break;
            case '-': mask_letter =pop::_caractere<0>::Lettersub ;
                break;
            case '+': mask_letter =pop::_caractere<0>::Letterplus ;
                break;
            case 'a': mask_letter =pop::_caractere<0>::Lettera ;
                break;
            case 'b': mask_letter =pop::_caractere<0>::Letterb ;
                break;
            case 'c': mask_letter =pop::_caractere<0>::Letterc ;
                break;
            case 'd': mask_letter =pop::_caractere<0>::Letterd ;
                break;
            case 'e': mask_letter =pop::_caractere<0>::Lettere ;
                break;
            case 'f': mask_letter =pop::_caractere<0>::Letterf ;
                break;
            case 'g': mask_letter =pop::_caractere<0>::Letterg ;
                break;
            case 'h': mask_letter =pop::_caractere<0>::Letterh ;
                break;
            case 'i': mask_letter =pop::_caractere<0>::Letteri ;
                break;
            case 'j': mask_letter =pop::_caractere<0>::Letterj ;
                break;
            case 'k': mask_letter =pop::_caractere<0>::Letterk ;
                break;
            case 'l': mask_letter =pop::_caractere<0>::Letterl ;
                break;
            case 'm': mask_letter =pop::_caractere<0>::Letterm ;
                break;
            case 'n': mask_letter =pop::_caractere<0>::Lettern ;
                break;
            case 'o': mask_letter =pop::_caractere<0>::Lettero ;
                break;
            case 'p': mask_letter =pop::_caractere<0>::Letterp ;
                break;
            case 'q': mask_letter =pop::_caractere<0>::Letterq ;
                break;
            case 'r': mask_letter =pop::_caractere<0>::Letterr ;
                break;
            case 's': mask_letter =pop::_caractere<0>::Letters ;
                break;
            case 't': mask_letter =pop::_caractere<0>::Lettert ;
                break;
            case 'u': mask_letter =pop::_caractere<0>::Letteru ;
                break;
            case 'v': mask_letter =pop::_caractere<0>::Letterv ;
                break;
            case 'w': mask_letter =pop::_caractere<0>::Letterw ;
                break;
            case 'x': mask_letter =pop::_caractere<0>::Letterx ;
                break;
            case 'y': mask_letter =pop::_caractere<0>::Lettery ;
                break;
            case 'z': mask_letter =pop::_caractere<0>::Letterz ;
                break;
            case 'A': mask_letter =pop::_caractere<0>::LetterA ;
                break;
            case 'B': mask_letter =pop::_caractere<0>::LetterB ;
                break;
            case 'C': mask_letter =pop::_caractere<0>::LetterC ;
                break;
            case 'D': mask_letter =pop::_caractere<0>::LetterD ;
                break;
            case 'E': mask_letter =pop::_caractere<0>::LetterE ;
                break;
            case 'F': mask_letter =pop::_caractere<0>::LetterF ;
                break;
            case 'G': mask_letter =pop::_caractere<0>::LetterG ;
                break;
            case 'H': mask_letter =pop::_caractere<0>::LetterH ;
                break;
            case 'I': mask_letter =pop::_caractere<0>::LetterI ;
                break;
            case 'J': mask_letter =pop::_caractere<0>::LetterJ ;
                break;
            case 'K': mask_letter =pop::_caractere<0>::LetterK ;
                break;
            case 'L': mask_letter =pop::_caractere<0>::LetterL ;
                break;
            case 'M': mask_letter =pop::_caractere<0>::LetterM ;
                break;
            case 'N': mask_letter =pop::_caractere<0>::LetterN ;
                break;
            case 'O': mask_letter =pop::_caractere<0>::LetterO ;
                break;
            case 'P': mask_letter =pop::_caractere<0>::LetterP ;
                break;
            case 'Q': mask_letter =pop::_caractere<0>::LetterQ ;
                break;
            case 'R': mask_letter =pop::_caractere<0>::LetterR ;
                break;
            case 'S': mask_letter =pop::_caractere<0>::LetterS ;
                break;
            case 'T': mask_letter =pop::_caractere<0>::LetterT ;
                break;
            case 'U': mask_letter =pop::_caractere<0>::LetterU ;
                break;
            case 'V': mask_letter =pop::_caractere<0>::LetterV ;
                break;
            case 'W': mask_letter =pop::_caractere<0>::LetterW ;
                break;
            case 'X': mask_letter =pop::_caractere<0>::LetterX ;
                break;
            case 'Y': mask_letter =pop::_caractere<0>::LetterY ;
                break;
            case 'Z': mask_letter =pop::_caractere<0>::LetterZ ;
                break;
            case '0': mask_letter =pop::_caractere<0>::Letter0 ;
                break;
            case '1': mask_letter =pop::_caractere<0>::Letter1 ;
                break;
            case '2': mask_letter =pop::_caractere<0>::Letter2 ;
                break;
            case '3': mask_letter =pop::_caractere<0>::Letter3 ;
                break;
            case '4': mask_letter =pop::_caractere<0>::Letter4 ;
                break;
            case '5': mask_letter =pop::_caractere<0>::Letter5 ;
                break;
            case '6': mask_letter =pop::_caractere<0>::Letter6 ;
                break;
            case '7': mask_letter =pop::_caractere<0>::Letter7 ;
                break;
            case '8': mask_letter =pop::_caractere<0>::Letter8 ;
                break;
            case '9': mask_letter =pop::_caractere<0>::Letter9 ;
                break;
            }
            int add = scale_factor*(mask_letter.getDomain()(0)-3);

            //            Draw::addBorder(mask_letter,1);
            Mat2F32 mrot = GeometricalTransformation::rotation2D(-angle_radian);
            Vec2F32 transletter(0,(F32)add);
            transletter =  mrot*transletter;
            Mat2F32::IteratorEDomain it(mask_letter.getDomain()*scale_factor);
            while(it.next()){
                Vec2F32 xletter = it.x();
                xletter= xletter/(F32)scale_factor;
                if(mask_letter(xletter)!=0){
                    Vec2F32 xx = mrot*Vec2F32(it.x());
                    xx+=x;
                    if(img.isValid(xx)){
                        img(xx)=value_text;
                    }
                }
            }
            x+=transletter;
        }
    }
    //@}
    //-------------------------------------
    //
    //! \name merge two matrices
    //@{
    //-------------------------------------
    /*!
     * \brief merge at horizontal two matrices
     * \param img1  input first function
     * \param img2  input second function
     * \return merge matrix
     *
     *
     * \code
     * Mat2RGBUI8 lena;
     * lena.load((std::string(POP_PROJECT_SOURCE_DIR)+"/image/Lena.bmp").c_str());;
     * int value;
     * Mat2RGBUI8 threshold = Processing::thresholdOtsuMethod(lena,value);
     * threshold = Draw::mergeTwoMatrixHorizontal(lena,threshold);
     * threshold.display();
     * \endcode
     * \image html lenamerge.jpg
    */

    template<int DIM,typename PixelType>
    static MatN<DIM,PixelType> mergeTwoMatrixHorizontal(const MatN<DIM,PixelType> & img1,const MatN<DIM,PixelType> & img2){
        typename MatN<DIM,PixelType>::E x = img1.getDomain();
        x(1)+=img2.getDomain()(1);
        MatN<DIM,PixelType> out(x);
        typename MatN<DIM,PixelType>::IteratorEDomain it(out.getIteratorEDomain());
        while(it.next()){
            x = it.x();
            if(x(1)<img1.getDomain()(1)){
                if(img1.isValid(it.x()))
                    out(it.x())=img1(it.x());
            }
            else{
                x(1)-=img1.getDomain()(1);
                if(img2.isValid(x))
                    out(it.x())=img2(x);
            }
        }
        return out;
    }
    /*!
     * \brief merge vertically two matrices
     * \param img1  input first function
     * \param img2  input second function
     * \return merge matrix
     *
     *
     * \code
     * Mat2RGBUI8 lena;
     * lena.load((std::string(POP_PROJECT_SOURCE_DIR)+"/image/Lena.bmp").c_str());;
     * int value;
     * Mat2RGBUI8 threshold = Processing::thresholdOtsuMethod(lena,value);
     * threshold = Draw::mergeTwoMatrixVertical(lena,threshold);
     * threshold.display();
     * \endcode
     * \image html lenamerge2.jpg
    */
    template<int DIM,typename PixelType>
    static MatN<DIM,PixelType> mergeTwoMatrixVertical(const MatN<DIM,PixelType> & img1,const MatN<DIM,PixelType> & img2){

        typename MatN<DIM,PixelType>::E x = img1.getDomain();
        x(0)+=img2.getDomain()(0);
        MatN<DIM,PixelType> out(x);
        typename MatN<DIM,PixelType>::IteratorEDomain it(out.getIteratorEDomain());
        while(it.next()){
            x = it.x();
            if(x(0)<img1.getDomain()(0)){
                if(img1.isValid(it.x()))
                    out(it.x())=img1(it.x());
            }
            else{
                x(0)-=img1.getDomain()(0);
                if(img2.isValid(x))
                    out(it.x())=img2(x);
            }
        }
        return out;
    }
    /*!
     *  \brief  Insert an matrix
     * \param f input function
     * \param postit small matrix to insert
     * \param trans position
     * \return output function noted h
     *
     *  Insert the matrix postit in the matrix f such that the center of the matrix postit is located at the position trans
     * \code
        Mat2RGBUI8 lena,lena_iterate,lenasmall;
        lena.load((std::string(POP_PROJECT_SOURCE_DIR)+"/image/Lena.bmp").c_str());
        lena_iterate = lena;
        lenasmall    = lena;
        lenasmall = GeometricalTransformation::scale(lenasmall,Vec2F32(0.05,0.05));
        MatNDisplay windows(lena_iterate);
        DistributionUniformInt di(0.,lena.getDomain()(0)-1);
        DistributionUniformInt dj(0.,lena.getDomain()(1)-1);
        while(!windows.is_closed()){
            Vec2I32 x ;
            x(0)=di.randomVariable();
            x(1)=dj.randomVariable();
            Mat2RGBUI8 temp = pop::Processing::greylevelTranslateMeanValue(lenasmall,lena(x));
            lena_iterate = Draw::insertMatrix(lena_iterate,temp,x);
            windows.display(lena_iterate);
        }
     * \endcode
     * \image html lenaiterate.jpg
    */
    template<int DIM, typename PixelType1,typename PixelType2>
    static MatN<DIM,PixelType1> insertMatrix(const MatN<DIM,PixelType1> & f,const MatN<DIM,PixelType2> & postit, typename MatN<DIM,PixelType1>::E trans)
    {
        MatN<DIM,PixelType1> h(f);
        typename MatN<DIM,PixelType1>::E  transs;
        typename FunctionTypeTraitsSubstituteF<typename MatN<DIM,PixelType1>::E,F32>::Result temptrans(trans);
        typename FunctionTypeTraitsSubstituteF<typename MatN<DIM,PixelType1>::E,F32>::Result tempgetdomain(postit.getDomain());

        transs=   typename MatN<DIM,PixelType1>::E (temptrans-0.5f*tempgetdomain);
        typename MatN<DIM,PixelType2>::IteratorEDomain it (postit.getIteratorEDomain());
        typename MatN<DIM,PixelType1>::E x;
        while(it.next()){

            x =it.x()+ transs;
            if(f.isValid(x)==true)
            {
                h.operator ()(x)=postit.operator ()(it.x());
            }
        }
        return h;
    }
    //@}
    //-------------------------------------
    //
    //! \name Others
    //@{
    //-------------------------------------
    /*!
    * \brief set a color the pixel of the selected face
    * \param h  input/output function
    * \param coordinate coordinate
    * \param leftorright  (left or down=0 and right or top=1)
    * \param value the affectation value
    *
    * \code
        Mat2UI8 eutel;
        eutel.load("../doc/image2/eutel.bmp");
        int value;
        eutel = Processing::thresholdOtsuMethod(eutel,value);
        eutel.display();
        Mat2UI8 face(eutel.getDomain());
        Draw::setFace(face,0,0,255);
        Mat2UI32 dist = Processing::distance(face,eutel);
        Visualization::labelToRGBGradation(dist).display();
    * \endcode
    * \image html eutel.bmp "original image"
    * \image html euteldist.jpg "distance function used to estimate the geometrical tortuosity"
    */
    template<int DIM,typename PixelType>
    static void setFace( MatN<DIM,PixelType> & h,int coordinate,int leftorright,typename MatN<DIM,PixelType>::F value=NumericLimits<PixelType>::maximumRange())
    {
        typename MatN<DIM,PixelType>::E d = h.getDomain();
        d(coordinate)=1;
        typename MatN<DIM,PixelType>::IteratorEDomain it (d);
        typename MatN<DIM,PixelType>::E x;
        while(it.next()){
            x = it.x();
            if(leftorright!=0)
                x(coordinate)=h.getDomain()(coordinate)-1;
            h.operator ()(x)=value;
        }
    }
    /*!
    \brief set at the given value all pixels near the border
    \param h  input/output function
    \param value value
    \param borderlenght  near = at this distance
    *
    * In this code, we remove all connected components touching the border of the image
    *
    * \code
    * pop::Mat2UI8 image;
    * image.load("D:/Users/Vincent/Desktop/eutel.bmp");
    * image =pop::Processing::threshold(image,125);
    * image = image.opposite();
    * image.display();
    * pop::Mat2UI8 border(image.getDomain());
    * pop::Draw::setBorder(border,1,1);
    * pop::Processing::threshold(border,1).display();
    * border = pop::Processing::voronoiTesselation(border,image);
    * pop::Processing::threshold(border,1).display();
    * image = image -pop::Processing::threshold(border,1);
    * image.display();
    * \endcode
    */
    template<int DIM,typename PixelType>
    static void setBorder( MatN<DIM,PixelType> & h,typename MatN<DIM,PixelType>::F value,int borderlenght=1 )
    {
        typename MatN<DIM,PixelType>::IteratorEDomain it (h.getIteratorEDomain());
        while(it.next()){
            for(int i=0;i<DIM;i++)
            {
                if(it.x()(i)<borderlenght)
                    h(it.x())=value;
                if(it.x()(i)>=h.getDomain()(i)-borderlenght)
                    h(it.x())=value;
            }
        }
    }
    /*!
     * \brief add a border with the given value to the input matrix
     * \param f  input function
     * \param widthborder width border
     * \param border_value value of the border
     * \param boundary_condition boundary condition
    *
    * add a border with the given value to the input matrix
    *
    * \code
    * Mat2RGBUI8 lena;
    * lena.load((std::string(POP_PROJECT_SOURCE_DIR)+"/image/Lena.bmp").c_str());;
    * Draw::addBorder(lena,50,RGBUI8(255,0,0));
    * lena.display();
    * \endcode
    * \image html lenaborderadd.jpg
    */
    template<int DIM,typename PixelType>
    static void addBorder( MatN<DIM,PixelType> & f,int widthborder, const typename MatN<DIM,PixelType>::F& border_value=PixelType(0),MatNBoundaryConditionType boundary_condition=MATN_BOUNDARY_CONDITION_BOUNDED)
    {
        if(DIM==2){
            MatN<DIM,PixelType> out(f.getDomain()+2*widthborder,border_value);
            if(boundary_condition==MATN_BOUNDARY_CONDITION_BOUNDED){
                int sift_in_core =0;
                int sift_out_core=widthborder*out.sizeJ()+widthborder;
                for(unsigned int i=0;i<f.sizeI();i++){
                    std::copy(f.begin()+sift_in_core,f.begin()+sift_in_core+f.sizeJ(),out.begin()+sift_out_core);
                    sift_in_core +=f.sizeJ();
                    sift_out_core+=out.sizeJ();
                }
            }
            else if(boundary_condition==MATN_BOUNDARY_CONDITION_MIRROR){
                int sift_in_core =0;
                int sift_out_core=widthborder*out.sizeJ();
                for(unsigned int i=0;i<f.sizeI();i++){
                    std::reverse_copy(f.begin()+sift_in_core,f.begin()+sift_in_core+widthborder,out.begin()+sift_out_core);
                    sift_out_core+=widthborder;
                    std::copy(f.begin()+sift_in_core,f.begin()+sift_in_core+f.sizeJ(),out.begin()+sift_out_core);
                    sift_in_core +=f.sizeJ();
                    sift_out_core+=f.sizeJ();
                    std::reverse_copy(f.begin()+sift_in_core-widthborder,f.begin()+sift_in_core,out.begin()+sift_out_core);
                    sift_out_core+=widthborder;
                }

                sift_in_core =(widthborder-1)*(f.sizeJ());
                sift_out_core=widthborder;
                for( int i=0;i<widthborder;i++){
                    std::copy(f.begin()+sift_in_core,f.begin()+sift_in_core+f.sizeJ(),out.begin()+sift_out_core);
                    sift_in_core -=f.sizeJ();
                    sift_out_core+=out.sizeJ();

                }
                sift_in_core =(f.sizeI()-1)*(f.sizeJ());
                sift_out_core=widthborder+(out.sizeI()-widthborder)*out.sizeJ();
                for( int i=0;i<widthborder;i++){
                    std::copy(f.begin()+sift_in_core,f.begin()+sift_in_core+f.sizeJ(),out.begin()+sift_out_core);
                    sift_in_core -=f.sizeJ();
                    sift_out_core+=out.sizeJ();

                }
            }
            f = out;
        }else{
            MatN<DIM,PixelType> h(f.getDomain()+2*widthborder,border_value);
            typename MatN<DIM,PixelType>::IteratorEDomain it (h.getIteratorEDomain());
            MatNBoundaryCondition c(boundary_condition);
            while(it.next()){
                typename MatN<DIM,PixelType>::E x = it.x()-widthborder;
                if(c.isValid(f.getDomain(),x)){
                    c.apply(f.getDomain(),x);
                    h(it.x())=f(x);
                }
            }
            f = h;
        }
    }


    /*!
     * \brief generate a calibration grid (a chess board)
     * \param sizej  horizontal size of the grid
     * \param sizei  vertical size of the grid
     * \param sizeceil length of the chess borad ceil
     * \return calibration grid
     *
     * \image html calibration.jpg
    */
    static inline Mat2UI8 calibrationGridChessBoard(int sizei,int sizej, int sizeceil){
        Mat2UI8 m(sizei,sizej);
        Mat2UI8::IteratorEDomain it = m.getIteratorEDomain();
        while(it.next())
        {
            Mat2UI8::E x = it.x();
            int vi = (x(0)/sizeceil)%2;
            int vj = (x(1)/sizeceil)%2;
            if(vi==vj)
                m(x)=255;
            else
                m(x)=0;
        }
        return m;
    }





    template<int DIM,typename PixelType>
    static void axis(MatN<DIM,PixelType> &img,F32 xmin,F32 xmax,F32 ymin, F32 ymax,typename MatN<DIM,PixelType>::F value)
    {

        int space = 15;
        int y= img.getDomain()(0)-space;
        if(ymin<0&&ymax>0)
            y = static_cast<int>((1.f*ymax)/(ymax-ymin)*img.getDomain()(0));
        else if(ymin<0){
            y= space;
        }



        //drax x-coordinate
        F32 xmm =maximum(absolute(xmin),absolute(xmax))*(1+0.0001f);
        int l = static_cast<int>(std::floor( std::log(xmm)/std::log((F32)10)));//max digit different to zero
        l -=2;//three significatif numbers
        F32 value2 = std::pow((F32)10,l);
        //Drawline
        for(int i=0;i<img.getDomain()(1);i++){
            img(y,i)=value;
        }
        //DrawNumber
        int BAR=10;
        for(int k=0;k<BAR;k++){
            int i =static_cast<int>(1.f*k*img.getDomain()(1)/BAR);
            img(y+1,i)=value;
            img(y-1,i)=value;
            F32 x = xmin + k*(xmax-xmin)/BAR;
            int v = static_cast<int>(x/value2);
            Draw::text(img,BasicUtility::Any2String(v),Vec2I32(y+5,i-7),value);
        }
        Draw::text(img,BasicUtility::Any2String(10),Vec2I32(y+5,img.getDomain()(1)-space-5),value);
        Draw::text(img,BasicUtility::Any2String(l),Vec2I32(y+3,img.getDomain()(1)-space+5),value);

        space=30;
        int x= 30;
        if(xmin<0&&xmax>0)
            x = static_cast<int>((-1.f*xmin)/(xmax-xmin)*img.getDomain()(1));
        else if(ymin<0){
            x= img.getDomain()(1)-space;
        }
        //drax y-coordinate
        F32 ymm =maximum(absolute(ymin),absolute(ymax))*(1+0.0001f);
        l = static_cast<int>(std::floor( std::log(ymm)/std::log((F32)10)));//max digit different to zero
        l -=2;//three significatif numbers
        value2 = std::pow((F32)10,l);
        //Drawline
        for(int j=0;j<img.getDomain()(0);j++){
            img(j,x)=value;
        }

        for(int k=1;k<BAR;k++){
            int j =static_cast<int>(1.f*k*img.getDomain()(0)/BAR);
            img(j,x+1)=value;
            img(j,x-1)=value;
            F32 y2 = ymax + k*(ymin-ymax)/BAR;//Buttom to up (inverse that the matrix order)
            int v = static_cast<int>(y2/value2);
            Draw::text(img,BasicUtility::Any2String(v),Vec2I32(j-6,x-28),value,1);
        }
        Draw::text(img,BasicUtility::Any2String(10),Vec2I32(5,x-28),value,1);
        Draw::text(img,BasicUtility::Any2String(l),Vec2I32(0,x-18),value,1);
    }
    template<int DIM,typename PixelType>
    static void distribution(const Distribution & d,F32 xmin,F32 xmax,F32 ymin,F32 ymax,typename MatN<DIM,PixelType>::F value,MatN<DIM,PixelType> &img)
    {
        int sizex = img.sizeJ();
        int sizey = img.sizeI();
        int jbefore=0;
        for(int i =0;i<sizex;i++){
            F32 x = xmin + i*(xmax-xmin)/sizex;
            F32 v =d.operator ()(x);
            int j = static_cast<int>((v-ymin)/(ymax-ymin)*sizey);
            j = sizey-j;
            if(i==0)
                jbefore = j;
            for(int jj=0;jj<=absolute(jbefore-j);jj++){
                int jjj= j+jj*sgn(jbefore-j);

                if(jjj>=0&&jjj<sizey)
                    img(jjj,i)=value;
            }
            jbefore = j;
        }

    }
    //@}




    pop::Mat2RGBUI8 MandelbrotSet(int size=512){
        pop::Mat2RGBUI8 m(2*size,static_cast<unsigned int>(3.5*size));
        unsigned int max_iteration = 1000;
        Vec<RGBUI8> v_palette;


        for(unsigned int i=0;i<max_iteration;i++){
            v_palette.push_back(RGBUI8::randomRGB());
        }


        for(int Px=0;Px<m.getDomain()(1);Px++ ){
            if(Px%100==0)
                std::cout<<Px<<std::endl;
            for(int Py=0;Py<m.getDomain()(0);Py++ ){

                //scale point  X scale (-2.5, 1))  and  Y scale (-1, 1))
                F32 x0 = Px*1.f/m.getDomain()(1)*3.5f-2.5f;
                F32 y0 = Py*1.f/m.getDomain()(0)*2.f-1.f;
                F32 x=0;
                F32 y=0;

                unsigned int iteration = 0;
                while (x*x + y*y < 2*2  &&  iteration < max_iteration )
                {
                    F32 xtemp =x*x - y*y + x0;
                    y = 2*x*y + y0;
                    x = xtemp;
                    iteration++;
                }
                if(iteration < max_iteration)
                    m(Py,Px)=v_palette[iteration];
                else
                    m(Py,Px)=RGBUI8(0,0,0);
            }

        }
        return m;
    }
};


template<int DIM>F32 _caractere<DIM>::_interrogation_[]={
    0,0,0,0,0,0,0,
    0,0,1,1,1,0,0,
    0,1,0,0,0,1,0,
    0,0,0,0,1,0,0,
    0,0,0,1,0,0,0,
    0,0,0,1,0,0,0,
    0,0,0,0,0,0,0,
    0,0,0,1,0,0,0,
    0,0,0,0,0,0,0
};

template<int DIM>Mat2F32 _caractere<DIM>::Letterinterrogation =Mat2F32(Vec2I32(9,7),_interrogation_);

template<int DIM>F32 _caractere<DIM>::_exclamation_[]={
    0,0,0,0,0,0,0,
    0,0,0,1,0,0,0,
    0,0,0,1,0,0,0,
    0,0,0,1,0,0,0,
    0,0,0,1,0,0,0,
    0,0,0,1,0,0,0,
    0,0,0,0,0,0,0,
    0,0,0,1,0,0,0,
    0,0,0,0,0,0,0
};

template<int DIM>Mat2F32 _caractere<DIM>::Letterexclamation =Mat2F32(Vec2I32(9,7),_exclamation_);



template<int DIM>F32 _caractere<DIM>::_plus_[]={
    0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,
    0,0,0,1,0,0,0,
    0,0,0,1,0,0,0,
    0,1,1,1,1,1,0,
    0,0,0,1,0,0,0,
    0,0,0,1,0,0,0,
    0,0,0,0,0,0,0,
    0,0,0,0,0,0,0
};

template<int DIM>Mat2F32 _caractere<DIM>::Letterplus =Mat2F32(Vec2I32(9,7),_plus_);

template<int DIM>F32 _caractere<DIM>::_sub_[]={
    0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,
    0,1,1,1,1,1,0,
    0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,
    0,0,0,0,0,0,0
};

template<int DIM>Mat2F32 _caractere<DIM>::Lettersub =Mat2F32(Vec2I32(9,7),_sub_);

template<int DIM>F32 _caractere<DIM>::_mult_[]={
    0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,
    0,1,0,1,0,1,0,
    0,0,1,1,1,0,0,
    0,1,1,1,1,1,0,
    0,0,1,1,1,0,0,
    0,1,0,1,0,1,0,
    0,0,0,0,0,0,0,
    0,0,0,0,0,0,0
};

template<int DIM>Mat2F32 _caractere<DIM>::Lettermult =Mat2F32(Vec2I32(9,7),_mult_);

template<int DIM>F32 _caractere<DIM>::_div_[]={
    0,0,0,0,0,0,0,
    0,0,0,0,0,1,0,
    0,0,0,0,1,0,0,
    0,0,0,0,1,0,0,
    0,0,0,1,0,0,0,
    0,0,1,0,0,0,0,
    0,0,1,0,0,0,0,
    0,1,0,0,0,0,0,
    0,0,0,0,0,0,0
};

template<int DIM>Mat2F32 _caractere<DIM>::Letterdiv =Mat2F32(Vec2I32(9,7),_div_);


template<int DIM>F32 _caractere<DIM>::_parenthese_close_[]={
    0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,
    0,0,0,1,0,0,0,
    0,0,0,0,1,0,0,
    0,0,0,0,1,0,0,
    0,0,0,0,1,0,0,
    0,0,0,1,0,0,0,
    0,0,1,0,0,0,0,
    0,0,0,0,0,0,0
};

template<int DIM>Mat2F32 _caractere<DIM>::Letterparenthese_close =Mat2F32(Vec2I32(9,7),_parenthese_close_);

template<int DIM>F32 _caractere<DIM>::_parenthese_open_[]={
    0,0,0,0,0,0,0,
    0,0,0,0,1,0,0,
    0,0,0,1,0,0,0,
    0,0,1,0,0,0,0,
    0,0,1,0,0,0,0,
    0,0,1,0,0,0,0,
    0,0,0,1,0,0,0,
    0,0,0,0,1,0,0,
    0,0,0,0,0,0,0
};

template<int DIM>Mat2F32 _caractere<DIM>::Letterparenthese_open =Mat2F32(Vec2I32(9,7),_parenthese_open_);


template<int DIM>F32 _caractere<DIM>::_space_[]={
    0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,
    0,0,0,0,0,0,0
};

template<int DIM>Mat2F32 _caractere<DIM>::Letterspace =Mat2F32(Vec2I32(9,7),_space_);


template<int DIM>F32 _caractere<DIM>::_colons_[]={
    0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,
    0,0,0,1,0,0,0,
    0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,
    0,0,0,1,0,0,0,
    0,0,0,0,0,0,0,
    0,0,0,0,0,0,0
};

template<int DIM>Mat2F32 _caractere<DIM>::Lettercolons =Mat2F32(Vec2I32(9,7),_colons_);

template<int DIM>F32 _caractere<DIM>::_apostrope_[]={
    0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,
    0,0,0,1,0,0,0,
    0,0,0,1,0,0,0,
    0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,
    0,0,0,0,0,0,0
};

template<int DIM>Mat2F32 _caractere<DIM>::Letterapostrope =Mat2F32(Vec2I32(9,7),_apostrope_);

template<int DIM>F32 _caractere<DIM>::_dash_[]={
    0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,
    0,1,1,1,1,1,0,
    0,0,0,0,0,0,0
};

template<int DIM>Mat2F32 _caractere<DIM>::Letterdash =Mat2F32(Vec2I32(9,7),_dash_);


template<int DIM>F32 _caractere<DIM>::_point_[]={
    0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,
    0,0,0,1,0,0,0,
    0,0,0,0,0,0,0
};

template<int DIM>Mat2F32 _caractere<DIM>::Letterpoint =Mat2F32(Vec2I32(9,7),_point_);

template<int DIM>F32 _caractere<DIM>::_virgule_[]={
    0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,
    0,0,0,0,1,0,0,
    0,0,0,1,0,0,0,
    0,0,0,0,0,0,0
};

template<int DIM>Mat2F32 _caractere<DIM>::Lettervirgule =Mat2F32(Vec2I32(9,7),_virgule_);


template<int DIM>F32 _caractere<DIM>::_a_[]={
    0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,
    0,0,1,1,1,0,0,
    0,0,0,0,0,1,0,
    0,0,1,1,1,1,0,
    0,1,0,0,0,1,0,
    0,0,1,1,1,1,0,
    0,0,0,0,0,0,0
};

template<int DIM>Mat2F32 _caractere<DIM>::Lettera =Mat2F32(Vec2I32(9,7),_a_);


template<int DIM>F32 _caractere<DIM>::_b_[]={
    0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,
    0,1,0,0,0,0,0,
    0,1,0,0,0,0,0,
    0,1,1,1,1,0,0,
    0,1,0,0,0,1,0,
    0,1,0,0,0,1,0,
    0,1,1,1,1,0,0,
    0,0,0,0,0,0,0
};

template<int DIM>Mat2F32 _caractere<DIM>::Letterb =Mat2F32(Vec2I32(9,7),_b_);



template<int DIM>F32 _caractere<DIM>::_c_[]={
    0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,
    0,0,1,1,1,1,0,
    0,1,0,0,0,0,0,
    0,1,0,0,0,0,0,
    0,1,0,0,0,0,0,
    0,0,1,1,1,1,0,
    0,0,0,0,0,0,0
};

template<int DIM>Mat2F32 _caractere<DIM>::Letterc =Mat2F32(Vec2I32(9,7),_c_);

template<int DIM>F32 _caractere<DIM>::_d_[]={
    0,0,0,0,0,0,0,
    0,0,0,0,0,1,0,
    0,0,0,0,0,1,0,
    0,0,0,0,0,1,0,
    0,0,1,1,1,1,0,
    0,1,0,0,0,1,0,
    0,1,0,0,0,1,0,
    0,0,1,1,1,1,0,
    0,0,0,0,0,0,0
};

template<int DIM>Mat2F32 _caractere<DIM>::Letterd =Mat2F32(Vec2I32(9,7),_d_);


template<int DIM>F32 _caractere<DIM>::_e_[]={
    0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,
    0,0,1,1,1,0,0,
    0,1,0,0,0,1,0,
    0,1,1,1,1,0,0,
    0,1,0,0,0,0,0,
    0,0,1,1,1,1,0,
    0,0,0,0,0,0,0
};

template<int DIM>Mat2F32 _caractere<DIM>::Lettere =Mat2F32(Vec2I32(9,7),_e_);


template<int DIM>F32 _caractere<DIM>::_f_[]={
    0,0,0,0,0,0,0,
    0,0,0,0,1,1,0,
    0,0,0,1,0,0,0,
    0,1,1,1,1,1,0,
    0,0,0,1,0,0,0,
    0,0,0,1,0,0,0,
    0,0,0,1,0,0,0,
    0,0,0,1,0,0,0,
    0,0,0,0,0,0,0
};

template<int DIM>Mat2F32 _caractere<DIM>::Letterf =Mat2F32(Vec2I32(9,7),_f_);


template<int DIM>F32 _caractere<DIM>::_g_[]={
    0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,
    0,0,1,1,1,1,0,
    0,1,0,0,0,1,0,
    0,0,1,1,1,1,0,
    0,0,0,0,0,1,0,
    0,0,1,1,1,0,0,
    0,0,0,0,0,0,0
};

template<int DIM>Mat2F32 _caractere<DIM>::Letterg =Mat2F32(Vec2I32(9,7),_g_);


template<int DIM>F32 _caractere<DIM>::_h_[]={
    0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,
    0,1,0,0,0,0,0,
    0,1,1,1,1,0,0,
    0,1,0,0,0,1,0,
    0,1,0,0,0,1,0,
    0,1,0,0,0,1,0,
    0,1,0,0,0,1,0,
    0,0,0,0,0,0,0
};

template<int DIM>Mat2F32 _caractere<DIM>::Letterh =Mat2F32(Vec2I32(9,7),_h_);

template<int DIM>F32 _caractere<DIM>::_i_[]={
    0,0,0,0,0,0,0,
    0,0,0,1,0,0,0,
    0,0,0,0,0,0,0,
    0,0,1,1,0,0,0,
    0,0,0,1,0,0,0,
    0,0,0,1,0,0,0,
    0,0,0,1,0,0,0,
    0,1,1,1,1,1,0,
    0,0,0,0,0,0,0
};

template<int DIM>Mat2F32 _caractere<DIM>::Letteri =Mat2F32(Vec2I32(9,7),_i_);

template<int DIM>F32 _caractere<DIM>::_j_[]={
    0,0,0,0,0,0,0,
    0,0,0,0,1,0,0,
    0,0,0,0,0,0,0,
    0,0,0,1,1,0,0,
    0,0,0,0,1,0,0,
    0,0,0,0,1,0,0,
    0,0,0,0,1,0,0,
    0,0,1,1,0,0,0,
    0,0,0,0,0,0,0
};

template<int DIM>Mat2F32 _caractere<DIM>::Letterj =Mat2F32(Vec2I32(9,7),_j_);



template<int DIM>F32 _caractere<DIM>::_k_[]={
    0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,
    0,0,1,0,0,0,0,
    0,0,1,0,1,0,0,
    0,0,1,1,0,0,0,
    0,0,1,0,0,0,0,
    0,0,1,1,0,0,0,
    0,0,1,0,1,0,0,
    0,0,0,0,0,0,0
};

template<int DIM>Mat2F32 _caractere<DIM>::Letterk =Mat2F32(Vec2I32(9,7),_k_);


template<int DIM>F32 _caractere<DIM>::_l_[]={
    0,0,0,0,0,0,0,
    0,0,1,1,0,0,0,
    0,0,0,1,0,0,0,
    0,0,0,1,0,0,0,
    0,0,0,1,0,0,0,
    0,0,0,1,0,0,0,
    0,0,0,1,0,0,0,
    0,0,0,0,1,0,0,
    0,0,0,0,0,0,0
};

template<int DIM>Mat2F32 _caractere<DIM>::Letterl =Mat2F32(Vec2I32(9,7),_l_);

template<int DIM>F32 _caractere<DIM>::_m_[]={
    0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,
    0,0,1,0,1,0,0,
    0,1,0,1,0,1,0,
    0,1,0,1,0,1,0,
    0,1,0,1,0,1,0,
    0,1,0,1,0,1,0,
    0,0,0,0,0,0,0
};

template<int DIM>Mat2F32 _caractere<DIM>::Letterm =Mat2F32(Vec2I32(9,7),_m_);

template<int DIM>F32 _caractere<DIM>::_n_[]={
    0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,
    0,0,1,1,1,0,0,
    0,0,1,0,0,1,0,
    0,0,1,0,0,1,0,
    0,0,1,0,0,1,0,
    0,0,1,0,0,1,0,
    0,0,0,0,0,0,0
};

template<int DIM>Mat2F32 _caractere<DIM>::Lettern =Mat2F32(Vec2I32(9,7),_n_);

template<int DIM>F32 _caractere<DIM>::_o_[]={
    0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,
    0,0,1,1,1,0,0,
    0,1,0,0,0,1,0,
    0,1,0,0,0,1,0,
    0,1,0,0,0,1,0,
    0,0,1,1,1,0,0,
    0,0,0,0,0,0,0
};

template<int DIM>Mat2F32 _caractere<DIM>::Lettero =Mat2F32(Vec2I32(9,7),_o_);

template<int DIM>F32 _caractere<DIM>::_p_[]={
    0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,
    0,0,1,1,1,0,0,
    0,0,1,0,0,1,0,
    0,0,1,1,1,0,0,
    0,0,1,0,0,0,0,
    0,0,1,0,0,0,0,
    0,0,0,0,0,0,0
};

template<int DIM>Mat2F32 _caractere<DIM>::Letterp =Mat2F32(Vec2I32(9,7),_p_);


template<int DIM>F32 _caractere<DIM>::_q_[]={
    0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,
    0,0,1,1,1,0,0,
    0,1,0,0,1,0,0,
    0,0,1,1,1,0,0,
    0,0,0,0,1,0,0,
    0,0,0,0,1,0,0,
    0,0,0,0,0,0,0
};

template<int DIM>Mat2F32 _caractere<DIM>::Letterq =Mat2F32(Vec2I32(9,7),_q_);


template<int DIM>F32 _caractere<DIM>::_r_[]={
    0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,
    0,0,1,1,1,0,0,
    0,0,1,0,0,0,0,
    0,0,1,0,0,0,0,
    0,0,1,0,0,0,0,
    0,0,1,0,0,0,0,
    0,0,0,0,0,0,0
};

template<int DIM>Mat2F32 _caractere<DIM>::Letterr =Mat2F32(Vec2I32(9,7),_r_);


template<int DIM>F32 _caractere<DIM>::_s_[]={
    0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,
    0,0,1,1,1,1,0,
    0,1,0,0,0,0,0,
    0,0,1,1,1,0,0,
    0,0,0,0,0,1,0,
    0,1,1,1,1,0,0,
    0,0,0,0,0,0,0
};

template<int DIM>Mat2F32 _caractere<DIM>::Letters =Mat2F32(Vec2I32(9,7),_s_);


template<int DIM>F32 _caractere<DIM>::_t_[]={
    0,0,0,0,0,0,0,
    0,0,0,1,0,0,0,
    0,0,0,1,0,0,0,
    0,0,1,1,1,0,0,
    0,0,0,1,0,0,0,
    0,0,0,1,0,0,0,
    0,0,0,1,0,0,0,
    0,0,0,0,1,1,0,
    0,0,0,0,0,0,0
};

template<int DIM>Mat2F32 _caractere<DIM>::Lettert =Mat2F32(Vec2I32(9,7),_t_);


template<int DIM>F32 _caractere<DIM>::_u_[]={
    0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,
    0,1,0,0,0,1,0,
    0,1,0,0,0,1,0,
    0,1,0,0,0,1,0,
    0,1,0,0,0,1,0,
    0,0,1,1,1,0,0,
    0,0,0,0,0,0,0
};

template<int DIM>Mat2F32 _caractere<DIM>::Letteru =Mat2F32(Vec2I32(9,7),_u_);



template<int DIM>F32 _caractere<DIM>::_v_[]={
    0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,
    0,1,0,0,0,1,0,
    0,0,1,0,1,0,0,
    0,0,1,0,1,0,0,
    0,0,0,1,0,0,0,
    0,0,0,1,0,0,0,
    0,0,0,0,0,0,0
};

template<int DIM>Mat2F32 _caractere<DIM>::Letterv =Mat2F32(Vec2I32(9,7),_v_);

template<int DIM>F32 _caractere<DIM>::_w_[]={
    0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,
    0,1,0,0,0,1,0,
    0,1,0,0,0,1,0,
    0,1,0,1,0,1,0,
    0,0,1,0,1,0,0,
    0,0,1,0,1,0,0,
    0,0,0,0,0,0,0
};

template<int DIM>Mat2F32 _caractere<DIM>::Letterw =Mat2F32(Vec2I32(9,7),_w_);

template<int DIM>F32 _caractere<DIM>::_x_[]={
    0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,
    0,1,0,0,0,1,0,
    0,0,1,0,1,0,0,
    0,0,0,1,0,0,0,
    0,0,1,0,1,0,0,
    0,1,0,0,0,1,0,
    0,0,0,0,0,0,0
};

template<int DIM>Mat2F32 _caractere<DIM>::Letterx =Mat2F32(Vec2I32(9,7),_x_);

template<int DIM>F32 _caractere<DIM>::_y_[]={
    0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,
    0,1,0,0,0,1,0,
    0,0,1,0,1,0,0,
    0,0,0,1,0,0,0,
    0,0,1,0,0,0,0,
    0,1,1,0,0,0,0,
    0,0,0,0,0,0,0
};

template<int DIM>Mat2F32 _caractere<DIM>::Lettery =Mat2F32(Vec2I32(9,7),_y_);

template<int DIM>F32 _caractere<DIM>::_z_[]={
    0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,
    0,1,1,1,1,1,0,
    0,0,0,0,1,0,0,
    0,0,0,1,0,0,0,
    0,0,1,0,0,0,0,
    0,1,1,1,1,1,0,
    0,0,0,0,0,0,0
};

template<int DIM>Mat2F32 _caractere<DIM>::Letterz =Mat2F32(Vec2I32(9,7),_z_);




template<int DIM>F32 _caractere<DIM>::_A_[]={
    0,0,0,0,0,0,0,
    0,1,1,1,1,1,0,
    0,1,0,0,0,1,0,
    0,1,0,0,0,1,0,
    0,1,1,1,1,1,0,
    0,1,0,0,0,1,0,
    0,1,0,0,0,1,0,
    0,1,0,0,0,1,0,
    0,0,0,0,0,0,0
};

template<int DIM>Mat2F32 _caractere<DIM>::LetterA =Mat2F32(Vec2I32(9,7),_A_);


template<int DIM>F32 _caractere<DIM>::_B_[]={
    0,0,0,0,0,0,0,
    0,1,1,1,1,0,0,
    0,1,0,0,0,1,0,
    0,1,0,0,0,1,0,
    0,1,1,1,1,1,0,
    0,1,0,0,0,1,0,
    0,1,0,0,0,1,0,
    0,1,1,1,1,0,0,
    0,0,0,0,0,0,0
};

template<int DIM>Mat2F32 _caractere<DIM>::LetterB =Mat2F32(Vec2I32(9,7),_B_);



template<int DIM>F32 _caractere<DIM>::_C_[]={
    0,0,0,0,0,0,0,
    0,1,1,1,1,1,0,
    0,1,0,0,0,0,0,
    0,1,0,0,0,0,0,
    0,1,0,0,0,0,0,
    0,1,0,0,0,0,0,
    0,1,0,0,0,0,0,
    0,1,1,1,1,1,0,
    0,0,0,0,0,0,0
};

template<int DIM>Mat2F32 _caractere<DIM>::LetterC =Mat2F32(Vec2I32(9,7),_C_);

template<int DIM>F32 _caractere<DIM>::_D_[]={
    0,0,0,0,0,0,0,
    0,1,1,1,1,0,0,
    0,1,0,0,0,1,0,
    0,1,0,0,0,1,0,
    0,1,0,0,0,1,0,
    0,1,0,0,0,1,0,
    0,1,0,0,0,1,0,
    0,1,1,1,1,0,0,
    0,0,0,0,0,0,0
};

template<int DIM>Mat2F32 _caractere<DIM>::LetterD =Mat2F32(Vec2I32(9,7),_D_);


template<int DIM>F32 _caractere<DIM>::_E_[]={
    0,0,0,0,0,0,0,
    0,1,1,1,1,1,0,
    0,1,0,0,0,0,0,
    0,1,0,0,0,0,0,
    0,1,1,1,1,1,0,
    0,1,0,0,0,0,0,
    0,1,0,0,0,0,0,
    0,1,1,1,1,1,0,
    0,0,0,0,0,0,0
};

template<int DIM>Mat2F32 _caractere<DIM>::LetterE =Mat2F32(Vec2I32(9,7),_E_);


template<int DIM>F32 _caractere<DIM>::_F_[]={
    0,0,0,0,0,0,0,
    0,1,1,1,1,1,0,
    0,1,0,0,0,0,0,
    0,1,0,0,0,0,0,
    0,1,1,1,1,1,0,
    0,1,0,0,0,0,0,
    0,1,0,0,0,0,0,
    0,1,0,0,0,0,0,
    0,0,0,0,0,0,0
};

template<int DIM>Mat2F32 _caractere<DIM>::LetterF =Mat2F32(Vec2I32(9,7),_F_);


template<int DIM>F32 _caractere<DIM>::_G_[]={
    0,0,0,0,0,0,0,
    0,1,1,1,1,1,0,
    0,1,0,0,0,0,0,
    0,1,0,0,0,0,0,
    0,1,0,0,1,1,0,
    0,1,0,0,0,1,0,
    0,1,0,0,0,1,0,
    0,1,1,1,1,1,0,
    0,0,0,0,0,0,0
};

template<int DIM>Mat2F32 _caractere<DIM>::LetterG =Mat2F32(Vec2I32(9,7),_G_);


template<int DIM>F32 _caractere<DIM>::_H_[]={
    0,0,0,0,0,0,0,
    0,1,0,0,0,1,0,
    0,1,0,0,0,1,0,
    0,1,0,0,0,1,0,
    0,1,1,1,1,1,0,
    0,1,0,0,0,1,0,
    0,1,0,0,0,1,0,
    0,1,0,0,0,1,0,
    0,0,0,0,0,0,0
};

template<int DIM>Mat2F32 _caractere<DIM>::LetterH =Mat2F32(Vec2I32(9,7),_H_);

template<int DIM>F32 _caractere<DIM>::_I_[]={
    0,0,0,0,0,0,0,
    0,1,1,1,1,1,0,
    0,0,0,1,0,0,0,
    0,0,0,1,0,0,0,
    0,0,0,1,0,0,0,
    0,0,0,1,0,0,0,
    0,0,0,1,0,0,0,
    0,1,1,1,1,1,0,
    0,0,0,0,0,0,0
};

template<int DIM>Mat2F32 _caractere<DIM>::LetterI =Mat2F32(Vec2I32(9,7),_I_);

template<int DIM>F32 _caractere<DIM>::_J_[]={
    0,0,0,0,0,0,0,
    0,0,0,0,0,1,0,
    0,0,0,0,0,1,0,
    0,0,0,0,0,1,0,
    0,0,0,0,0,1,0,
    0,0,0,0,0,1,0,
    0,0,0,0,0,1,0,
    0,1,1,1,1,0,0,
    0,0,0,0,0,0,0
};

template<int DIM>Mat2F32 _caractere<DIM>::LetterJ =Mat2F32(Vec2I32(9,7),_J_);



template<int DIM>F32 _caractere<DIM>::_K_[]={
    0,0,0,0,0,0,0,
    0,1,0,0,0,1,0,
    0,1,0,0,1,0,0,
    0,1,0,1,0,0,0,
    0,1,1,0,0,0,0,
    0,1,0,1,0,0,0,
    0,1,0,0,1,0,0,
    0,1,0,0,0,1,0,
    0,0,0,0,0,0,0
};

template<int DIM>Mat2F32 _caractere<DIM>::LetterK =Mat2F32(Vec2I32(9,7),_K_);


template<int DIM>F32 _caractere<DIM>::_L_[]={
    0,0,0,0,0,0,0,
    0,1,0,0,0,0,0,
    0,1,0,0,0,0,0,
    0,1,0,0,0,0,0,
    0,1,0,0,0,0,0,
    0,1,0,0,0,0,0,
    0,1,0,0,0,0,0,
    0,1,1,1,1,1,0,
    0,0,0,0,0,0,0
};

template<int DIM>Mat2F32 _caractere<DIM>::LetterL =Mat2F32(Vec2I32(9,7),_L_);

template<int DIM>F32 _caractere<DIM>::_M_[]={
    0,0,0,0,0,0,0,
    0,1,0,0,0,1,0,
    0,1,1,0,1,1,0,
    0,1,0,1,0,1,0,
    0,1,0,1,0,1,0,
    0,1,0,0,0,1,0,
    0,1,0,0,0,1,0,
    0,1,0,0,0,1,0,
    0,0,0,0,0,0,0
};

template<int DIM>Mat2F32 _caractere<DIM>::LetterM =Mat2F32(Vec2I32(9,7),_M_);

template<int DIM>F32 _caractere<DIM>::_N_[]={
    0,0,0,0,0,0,0,
    0,1,0,0,0,1,0,
    0,1,1,0,0,1,0,
    0,1,0,1,0,1,0,
    0,1,0,1,0,1,0,
    0,1,0,1,0,1,0,
    0,1,0,0,1,1,0,
    0,1,0,0,0,1,0,
    0,0,0,0,0,0,0
};

template<int DIM>Mat2F32 _caractere<DIM>::LetterN =Mat2F32(Vec2I32(9,7),_N_);

template<int DIM>F32 _caractere<DIM>::_O_[]={
    0,0,0,0,0,0,0,
    0,0,1,1,1,0,0,
    0,1,0,0,0,1,0,
    0,1,0,0,0,1,0,
    0,1,0,0,0,1,0,
    0,1,0,0,0,1,0,
    0,1,0,0,0,1,0,
    0,0,1,1,1,0,0,
    0,0,0,0,0,0,0
};

template<int DIM>Mat2F32 _caractere<DIM>::LetterO =Mat2F32(Vec2I32(9,7),_O_);

template<int DIM>F32 _caractere<DIM>::_P_[]={
    0,0,0,0,0,0,0,
    0,1,1,1,1,0,0,
    0,1,0,0,0,1,0,
    0,1,0,0,0,1,0,
    0,1,1,1,1,0,0,
    0,1,0,0,0,0,0,
    0,1,0,0,0,0,0,
    0,1,0,0,0,0,0,
    0,0,0,0,0,0,0
};

template<int DIM>Mat2F32 _caractere<DIM>::LetterP =Mat2F32(Vec2I32(9,7),_P_);


template<int DIM>F32 _caractere<DIM>::_Q_[]={
    0,0,0,0,0,0,0,
    0,0,1,1,1,0,0,
    0,1,0,0,0,1,0,
    0,1,0,0,0,1,0,
    0,1,0,0,0,1,0,
    0,1,0,0,0,1,0,
    0,0,1,1,1,0,0,
    0,0,0,0,0,1,0,
    0,0,0,0,0,0,0
};

template<int DIM>Mat2F32 _caractere<DIM>::LetterQ =Mat2F32(Vec2I32(9,7),_Q_);


template<int DIM>F32 _caractere<DIM>::_R_[]={
    0,0,0,0,0,0,0,
    0,1,1,1,1,0,0,
    0,1,0,0,0,1,0,
    0,1,0,0,0,1,0,
    0,1,1,1,1,0,0,
    0,1,0,1,0,0,0,
    0,1,0,0,1,0,0,
    0,1,0,0,0,1,0,
    0,0,0,0,0,0,0
};

template<int DIM>Mat2F32 _caractere<DIM>::LetterR =Mat2F32(Vec2I32(9,7),_R_);


template<int DIM>F32 _caractere<DIM>::_S_[]={
    0,0,0,0,0,0,0,
    0,0,1,1,1,1,0,
    0,1,0,0,0,0,0,
    0,1,0,0,0,0,0,
    0,0,1,1,1,0,0,
    0,0,0,0,0,1,0,
    0,0,0,0,0,1,0,
    0,1,1,1,1,0,0,
    0,0,0,0,0,0,0
};

template<int DIM>Mat2F32 _caractere<DIM>::LetterS =Mat2F32(Vec2I32(9,7),_S_);


template<int DIM>F32 _caractere<DIM>::_T_[]={
    0,0,0,0,0,0,0,
    0,1,1,1,1,1,0,
    0,0,0,1,0,0,0,
    0,0,0,1,0,0,0,
    0,0,0,1,0,0,0,
    0,0,0,1,0,0,0,
    0,0,0,1,0,0,0,
    0,0,0,1,0,0,0,
    0,0,0,0,0,0,0
};

template<int DIM>Mat2F32 _caractere<DIM>::LetterT =Mat2F32(Vec2I32(9,7),_T_);


template<int DIM>F32 _caractere<DIM>::_U_[]={
    0,0,0,0,0,0,0,
    0,1,0,0,0,1,0,
    0,1,0,0,0,1,0,
    0,1,0,0,0,1,0,
    0,1,0,0,0,1,0,
    0,1,0,0,0,1,0,
    0,1,0,0,0,1,0,
    0,0,1,1,1,0,0,
    0,0,0,0,0,0,0
};

template<int DIM>Mat2F32 _caractere<DIM>::LetterU =Mat2F32(Vec2I32(9,7),_U_);



template<int DIM>F32 _caractere<DIM>::_V_[]={
    0,0,0,0,0,0,0,
    0,1,0,0,0,1,0,
    0,1,0,0,0,1,0,
    0,1,0,0,0,1,0,
    0,0,1,0,1,0,0,
    0,0,1,0,1,0,0,
    0,0,0,1,0,0,0,
    0,0,0,1,0,0,0,
    0,0,0,0,0,0,0
};

template<int DIM>Mat2F32 _caractere<DIM>::LetterV =Mat2F32(Vec2I32(9,7),_V_);

template<int DIM>F32 _caractere<DIM>::_W_[]={
    0,0,0,0,0,0,0,
    0,1,0,0,0,1,0,
    0,1,0,0,0,1,0,
    0,1,0,0,0,1,0,
    0,1,0,1,0,1,0,
    0,1,0,1,0,1,0,
    0,0,1,0,1,0,0,
    0,0,1,0,1,0,0,
    0,0,0,0,0,0,0
};

template<int DIM>Mat2F32 _caractere<DIM>::LetterW =Mat2F32(Vec2I32(9,7),_W_);

template<int DIM>F32 _caractere<DIM>::_X_[]={
    0,0,0,0,0,0,0,
    0,1,0,0,0,1,0,
    0,0,1,0,1,0,0,
    0,0,1,0,1,0,0,
    0,0,0,1,0,0,0,
    0,0,1,0,1,0,0,
    0,0,1,0,1,0,0,
    0,1,0,0,0,1,0,
    0,0,0,0,0,0,0
};

template<int DIM>Mat2F32 _caractere<DIM>::LetterX =Mat2F32(Vec2I32(9,7),_X_);

template<int DIM>F32 _caractere<DIM>::_Y_[]={
    0,0,0,0,0,0,0,
    0,1,0,0,0,1,0,
    0,1,0,0,0,1,0,
    0,0,1,0,1,0,0,
    0,0,0,1,0,0,0,
    0,0,0,1,0,0,0,
    0,0,0,1,0,0,0,
    0,0,0,1,0,0,0,
    0,0,0,0,0,0,0
};

template<int DIM>Mat2F32 _caractere<DIM>::LetterY =Mat2F32(Vec2I32(9,7),_Y_);

template<int DIM>F32 _caractere<DIM>::_Z_[]={
    0,0,0,0,0,0,0,
    0,1,1,1,1,1,0,
    0,0,0,0,1,0,0,
    0,0,0,0,1,0,0,
    0,0,0,1,0,0,0,
    0,0,1,0,0,0,0,
    0,0,1,0,0,0,0,
    0,1,1,1,1,1,0,
    0,0,0,0,0,0,0
};

template<int DIM>Mat2F32 _caractere<DIM>::LetterZ =Mat2F32(Vec2I32(9,7),_Z_);



template<int DIM>F32 _caractere<DIM>::_0_[]={
    0,0,0,0,0,0,
    0,0,1,1,0,0,
    0,1,0,0,1,0,
    0,1,0,0,1,0,
    0,1,0,0,1,0,
    0,1,0,0,1,0,
    0,1,0,0,1,0,
    0,0,1,1,0,0,
    0,0,0,0,0,0
};

template<int DIM>Mat2F32 _caractere<DIM>::Letter0 =Mat2F32(Vec2I32(9,6),_0_);

template<int DIM>F32 _caractere<DIM>::_1_[]={
    0,0,0,0,0,
    0,0,0,1,0,
    0,0,1,1,0,
    0,1,0,1,0,
    0,0,0,1,0,
    0,0,0,1,0,
    0,0,0,1,0,
    0,0,0,1,0,
    0,0,0,0,0
};

template<int DIM>Mat2F32 _caractere<DIM>::Letter1 =Mat2F32(Vec2I32(9,5),_1_);

template<int DIM>F32 _caractere<DIM>::_2_[]={
    0,0,0,0,0,0,
    0,0,1,1,0,0,
    0,1,0,0,1,0,
    0,0,0,0,1,0,
    0,0,0,1,0,0,
    0,0,1,0,0,0,
    0,1,0,0,0,0,
    0,1,1,1,1,0,
    0,0,0,0,0,0
};

template<int DIM>Mat2F32 _caractere<DIM>::Letter2 =Mat2F32(Vec2I32(9,6),_2_);

template<int DIM>F32 _caractere<DIM>::_3_[]={
    0,0,0,0,0,0,
    0,1,1,1,0,0,
    0,0,0,0,1,0,
    0,0,0,0,1,0,
    0,0,1,1,0,0,
    0,0,0,0,1,0,
    0,0,0,0,1,0,
    0,1,1,1,0,0,
    0,0,0,0,0,0
};

template<int DIM>Mat2F32 _caractere<DIM>::Letter3 =Mat2F32(Vec2I32(9,6),_3_);

template<int DIM>F32 _caractere<DIM>::_4_[]={
    0,0,0,0,0,0,
    0,0,0,1,0,0,
    0,0,1,1,0,0,
    0,0,1,1,0,0,
    0,1,0,1,0,0,
    0,1,0,1,0,0,
    0,1,1,1,1,0,
    0,0,0,1,0,0,
    0,0,0,0,0,0
};

template<int DIM>Mat2F32 _caractere<DIM>::Letter4 =Mat2F32(Vec2I32(9,6),_4_);

template<int DIM>F32 _caractere<DIM>::_5_[]={
    0,0,0,0,0,0,
    0,1,1,1,1,0,
    0,1,0,0,0,0,
    0,1,0,0,0,0,
    0,1,1,1,0,0,
    0,0,0,0,1,0,
    0,0,0,0,1,0,
    0,1,1,1,0,0,
    0,0,0,0,0,0
};

template<int DIM>Mat2F32 _caractere<DIM>::Letter5 =Mat2F32(Vec2I32(9,6),_5_);

template<int DIM>F32 _caractere<DIM>::_6_[]={
    0,0,0,0,0,0,
    0,0,1,1,0,0,
    0,1,0,0,0,0,
    0,1,0,0,0,0,
    0,1,1,1,0,0,
    0,1,0,0,1,0,
    0,1,0,0,1,0,
    0,0,1,1,0,0,
    0,0,0,0,0,0
};

template<int DIM>Mat2F32 _caractere<DIM>::Letter6 =Mat2F32(Vec2I32(9,6),_6_);

template<int DIM>F32 _caractere<DIM>::_7_[]={
    0,0,0,0,0,0,
    0,1,1,1,1,0,
    0,0,0,0,1,0,
    0,0,0,1,0,0,
    0,0,0,1,0,0,
    0,0,1,0,0,0,
    0,0,1,0,0,0,
    0,1,0,0,0,0,
    0,0,0,0,0,0
};

template<int DIM>Mat2F32 _caractere<DIM>::Letter7 =Mat2F32(Vec2I32(9,6),_7_);

template<int DIM>F32 _caractere<DIM>::_8_[]={
    0,0,0,0,0,0,
    0,0,1,1,0,0,
    0,1,0,0,1,0,
    0,1,0,0,1,0,
    0,0,1,1,0,0,
    0,1,0,0,1,0,
    0,1,0,0,1,0,
    0,0,1,1,0,0,
    0,0,0,0,0,0
};

template<int DIM>Mat2F32 _caractere<DIM>::Letter8 =Mat2F32(Vec2I32(9,6),_8_);

template<int DIM>F32 _caractere<DIM>::_9_[]={
    0,0,0,0,0,0,
    0,0,1,1,0,0,
    0,1,0,0,1,0,
    0,1,0,0,1,0,
    0,0,1,1,1,0,
    0,0,0,0,1,0,
    0,0,0,0,1,0,
    0,0,1,1,0,0,
    0,0,0,0,0,0
};

template<int DIM>Mat2F32 _caractere<DIM>::Letter9 =pop::Mat2F32(pop::Vec2I32(9,6),_9_);

}


//    inline int bin(int N){
//        if(N>1)
//            return N%10+2*bin(N/10);
//        else
//            return N;
//    }
//    template<int DIM,typename PixelType>
//    static void number(MatN<DIM,PixelType> &img,F32 x1,F32 y1, int value,typename MatN<DIM,PixelType>::F value,int width=3)
//    {

//        int valuepossitive = absolute(value);
//        std::vector<int> v;
//        v.push_back(bin(1110111));//represent character 0
//        v.push_back(bin(100100));//represent character 1
//        v.push_back(bin(1011101));//represent character 2
//        v.push_back(bin(1101101));//represent character 3
//        v.push_back(bin(101110));//represent character 4
//        v.push_back(bin(1101011));//represent character 5
//        v.push_back(bin(1111011));//represent character 6
//        v.push_back(bin(100101));//represent character 7
//        v.push_back(bin(1111111));//represent character 8
//        v.push_back(bin(1101111));//represent character 9
//        int translation=0;

//        while(valuepossitive!=-1){
//            int i = valuepossitive%10;
//            if(valuepossitive>9)
//                valuepossitive/=10;
//            else
//                valuepossitive = -1;
//            int vv = v[i];
//            //top line horizontal
//            if(vv%2==1){
//                for(int x=x1;x<=x1+width;x++)
//                {
//                    if(img.isValid(translation+x,y1))
//                        img(translation+x,y1)=value;
//                }
//            }
//            vv/=2;
//            if(vv%2==1){
//                //top left vertial
//                for(int y=y1;y<=y1+width;y++)
//                {
//                    if(img.isValid(translation+x1,y))
//                        img(translation+x1,y)=value;
//                }
//            }
//            vv/=2;
//            if(vv%2==1){
//                //top right vertial
//                for(int y=y1;y<=y1+width;y++)
//                {
//                    if(img.isValid(translation+x1+width,y))
//                        img(translation+x1+width,y)=value;
//                }
//            }
//            vv/=2;
//            if(vv%2==1){
//                //middel line horizontal
//                for(int x=x1;x<=x1+width;x++)
//                {
//                    if(img.isValid(translation+x,y1+width))
//                        img(translation+x,y1+width)=value;
//                }
//            }
//            vv/=2;
//            if(vv%2==1){
//                //buttom left vertial
//                for(int y=y1+width;y<=y1+2*width;y++)
//                {
//                    if(img.isValid(translation+x1,y))
//                        img(translation+x1,y)=value;
//                }
//            }
//            vv/=2;
//            if(vv%2==1){
//                //buttom right vertial
//                for(int y=y1+width;y<=y1+2*width;y++)
//                {
//                    if(img.isValid(translation+x1+width,y))
//                        img(translation+x1+width,y)=value;
//                }
//            }
//            vv/=2;
//            if(vv%2==1){
//                //buttom line horizontal
//                for(int x=x1;x<=x1+width;x++)
//                {
//                    if(img.isValid(translation+x,y1+2*width))
//                        img(translation+x,y1+2*width)=value;
//                }
//            }
//            translation-=(width+2);
//        }
//        if(value<0){
//            for(int x=x1;x<=x1+width;x++)
//            {
//                if(img.isValid(translation+x,y1+width))
//                    img(translation+x,y1+width)=value;
//            }
//        }


//    }
#endif // DRAW_H
