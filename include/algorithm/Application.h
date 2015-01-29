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
#ifndef APPLICATION_H
#define APPLICATION_H
#include"data/typeF/RGB.h"
#include"data/mat/MatNDisplay.h"
#include"algorithm/GeometricalTransformation.h"
#include"algorithm/Processing.h"

namespace pop
{
/*!
\defgroup Application Application
\ingroup Algorithm
\brief Matrix In -> information
*/

class POP_EXPORTS Application
{

    /*!
        \class pop::Application
        \ingroup Application
        \author Tariel Vincent
    */
public:
    /*!
     * \param in input label matrix
     * \param automaticresize automatic resize the matrix to fit your screen
     * \return a pair containing low and hight value selected
     *
     * Use the arrows key to change the theshold value ARROWUP=low_value++, ARROWDOWN=low_value--, ARROWRIGHT=high_value++, ARROWLEFT=high_value--
     *
     * For 3d matrix, use the key V=slice++  and C=slice--
     *
    */
    template<int DIM,typename PixelType>
    static std::pair<int,int> thresholdSelection(const MatN<DIM,PixelType> & in,bool automaticresize=true )
    {
        int low_value, high_value;
        MatN<DIM,PixelType>  img(in);
        if(automaticresize ==true){
            VecN<MatN<DIM,PixelType>::DIM,pop::F32> scale(600./img.getDomain()(0));
            img= GeometricalTransformation::scale(img,scale);
        }
        high_value=255;
        low_value=125;


        MatNDisplay window;


        int index=0;

        do{
            if(window.is_keyARROWUP()){
                low_value++;
            }
            else if(window.is_keyARROWDOWN()){
                low_value--;
            }
            else if(window.is_keyARROWRIGHT()){
                high_value++;
            }
            else if(window.is_keyARROWLEFT())
                high_value--;
            else if(window.is_keyV()){
                index+=1;
            }
            else if(window.is_keyC())
                index-=1;
            if(DIM==3){
                if(index<0)
                    index=0;
                if(index>=img.getDomain()(2))
                    index = img.getDomain()(2)-1;
                MatN<DIM-1,PixelType> plane;
                typename MatN<DIM-1,PixelType>::IteratorEDomain it_plane(plane.getIteratorEDomain());
                while(it_plane.next()){
                    plane(it_plane.x()) = img(it_plane.x().addCoordinate(2,index));
                }


                MatN<DIM-1,UI8> threshold= pop::Processing::threshold(plane,low_value,high_value);
                MatN<DIM-1,RGBUI8> visu = pop::Visualization::labelForeground(threshold,plane);
                std::string title = "minValue="+BasicUtility::Any2String(low_value)+ " maxValue="+BasicUtility::Any2String(high_value)+" Use arrows to slide the threshold values and the Key C and V to move in z-axis";
                window.set_title(title.c_str());
                window.display(visu);
            }
            else{
                MatN<DIM,UI8>  threshold= pop::Processing::threshold(img,low_value,high_value);
                MatN<DIM,RGBUI8>  visu = pop::Visualization::labelForeground(threshold,img);
                std::string title = "minValue="+BasicUtility::Any2String(low_value)+ " maxValue="+BasicUtility::Any2String(high_value)+" Use arrows to slide the threshold values";
                window.set_title(title.c_str());
                window.display(visu);
            }
            if(window.is_key()==true){
                window.waitTime (20);
            }
        }while(!window.is_closed());
        return std::make_pair(low_value, high_value);
    }

    /*!
     * \param in input label matrix
     * \param lowvalue low threshold value
     * \param highvalue high threshold value
     *
     *
    */
    static inline void thresholdSelectionColor(const Mat2RGBUI8& in,RGBUI8 &lowvalue, RGBUI8 &highvalue ){

        Mat2RGBUI8  img(in);

        VecN<2,pop::F32> scale(600.f/img.getDomain()(0));
        img= GeometricalTransformation::scale(img,scale);

        lowvalue =RGBUI8(125,125,125);
        highvalue=RGBUI8(255,255,255);

        MatNDisplay window;

        std::string title = "minValue="+BasicUtility::Any2String(lowvalue)+ " maxValue="+BasicUtility::Any2String(highvalue)+" keys XSDE for red-channel, keys VFGT for green-channel, keys NHJU for blue-channel";
        window.set_title(title.c_str());

        bool click=false;
        RGBUI8 value;
        do{
            if(window.is_keyS()){
                if(lowvalue.r()>0)
                    lowvalue.r()--;
            }
            else if(window.is_keyD()){
                if(lowvalue.r()<255)
                    lowvalue.r()++;
            }
            else if(window.is_keyX()){
                if(highvalue.r()>0)
                    highvalue.r()--;
            }
            else if(window.is_keyE()){
                if(highvalue.r()<255)
                    highvalue.r()++;
            }
            if(window.is_keyF()){
                if(lowvalue.g()>0)
                    lowvalue.g()--;
            }
            else if(window.is_keyG()){
                if(lowvalue.g()<255)
                    lowvalue.g()++;
            }
            else if(window.is_keyV()){
                if(highvalue.g()>0)
                    highvalue.g()--;
            }
            else if(window.is_keyT()){
                if(highvalue.g()<255)
                    highvalue.g()++;
            }
            else if(window.is_keyH()){
                if(lowvalue.b()>0)
                    lowvalue.b()--;
            }
            else if(window.is_keyJ()){
                if(lowvalue.b()<255)
                    lowvalue.b()++;
            }
            else if(window.is_keyN()){
                if(highvalue.b()>0)
                    highvalue.b()--;
            }
            else if(window.is_keyU()){
                if(highvalue.b()<255)
                    highvalue.b()++;
            }else if(window.button()){
                int i =static_cast<int>(1.f*window.mouse_y()/window.height()*img.getDomain()(0));
                int j =static_cast<int>(1.f*window.mouse_x()/window.width()*img.getDomain()(1));
                if(img.isValid(i,j)){
                    value=  img(i,j);
                    click=true;
                    lowvalue = img(i,j)-RGBUI8(10,10,10);
                    highvalue = img(i,j)+RGBUI8(10,10,10);
                }



            }

            Mat2UI8 threshold(img.getDomain());
            Mat2UI8::IteratorEDomain it = threshold.getIteratorEDomain();
            while(it.next())
            {
                Mat2UI8::E x = it.x();
                if(img(x).r()>=lowvalue.r()&&img(x).r()<=highvalue.r()
                        &&img(x).g()>=lowvalue.g()&&img(x).g()<=highvalue.g()&&
                        img(x).b()>=lowvalue.b()&&img(x).b()<=highvalue.b()
                        )
                    threshold(x)=255;
                else
                    threshold(x)=0;

            }
            Mat2RGBUI8 visu = pop::Visualization::labelForeground(threshold,img,0.2f);

            window.display(visu);
            if(click==true)
                title ="minValue="+BasicUtility::Any2String(lowvalue)+ " maxValue="+BasicUtility::Any2String(highvalue)+", f(i,j)="+BasicUtility::Any2String(value);
            else
                title = "minValue="+BasicUtility::Any2String(lowvalue)+ " maxValue="+BasicUtility::Any2String(highvalue)+" keys XSDE for red-channel, keys VFGT for green-channel, keys NHJU for blue-channel";

            window.set_title(title.c_str());
            //            if(window.is_key()==true){
            //                window.waitTime (5);
            //            }

        }while(!window.is_closed());
    }
};


}
#endif // APPLICATION_H
