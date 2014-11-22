#ifndef OPERATORLOADFROMDIRECTORYMatN_H
#define OPERATORLOADFROMDIRECTORYMatN_H

#include"COperator.h"
#include"data/mat/MatN.h"
#include<QImage>
#include"data/utility/CollectorExecutionInformation.h"
#include"dependency/ConvertorQImage.h"
using namespace pop;
class OperatorLoadFromDirectoryMatN: public COperator
{
public:
    OperatorLoadFromDirectoryMatN();
    void exec();
    void initState();
    COperator * clone();
    struct foo
    {
        template<int DIM,typename Type>
        void operator()(MatN<DIM,Type> * img ,vector<string> vec,  BaseMatN * &out)
        {
            CollectorExecutionInformationSingleton::getInstance()->startExecution("Load");
            VecN<DIM+1,int> d;
            d(0)=img->getDomain()(0);
            d(1)=img->getDomain()(1);
            d(2)=vec.size();
            MatN<DIM+1,Type> * outcast = new MatN<DIM+1,Type>(d);
            for(int i = 0;i<(int)vec.size();i++){
                CollectorExecutionInformationSingleton::getInstance()->progression(1.0*i/vec.size(),"load "+vec[i]);
                MatN<DIM,Type> plane(img->getDomain());
                string ext =  UtilityString::getExtension(vec[i]);
                if(ext==".pgm")
                {

                    plane.load(vec[i].c_str());
                }else
                {
                    QImage qimg;
                    qimg.load(vec[i].c_str());
                    plane = ConvertorQImage::fromQImage<DIM,Type>(qimg);
                }
                outcast->setPlane(2,i,plane);
            }
            CollectorExecutionInformationSingleton::getInstance()->endExecution("Load");
            out=outcast;
        }
    };
};

#endif // OPERATORLOADFROMDIRECTORYMatN_H
