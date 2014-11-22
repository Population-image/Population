#ifndef OPERATORsetPlaneMatN_H
#define OPERATORsetPlaneMatN_H

#include<COperator.h>
#include"data/mat/MatN.h"
using namespace pop;
class OperatorsetPlaneMatN : public COperator
{
public:
    OperatorsetPlaneMatN();
    void exec();
    COperator * clone();
    void initState();
    struct foo
    {
        template<int DIM,typename Type>
        void operator()(MatN<DIM,Type> * in1cast, BaseMatN * plane, int index, int coordinate, BaseMatN * &out)throw(std::string)
        {
            if(MatN<DIM-1,Type> *planec = dynamic_cast<MatN<DIM-1,Type> *>(plane)){
                MatN<DIM,Type> * outcast = new MatN<DIM,Type>(*in1cast);
                outcast->setPlane(coordinate,index,*planec);
                out=outcast;
            }
            else {
                throw(string("The plane must have the same pixel type and the dimension minus one of the 3dimage"));
            }

        }
    };

};


#endif // OPERATORsetPlaneMatN_H
