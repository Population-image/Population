#ifndef OPERATORDIFFMatN_H
#define OPERATORDIFFMatN_H
#include"COperator.h"
#include"data/mat/MatN.h"
#include"algorithm/Processing.h"
using namespace pop;
class OperatorDiffMatN : public COperator
{
public:
    OperatorDiffMatN();
    void exec();
    COperator * clone();
    struct foo
    {
        template<int DIM,typename Type>
        void operator()(MatN<DIM,Type> * in1cast,BaseMatN * in2, BaseMatN * &out)throw(pexception)
        {
            if(MatN<DIM,Type> * in2cast = dynamic_cast<MatN<DIM,Type> *>(in2))
            {
                MatN<DIM,unsigned char> * outcast = new MatN<DIM,unsigned char>(in1cast->getDomain());
                (*outcast) =   (std::max((*in1cast),(*in2cast))-std::min((*in1cast),(*in2cast)));
                outcast->opposite();
                out=outcast;
            }
            else
            {
                int dim;
                string type1;
                in1cast->getInformation(type1,dim);
                string type2;
                in2->getInformation(type2,dim);
                string msg="The pixel/voxel types of input images must be the same. Here the pixel/voxel type is f1="+ type1+" and f2="+type2 ;
                throw(pexception(msg));
                out= new MatN<DIM,Type>(in1cast->getDomain());
            }

        }
    };

};

#endif // OPERATORDIFFMatN_H
