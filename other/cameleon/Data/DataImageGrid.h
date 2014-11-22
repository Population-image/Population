#ifndef DATAMatN_H
#define DATAMatN_H
#include<CDataByFile.h>
#include"../Data/MatNCameleon/MatNN.h"
#include"../Data/MatNCameleon/MatNNListType.h"
#include"data/GP/Dynamic2Static.h"
#include"data/mat/MatNListType.h"
#include"GP/Factory.h"
#include"GP/Singleton.h"
#include"GP/CartesianProduct.h"
#include"GP/TypeTraitsTemplateTemplate.h"
#include"QFileInfo"
using namespace pop;

class DataMatN : public CDataByFile<pop::BaseMatN>
{
public:
    DataMatN();
     static string KEY;
     DataMatN * clone();
     shared_ptr<BaseMatN> getDataByFile();
     void setDataByFile(shared_ptr<BaseMatN> type);
     void setDataByCopy(shared_ptr<BaseMatN> type);

//     ImageD_UC * getImageD_UC()throw(pexception);
//     Image3D_UC * getImage3D_UC()throw(pexception);
//     ImageD_Color * getImageD_Color()throw(pexception);
//     Image3D_Color * getImage3D_Color()throw(pexception);
};
#endif // DATAMatN_H
