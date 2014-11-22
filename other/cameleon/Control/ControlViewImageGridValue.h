#ifndef CONTROLVIEWMatNVALUE_H
#define CONTROLVIEWMatNVALUE_H

#include<CControl.h>
#include<QtGui>
#include"data/mat/MatN.h"
using namespace pop;
class ControlViewMatNValue :  public CControl
{
public:
    ControlViewMatNValue(QWidget * parent = 0);
    virtual CControl * clone();

    void updatePlugInControl(int indexplugin, CData* data);


    struct foo
    {
        template<typename Type>
        void operator()(MatN<2,Type> * incast,QTableWidget *box)throw(pexception){
            box->setRowCount(incast->getDomain()(0));
            box->setColumnCount(incast->getDomain()(1));
            VecN<2,int> x;
            for(x[0]=0;x[0]<incast->getDomain()(0);x[0]++){
                for(x[1]=0;x[1]<incast->getDomain()(1);x[1]++){
                    string str = UtilityString::Any2String(incast->operator ()(x));
                    QTableWidgetItem* item = new QTableWidgetItem(str.c_str());
                    box->setItem(x[1],x[0],item);
                }
            }
        }
    };
private:
    QTableWidget *box;
};

#endif // CONTROLVIEWMatNVALUE_H
