#ifndef CONTROLVIEWMATRIX_H
#define CONTROLVIEWMATRIX_H
#include<CControl.h>
#include<QtGui>
class ControlViewMatrix:  public CControl
{
public:
    ControlViewMatrix(QWidget * parent = 0);
    virtual CControl * clone();

    void updatePlugInControl(int indexplugin, CData* data);

private:
    QTableWidget *box;
};

#endif // CONTROLVIEWMATRIX_H
