#ifndef CONTROLVIEWPOINT_H
#define CONTROLVIEWPOINT_H

#include<CControl.h>
#include<QtGui>
class ControlViewPoint :  public CControl
{
public:
    ControlViewPoint(QWidget * parent = 0);
    virtual CControl * clone();

    void updatePlugInControl(int indexplugin, CData* data);

private:
    QTableWidget *box;
};
#endif // CONTROLVIEWPOINT_H
