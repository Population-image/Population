#ifndef CONTROLEDITORMatN_H
#define CONTROLEDITORMatN_H


#include<CControl.h>
#include<QtGui>
#include<DataImageGrid.h>
#include<ControlEditorMatrix.h>

class ControlEditorMatN2D  : public ControlEditorMatrix
{
public:
    ControlEditorMatN2D(QWidget * parent = 0);
    void apply();
    virtual CControl * clone();
    virtual QString defaultHeaderColumn(int col);
    virtual QString defaultHeaderRaw(int raw);
};

class ControlEditorMatN3D :  public CControl
{
    Q_OBJECT
public:
    ControlEditorMatN3D(QWidget * parent = 0);
    virtual CControl * clone();

    virtual string toString();
    virtual void fromString(string str);
    void apply();
public slots:
    void addX();
    void addY();
    void addZ();
    void rmX();
    void rmY();
    void rmZ();
    void geInformation();
    void changeZ(int z);
protected:
    void addXTable();
    void addYTable();
    void rmXTable();
    void rmYTable();

protected:
    QTableWidget * box;
    QPushButton* buttonaddx;
    QPushButton* buttonaddy;
    QPushButton* buttonaddz;
    QPushButton* buttonrmx;
    QPushButton* buttonrmy;
    QPushButton* buttonrmz;
    QSpinBox * spinz;
    QPushButton* save;
    MatN<3,pop::F64> img;
    int _slice;
    bool test;
};
#endif // CONTROLEDITORMatN_H
