#ifndef CONTROLEDITORMATRIX_H
#define CONTROLEDITORMATRIX_H
#include<EditorTable.h>
class ControlEditorMatrix : public EditorTable
{
public:
    ControlEditorMatrix(QWidget * parent = 0);
    void apply();
    virtual CControl * clone();

protected:
    virtual QString defaultString();
    virtual QString defaultHeaderColumn(int col);
    virtual QString defaultHeaderRaw(int raw);

};

#endif // CONTROLEDITORMATRIX_H
