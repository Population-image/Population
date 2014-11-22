#include "data/utility/XML.h"
#include "dependency/pugixml.hpp"
#include <sstream>
namespace pop{
class XMLNode::impl
{
public:
    pugi::xml_node node;
};
class XMLDocument::impl
{
public:
    pugi::xml_document doc;
};
XMLNode::XMLNode()
{
    _pImpl = new XMLNode::impl();
}
XMLNode::XMLNode(const XMLNode & node)
{
    _pImpl = new XMLNode::impl();
    _pImpl->node = node._pImpl->node;
}

XMLNode& XMLNode::operator ()(const XMLNode & node){
    _pImpl->node = node._pImpl->node;
    return *this;
}
XMLNode::~XMLNode()
{
    //    delete _pImpl;
}

void XMLNode::setName(std::string name){
    _pImpl->node.set_name(name.c_str());
}
void XMLNode::setValue(std::string value){
    //    _pImpl->node.set_value(value.c_str());
    if(_pImpl->node.first_child().set_value(value.c_str()) == false)
    {
        _pImpl->node.append_child(pugi::node_pcdata).set_value(value.c_str());
    }
}
XMLNode XMLNode::addChild(std::string name){
    XMLNode nod;
    nod._pImpl->node=  _pImpl->node.append_child(name.c_str());
    return nod;
}
bool XMLNode::rmChild(const XMLNode& node){
    return _pImpl->node.remove_child(node._pImpl->node);
}
std::string XMLNode::getName()const{
    return _pImpl->node.name();
}
std::string XMLNode::getValue()const{
    return _pImpl->node.first_child().value();
}
XMLNode XMLNode::getChild(std::string childname)const{
    XMLNode nod;
    nod._pImpl->node= _pImpl->node.child(childname.c_str());
    return nod;
}
XMLNode XMLNode::firstChild()const{
    XMLNode nod;
    nod._pImpl->node= _pImpl->node.first_child();
    return nod;
}
XMLNode::operator bool() const {
    return! _pImpl->node.operator !();
}


XMLNode XMLNode::nextSibling()const{
    XMLNode nod;
    nod._pImpl->node= _pImpl->node.next_sibling();
    return nod;
}
bool XMLNode::hasAttribute(std::string attributename)const{
    if(_pImpl->node.attribute(attributename.c_str()))
        return true;
    else
        return false;
}

std::string XMLNode::getAttribute(std::string attributename) const{
    std::string test = _pImpl->node.attribute(attributename.c_str()).value();;
    return test;
}
void XMLNode::setAttribute(std::string name,std::string  value){
    _pImpl->node.attribute(name.c_str()).set_value(value.c_str());
}
void XMLNode::addAttribute(std::string name,std::string  value){
    _pImpl->node.append_attribute(name.c_str()) = value.c_str();
}
bool XMLNode::rmAttribute(std::string name){
    return _pImpl->node.remove_attribute(name.c_str());
}


XMLDocument::XMLDocument()
{
    _pImpl = new XMLDocument::impl();
}
XMLDocument::~XMLDocument()
{
    delete _pImpl;
}

void XMLDocument::load(std::string file){
    _pImpl->doc.load_file(file.c_str());
}
void  XMLDocument::loadFromByteArray(const char *  file){
    _pImpl->doc.load(file);
}

//void XMLDocument::load(std::istream & is){
//    std::string str;
//    while(is >> str);
//    _pImpl->doc.load_buffer(str.c_str(),str.size());
//}

void XMLDocument::save(std::string file)const {
    _pImpl->doc.save_file(file.c_str());
}
//void save(std::ostream file)const {

//}

XMLNode XMLDocument::getChild(std::string name){
    XMLNode node;
    node._pImpl->node = _pImpl->doc.child(name.c_str());
    return node;
}

XMLNode XMLDocument::addChild(std::string name){
    XMLNode node;
    node._pImpl->node =  _pImpl->doc.append_child(name.c_str());
    return node;
}

}
