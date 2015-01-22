#ifndef BSPTREE_H
#define BSPTREE_H
#include <queue>
#include <vector>
#include <algorithm>
#include "PopulationConfig.h"
#include "data/vec/VecN.h"
namespace pop{



namespace Private {

struct DistanceDefault
{
    int _norm;
    DistanceDefault(int norm=2)
        :_norm(norm)
    {

    }
    template<typename T>
    F32 operator()(const T& a, const T& b) {
        return distance( a, b ,_norm);
    }
};
}
template<int Dim,typename Type=F32>
class POP_EXPORTS KDTree
{
public:
    Private::DistanceDefault dist;
    KDTree() : _root(0) {}

    ~KDTree() {
        delete _root;
    }

    void create( const std::vector<VecN<Dim,Type> >& items ) {
        delete _root;
        std::vector<VecN<Dim,Type> >  item(items);
        _root = this->create(item, 0,items.size(),0);
    }
    void addItem(const VecN<Dim,Type> & item){
        this->addItem(_root, item,0);
    }
    void search( const VecN<Dim,Type> & target, VecN<Dim,Type> & result,F32& distance_min)
    {
        distance_min = NumericLimits<F32>::maximumRange();
        search( _root,target, result, distance_min,0 );
    }
private:

    struct Node
    {
        VecN<Dim,Type> value;
        Node* left;
        Node* right;

        Node() :
            left(0), right(0) {}

        ~Node() {
            delete left;
            delete right;
        }
    }* _root;
    struct Compare
    {
        int _axis;
        Compare()
            :_axis(0)
        {

        }

        Compare(int axis)
            :_axis(axis)
        {

        }
        bool operator()(const VecN<Dim,Type> & x1,const VecN<Dim,Type>& x2)
        {
            if(x1(_axis) <x2(_axis))
                return true;
            else if(x1(_axis) >x2(_axis))
                return false;
            else
                return x1<x2;
        }
    };


    void addItem(Node *& node,const VecN<Dim,Type> & item,unsigned int depth){
        if(node==NULL){
            Node* _node = new Node();
            _node->value = item;
            node = _node;
        }else{
            if(item(depth%Dim)<node->value(depth%Dim)){
                depth++;
                addItem(node->left,item,depth);
            }else{
                depth++;
                addItem(node->right,item,depth);
            }
        }
    }
    Compare comp;
    Node * create(std::vector<VecN<Dim,Type> >& items,unsigned int lower,unsigned int upper,int depth){
        if ( upper == lower ) {
            return NULL;
        }
        else {
            Node* node = new Node();
            std::sort(items.begin()+lower,items.begin()+upper,Compare(depth%Dim));

            unsigned int median = ( upper + lower ) / 2;
            node->value = items[median];
            depth++;
            node->left = create(items, lower , median,depth );
            node->right = create(items, median+1, upper,depth );
            return node;
        }

    }


    void search(const Node * node, const VecN<Dim,Type> & target, VecN<Dim,Type> & result,F32& distance_min,int depth)
    {
        if(node!=NULL){
            F32 distance = dist(target,node->value);
            if(distance<distance_min){
                distance_min = distance;
                result =node->value;
            }
            comp._axis =depth%Dim;
            if(comp(target,node->value)){
                int depth1=depth;
                depth1++;
                search( node->left,target, result, distance_min,depth1);

                //min distance with the hyperplane
                VecN<Dim,Type> v;
                for(int i=0;i<Dim;i++)
                    if(i!=depth%Dim)
                        v(i)=target(i);
                v(depth%Dim)=node->value(depth%Dim);
                F32 distance_hyper = dist(target,v);
                if(distance_hyper<=distance_min){
                    search( node->right,target, result, distance_min,depth1);
                }
            }else{
                int depth1=depth;
                depth1++;
                search( node->right,target, result, distance_min,depth1);

                //min distance with the hyperplane
                VecN<Dim,Type> v;
                for(int i=0;i<Dim;i++)
                    if(i!=depth%Dim)
                        v(i)=target(i);
                v(depth%Dim)=node->value(depth%Dim);
                F32 distance_hyper = dist(target,v);
                if(distance_hyper<=distance_min){
                    search( node->left,target, result, distance_min,depth1);
                }
            }
        }
    }

};




// A VP-Tree implementation, by Steve Hanov. (steve.hanov@gmail.com)
template<typename T,typename DistanceOperator=Private::DistanceDefault>
class POP_EXPORTS VpTree
{
public:
    DistanceOperator op_dist;
    VpTree() : _root(0) {}

    ~VpTree() {
        delete _root;
    }

    void create( const std::vector<T>& items ) {
        delete _root;
        _items = items;
        _root = buildFromVecNs(0, items.size());
    }

    void search( const T& target, int k, std::vector<T>& results,
                 std::vector<F32>& distances)
    {
        std::priority_queue<HeapItem> heap;

        _tau = NumericLimits<F32>::maximumRange();
        search( _root, target, k, heap );

        results.clear(); distances.clear();

        while( !heap.empty() ) {
            results.push_back( _items[heap.top().index] );
            distances.push_back( heap.top().dist );
            heap.pop();
        }

        std::reverse( results.begin(), results.end() );
        std::reverse( distances.begin(), distances.end() );
    }


private:
    std::vector<T> _items;


    F32 _tau;

    struct Node
    {
        int index;
        F32 threshold;
        Node* left;
        Node* right;

        Node() :
            index(0), threshold(0.), left(0), right(0) {}

        ~Node() {
            delete left;
            delete right;
        }
    }* _root;
    struct HeapItem {
        HeapItem( int index_value, F32 dist_value) :
            index(index_value), dist(dist_value) {}
        int index;
        F32 dist;
        bool operator<( const HeapItem& o ) const {
            return dist < o.dist;
        }
    };

    struct DistanceComparator
    {
        const T& item;
        DistanceOperator op_dist;
        DistanceComparator( const T& item_value ) : item(item_value) {}
        bool operator()(const T& a, const T& b) {
            return op_dist( item, a ) <op_dist( item, b );
        }
    };

    Node* buildFromVecNs( int lower, int upper )
    {
        if ( upper == lower ) {
            return NULL;
        }

        Node* node = new Node();
        node->index = lower;

        if ( upper - lower > 1 ) {

            // choose an arbitrary VecN and move it to the start
            int i = (int)((F32)rand() / RAND_MAX * (upper - lower - 1) ) + lower;
            std::swap( _items[lower], _items[i] );

            int median = ( upper + lower ) / 2;

            // partitian around the median distance
            std::nth_element(
                        _items.begin() + lower + 1,
                        _items.begin() + median,
                        _items.begin() + upper,
                        DistanceComparator( _items[lower] ));

            // what was the median?
            node->threshold = op_dist( _items[lower], _items[median] );

            node->index = lower;
            node->left = buildFromVecNs( lower + 1, median );
            node->right = buildFromVecNs( median, upper );
        }

        return node;
    }

    void search( Node* node, const T& target, int k,
                 std::priority_queue<HeapItem>& heap )
    {
        if ( node == NULL ) return;

        F32 dist = op_dist( _items[node->index], target );
        //printf("dist=%g tau=%gn", dist, _tau );

        if ( dist < _tau ) {
            if ((int) heap.size() == k ) heap.pop();
            heap.push( HeapItem(node->index, dist) );
            if ( (int) heap.size() == k ) _tau = heap.top().dist;
        }

        if ( node->left == NULL && node->right == NULL ) {
            return;
        }

        if ( dist < node->threshold ) {
            if ( dist - _tau <= node->threshold ) {
                search( node->left, target, k, heap );
            }

            if ( dist + _tau >= node->threshold ) {
                search( node->right, target, k, heap );
            }

        } else {
            if ( dist + _tau >= node->threshold ) {
                search( node->right, target, k, heap );
            }

            if ( dist - _tau <= node->threshold ) {
                search( node->left, target, k, heap );
            }
        }
    }
};
}
#endif // BSPTREE_H
