#ifndef LAYERCONV_H
#define LAYERCONV_H

#include "vec.h"
#include "matrix.h"
#include "layer.h"
#include <cmath>

using namespace std;

class LayerConv : public Layer
{

protected:
  size_t m_inputs;
  Vec m_inputDims;
  Vec m_filterDims;
  Vec m_outputDims;





public:

  LayerConv(const Vec& inputDims,
                       const Vec& filterDims,
                       const Vec& outputDims);
	~LayerConv();

	virtual void activate(const Vec& weights,const Vec& x);
	virtual void backprop(const Vec& weights, Vec& prevBlame);
	virtual void update_gradient(const Vec& x, Vec& gradient);
  size_t getFilterSize();
  virtual bool isConv() { return true; }
  virtual bool isSin() { return false; }

	virtual size_t getInputCount();

};


//making it easier to access list items
template<class T>
struct _init_list_with_square_brackets {
    const std::initializer_list<T>& list;
    _init_list_with_square_brackets(const std::initializer_list<T>& _list): list(_list) {}
    T operator[](unsigned int index) {
        return *(list.begin() + index);
    }
};

// a function, with the short name _ (underscore) for creating
// the _init_list_with_square_brackets out of a "regular" std::initializer_list
template<class T>
_init_list_with_square_brackets<T> _(const std::initializer_list<T>& list) {
    return _init_list_with_square_brackets<T>(list);
}

#endif
