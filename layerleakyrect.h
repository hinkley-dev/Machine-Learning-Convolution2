#ifndef LAYERLEAKYRECT_H
#define LAYERLEAKYRECT_H

#include "vec.h"
#include "matrix.h"
#include "layer.h"
#include <cmath>

using namespace std;

class LayerLeakyRectifier : public Layer
{

protected:
  size_t m_inputs;



public:

	LayerLeakyRectifier(size_t inputs);
	~LayerLeakyRectifier();

	virtual void activate(const Vec& weights,const Vec& x);
	virtual void backprop(const Vec& weights, Vec& prevBlame);
	virtual void update_gradient(const Vec& x, Vec& gradient);

	virtual size_t getInputCount();
  virtual bool isConv() { return false; }
    virtual bool isSin() { return false; }

};

#endif
