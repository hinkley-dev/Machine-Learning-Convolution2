#ifndef LAYERSINUSOID_H
#define LAYERSINUSOID_H

#include "vec.h"
#include "matrix.h"
#include "layer.h"
#include <cmath>

using namespace std;

class LayerSinusoid : public Layer
{

protected:
  size_t m_inputs;
  Vec in;


public:

	LayerSinusoid(size_t inputs);
	~LayerSinusoid();

	virtual void activate(const Vec& weights,const Vec& x);
	virtual void backprop(const Vec& weights, Vec& prevBlame);
	virtual void update_gradient(const Vec& x, Vec& gradient);

	virtual size_t getInputCount();
  virtual bool isConv() { return false; }
    virtual bool isSin() { return true; }

};

#endif
