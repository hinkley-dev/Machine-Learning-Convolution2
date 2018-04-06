#ifndef LAYERTANH_H
#define LAYERTANH_H

#include "vec.h"
#include "matrix.h"
#include "layer.h"
#include <cmath>

using namespace std;

class LayerTanh : public Layer
{

protected:
  size_t m_inputs;



public:

	LayerTanh(size_t inputs);
	~LayerTanh();

	virtual void activate(const Vec& weights,const Vec& x);
	virtual void backprop(const Vec& weights, Vec& prevBlame);
	virtual void update_gradient(const Vec& x, Vec& gradient);

	virtual size_t getInputCount();
  virtual bool isConv() { return false; }

};

#endif
