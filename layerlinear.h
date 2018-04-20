#ifndef LAYERLINEAR_H
#define LAYERLINEAR_H

#include <vector>
#include "vec.h"
#include "matrix.h"

using namespace std;

class LayerLinear : public Layer
{
  size_t m_inputs;
  size_t m_outputs;





public:
	LayerLinear(size_t inputs, size_t outputs);
	~LayerLinear();

	virtual void activate(const Vec& weights,const Vec& x);
  virtual void backprop(const Vec& weights, Vec& prevBlame);
	virtual void update_gradient(const Vec& x, Vec& gradient);
  virtual size_t getInputCount();
  virtual bool isConv() { return false; }
    virtual bool isSin() { return false; }
};



#endif
