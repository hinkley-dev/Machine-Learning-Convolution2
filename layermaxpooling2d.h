#ifndef LAYERMAXPOOLING2D_H
#define LAYERMAXPOOLING2D_H

#include "vec.h"
#include "matrix.h"
#include "layer.h"
#include <cmath>

using namespace std;

class LayerMaxPooling2D : public Layer
{

protected:
  size_t m_inputs;
  size_t m_inputDim;
  size_t m_layerCount;
  Vec maxIndicies;
  const size_t poolDim = 2; //because 2d pooling

private:
  size_t maxIndexOfPool(const Vec& x, size_t a, size_t b, size_t c, size_t d);

public:

	LayerMaxPooling2D(size_t filterDim1, size_t filterDim2, size_t filterCount);
	~LayerMaxPooling2D();

	virtual void activate(const Vec& weights,const Vec& x);
	virtual void backprop(const Vec& weights, Vec& prevBlame);
	virtual void update_gradient(const Vec& x, Vec& gradient);

	virtual size_t getInputCount();
  virtual bool isConv() { return false; }
    virtual bool isSin() { return false; }

};

#endif
