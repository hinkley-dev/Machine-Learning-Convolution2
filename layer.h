#ifndef LAYER_H
#define LAYER_H

#include <vector>
#include "vec.h"
#include "matrix.h"

class Layer
{

protected:
	Vec activation;
	Vec blame;
	size_t weightCount;


public:

	Layer(size_t inputs, size_t outputs);
	Layer(size_t inputs);
	const Vec& getActivation();
	void setBlame(const Vec& _blame);
	size_t getWeightCount();
	static size_t countTensorElements(const Vec& dims);
	


	virtual ~Layer();

	virtual void activate(const Vec& weights,const Vec& x) = 0;
	virtual void backprop(const Vec& weights, Vec& prevBlame) = 0;
	virtual void update_gradient(const Vec& x, Vec& gradient) = 0;
	virtual bool isConv() = 0;
	virtual bool isSin() = 0;

	virtual size_t getInputCount() = 0;

};

#endif
