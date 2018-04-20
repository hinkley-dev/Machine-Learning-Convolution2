#include "neuralnet.h"
#include "supervised.h"

NeuralNet::NeuralNet(Rand r) : random(r)
{

}

NeuralNet::~NeuralNet()
{
  for(size_t i = 0; i < layers.size(); i++)
    		delete(layers[i]);
}

const char* NeuralNet::name()
{
  return "neural network";
}

Vec NeuralNet::get_weights()
{
  return weights;
}

void NeuralNet::train(Matrix& trainFeats, Matrix& trainLabs)
{

}


void NeuralNet::train(Matrix& trainFeats, Matrix& trainLabs, size_t batch_size, double momentum, double learning_rate)
{

  size_t trainingDataCount = trainFeats.rows();

  size_t *randomIndicies= new size_t[trainingDataCount];
  for(size_t j = 0; j < trainingDataCount; ++j)
    randomIndicies[j] = j;


   random_shuffle(&randomIndicies[0],&randomIndicies[trainingDataCount]);


  for(size_t j = 0; j < trainingDataCount; ++j)
  {
    size_t row = randomIndicies[j];
    train_incremental(trainFeats.row(row), trainLabs.row(row));

     if(j % batch_size == 0 && j > 0)
     {
       refine_weights(learning_rate);
       scale_gradient(momentum);
     }
   }

  delete[] randomIndicies;
}

size_t NeuralNet::countMisclassifications(const Matrix& features, const Matrix& labels)
{
	if(features.rows() != labels.rows())
		throw Ex("Mismatching number of rows");
	size_t mis = 0;
	for(size_t i = 0; i < features.rows(); i++)
	{
		const Vec& pred = predict(features[i]);
		const Vec& lab = labels[i];
		size_t predVal = pred.indexOfMax();
    size_t labVal = lab.indexOfMax();
			if(predVal != labVal)
			{
				mis++;
			}

	}
	return mis;
}


float NeuralNet::root_mean_squared_error(Matrix& features, Matrix& labels)
{
  float rmse = 0.0;
  float sse = sum_squared_error(features, labels);
  rmse = sqrt(sse / (features.rows()));
  return rmse;
}

void NeuralNet::setTestingData(const Matrix& testFeats, const Matrix& testLabs)
{
  testingFeatures.copy(testFeats);
  testingLabels.copy(testLabs);
}








void NeuralNet::scale_gradient(double scale)
{
  gradient *= scale;
}

void NeuralNet::train_incremental(const Vec& feat, const Vec& lab)
{

  predict(feat);
  backprop(lab);
  update_gradient(feat);

}



const Vec& NeuralNet::predict(const Vec& in)
{

  Vec layerInputs(in);
  size_t startWeight = 0;
  for(size_t i = 0; i < layers.size(); ++i)
  {
    VecWrapper layerWeights(weights, startWeight, layers[i]->getWeightCount());
    layers[i]->activate(layerWeights, layerInputs);

    layerInputs.copy(layers[i]->getActivation());
    startWeight += layers[i]->getWeightCount();
   // cout << "Layer " << i << " activation: " << endl;
    // layers[i]->getActivation().print();
    // cout << endl;
  }

  return layers[layers.size() -1]->getActivation();
}

void NeuralNet::backprop(const Vec& targetVals)
{
  Vec finalActivation(layers[layers.size() -1]->getActivation());
  Vec initialBlame(finalActivation.size());


  for(size_t i = 0; i < initialBlame.size(); ++i)
  {
    initialBlame[i] = targetVals[i] - finalActivation[i];
  }


  layers[layers.size() -1]->setBlame(initialBlame);

  size_t startWeight = weights.size();
  Vec prevBlame(initialBlame);

  for(size_t i = layers.size() - 1; i > 0; --i)
  {

    //build the weights

    layers[i]->setBlame(prevBlame);
    prevBlame.resize(layers[i]->getInputCount());
    prevBlame.fill(0.0);

    startWeight -= layers[i]->getWeightCount();
    VecWrapper layerWeights(weights, startWeight,layers[i]->getWeightCount());
    layers[i]->backprop(layerWeights, prevBlame);

  }
  layers[0]->setBlame(prevBlame);


}

void NeuralNet::update_gradient(const Vec& x)
{

  if(&x == nullptr) throw Ex("input to update gradient is null");
  if(x.size() == 0) throw Ex("input is not the right size");
  Vec in(x);
  // double lambda = 0.01;
  size_t startGradient = 0;

  for(size_t i = 0; i < layers.size(); ++i)
  {

    VecWrapper layerGradient(gradient, startGradient, layers[i]->getWeightCount());
    layers[i]->update_gradient(in, layerGradient);

    //copying over the gradient
    for(size_t j = startGradient, k = 0; k < layerGradient.size(); ++j, ++k)
    {

      gradient[j] = layerGradient[k];
  //if(i == 2) gradient[j] +=lambda*abs(weights[j]);
    }

    in.copy(layers[i]->getActivation());
    startGradient += layerGradient.size();
  }



}

void NeuralNet::init_weights()
{
  size_t startWeight = 0;

  for(size_t layerIndex = 0 ; layerIndex < layers.size(); ++layerIndex)
  {
    size_t weightItr = 0;
   for(size_t i = startWeight; i < startWeight + layers[layerIndex]->getWeightCount(); ++i)
    {
      if(i >= weights.size()) throw Ex("Math error in init weights");
      if( layers[layerIndex]->getInputCount() <= 0) throw Ex("Math error in init weights (trying to divide by 0)");
      // if(layers[layerIndex]->isConv())
      // {
      //   weights[i] = random.normal() / (double)layers[layerIndex]->getInputCount();
      // }

      if(layerIndex == 0)
      {

       // weights[i] = max(0.03, 1.0 / layers[layerIndex]->getInputCount()) * random.normal();
       if(i < (layers[layerIndex]->getWeightCount()) / 2 -1)
       {
         if(i < (layers[layerIndex]->getWeightCount() / 2 -1) /2)
            weights[i] = pi;
         else
            weights[i] = pi/2.0;
       }
       else if(i == (layers[layerIndex]->getWeightCount()) / 2 -1)
          weights[i] = 0;
       else
       {
         weightItr++;
         weights[i] = weightItr*2*pi;
         if(weightItr == (layers[layerIndex]->getWeightCount() / 2 -1) /2)
            weightItr = 0;
       }

     }
     else
     {
       if(i == startWeight) weights[i] = 0.0;
        else weights[i] = 0.01;
     }

    }
    if(layerIndex == 0)
    {
      weights[layers[layerIndex]->getWeightCount()-1] = 0.01;
    }
    startWeight += layers[layerIndex]->getWeightCount();

  }
  if(startWeight != weights.size()) throw Ex("Error, not all weights initialized");
  // for(size_t i = 0; i < weights.size(); ++i)
  // {
  //   weights[i] = i/100.0;
  // }

  gradient.fill(0.0);

}

void NeuralNet::setReg(size_t r)
{
  reg = r;
}

void NeuralNet::refine_weights(double learning_rate)
{
  weights.addScaled(learning_rate, gradient);
  if(reg ==1)
  {
    VecWrapper finalLayerWeights(weights, weights.size() - layers[layers.size() -1]->getWeightCount(),layers[layers.size() -1]->getWeightCount());
    double lambda = 0.0001;  //fro L2 0.001; for l2 0.0001
    finalLayerWeights.regularize_L1(lambda);
  }
  if(reg ==2)
  {
    VecWrapper finalLayerWeights(weights, weights.size() - layers[layers.size() -1]->getWeightCount(),layers[layers.size() -1]->getWeightCount());
    double lambda = 0.001;  //fro L2 0.001; for l2 0.0001
   finalLayerWeights*=(1-lambda);
  }



}

void NeuralNet::centralFiniteDifferencing(Vec& x, Vec& target)
{

  Vec gradientCFD(gradient.size());
  gradientCFD.fill(0.0);
  Vec cp(weights);
  double lr = 0.0003;
  for(size_t i = 0; i < weights.size(); ++i)
  {
    double orig = weights[i];
    weights[i] += lr;
    Vec positiveStep = predict(x);
    double pos = 0.0;
    for(size_t j = 0; j < positiveStep.size(); j++)
		{
			pos+= ((positiveStep[j] - target[j])*(positiveStep[j] - target[j]));
		}
    weights[i] = orig - lr;
    Vec negativeStep = predict(x);
    double neg = 0.0;
    for(size_t j = 0; j < positiveStep.size(); j++)
    {
     neg+= ((negativeStep[j] - target[j])*(negativeStep[j] - target[j]));
    }
    gradientCFD[i] = (neg - pos)/ (2*lr);
    weights[i] = orig;
    for(size_t j = 0; j < weights.size(); ++j)
      if(weights[j] != cp[j]) throw Ex("weights not set back correctly");
  }
  train_incremental(x, target);
  size_t count = 0;
  for(size_t i = 0; i < gradient.size(); ++i)
  {
    if((gradientCFD[i] - gradient[i]) / gradientCFD[i] > 0.005)
      count++;
  }
  // cout << "Gradtients with central finite differencing: " << endl;
  // gradientCFD.print();
  // cout << endl;
  // cout << "Gradtients with backprop: " << endl;
  // gradient.print();
  // cout << endl;
  //  cout << endl;
  //   cout << endl;
  // cout << "Amount of times when the difference between backprop and central finite differencing is more than 0.5%: " << count << " / " << gradient.size() << endl;
  // cout << "The 10 can be accounted but the fact that they are the biases and my finite differencing test does not treat them any differently." << endl;


}


void NeuralNet::add(Layer* l)
{

  layers.push_back(l);
  weights.resize(weights.size() + layers[layers.size()-1]->getWeightCount());
  gradient.resize(weights.size());
}

size_t NeuralNet::effective_batch_size(double momentum)
{
  return (size_t)(1/(1-momentum));
}
