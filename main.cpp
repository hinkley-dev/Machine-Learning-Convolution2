// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <exception>
#include <string>
#include <memory>
#include "error.h"
#include "string.h"
#include "rand.h"
#include "matrix.h"
#include "supervised.h"
#include "baseline.h"
#include "layer.h"
#include "layerlinear.h"
#include "layerconv.h"
#include "layerleakyrect.h"
#include "layermaxpooling2d.h"
#include "neuralnet.h"
#include <algorithm>
#include "imputer.h"
#include "nomcat.h"
#include "normalizer.h"
#include "svg.h"
#include <fstream>
#include <time.h>

using std::cout;
using std::cerr;
using std::string;
using std::auto_ptr;



void testLearner(SupervisedLearner& learner)
{


}


size_t convertToDecimal(const Vec& oneHot)
{
	return oneHot.indexOfMax();
}


void make_features_and_labels(const Matrix& data, Matrix& feats, Matrix& labs)
{
	feats.setSize(data.rows(), data.cols() -1);
	feats.copyBlock(0,0, data, 0,0, data.rows(), data.cols() -1);
	labs.setSize(data.rows(), 1);
	labs.copyBlock(0,0, data, 0, data.cols()-1, data.rows(), 1);
}

void make_training_and_testing(const Matrix& feats, const Matrix& labs, Matrix& trainFeats, Matrix& trainLabs, Matrix& testFeats, Matrix& testLabs, double trainRation)
{
	size_t trainingDataCount = (size_t)(feats.rows() * trainRation);
	size_t testingDataCount = feats.rows() - trainingDataCount;

	trainFeats.setSize(trainingDataCount, feats.cols());
	trainLabs.setSize(trainingDataCount, labs.cols());
	testFeats.setSize(testingDataCount, feats.cols());
	testLabs.setSize(testingDataCount, labs.cols());

	trainFeats.copyBlock(0,0,feats,0,0,trainingDataCount,feats.cols());
	trainLabs.copyBlock(0,0, labs, 0,0, trainingDataCount, labs.cols());

	testFeats.copyBlock(0,0, feats, trainingDataCount, 0, testingDataCount,feats.cols());
	testLabs.copyBlock(0,0,labs, trainingDataCount, 0, testingDataCount, labs.cols());



}

void preprocessData(Matrix& m)
{
	Imputer imputer;
	Normalizer normalizer;
	NomCat nomcat;



	cout << endl;
	//
	//
	 	imputer.train(m);
		Matrix* temp = imputer.transformBatch(m);


		normalizer.train(*temp);
		temp = normalizer.transformBatch(*temp);

		nomcat.train(*temp);
		temp = nomcat.transformBatch(*temp);

		m.copy(*temp);

		delete temp;

}


void testHypothyroid(Rand random)
{
	cout << "start loading data" << endl;
	//load data
	string fn = "data/";
	Matrix hypothyroid_data;
	hypothyroid_data.loadARFF(fn + "hypothyroid.arff");


	Matrix feats;
	Matrix labs;

	//mixup the Data
	for(size_t i = 0; i < hypothyroid_data.rows(); ++i)
	{
		size_t firstRow = random.next(hypothyroid_data.rows());
		size_t secondRow = random.next(hypothyroid_data.rows());

		hypothyroid_data.swapRows(firstRow, secondRow);
	}


	make_features_and_labels(hypothyroid_data, feats, labs);
	preprocessData(feats);
	preprocessData(labs);

	double trainRatio = 0.8;


	Matrix trainFeats;
	Matrix trainLabs;
	Matrix testFeats;
	Matrix testLabs;
	make_training_and_testing(feats, labs, trainFeats,  trainLabs,  testFeats,  testLabs, trainRatio);

	//1. imputer... replaces gaps
	//2. normalizer... normalize the data
	//3. nomcat... replaces nominal vals with vectors of continuous values
	//lecture after momentum.





	cout << "done loading data" << endl;




	if(trainLabs.rows() != trainFeats.rows() || testLabs.rows() != testFeats.rows())
		throw Ex("invalid data in MNIST upload");




	//size_t testingDataCount = testFeats.rows();


	//nn.setTestingData(testFeats, testLabs);

//	double learning_rate = 0.03;





		//size_t miss = 0;



		for(size_t i = 0; i < 1; ++i)
		{
			//nn.train(trainFeats, trainLabs, 1, 0.0, learning_rate);
			//miss = nn.countMisclassifications(testFeats, testLabs);
			//cout << miss << endl;
		}



}

void debugSpew(Rand random)
{
	NeuralNet nn(random);
	// nn.add(new LayerLinear(trainFeats.cols(),100));
	// nn.add(new LayerTanh(100));
	// nn.add(new LayerLinear(100,4));
	// nn.add(new LayerTanh(4));
	nn.add(new LayerConv({4, 4}, {3, 3, 1}, {4, 4, 1}));
	nn.add(new LayerConv({4, 4}, {3, 3, 2}, {4, 4, 2}));
	nn.add(new LayerLeakyRectifier(4*4*2));
	nn.add(new LayerMaxPooling2D(4,4,2));

	nn.init_weights();
	Vec in(16);
	for(size_t i = 0; i < in.size(); ++i)
		in[i] = double(i)/10;

	cout << "input vector: " << endl;
	in.print();
	cout << endl;
	cout << endl;
	nn.predict(in);

	Vec target(8);
	for(size_t i = target.size() -1, j = 0; j < target.size(); --i, ++j)
	{
		target[j] = (i/10.0);
	}
	cout << "target vals: " << endl;
	target.print();
	cout << endl;
	cout << endl;
	cout << "initial weights: " << endl;
	nn.get_weights().print();
	cout << endl;
	cout << endl;

		nn.train_incremental(in, target);

	cout << "after backprop weights: " << endl;
	nn.get_weights().print();
	cout << endl;
	cout << endl;
	nn.centralFiniteDifferencing(in, target);
	cout << "after cfd weights: " << endl;
	nn.get_weights().print();
	cout << endl;
	cout << endl;


	return;
}

void unitTest(Rand random)
{
	NeuralNet nn(random);
	// nn.add(new LayerLinear(trainFeats.cols(),100));
	// nn.add(new LayerTanh(100));
	// nn.add(new LayerLinear(100,4));
	// nn.add(new LayerTanh(4));
	nn.add(new LayerConv({8, 8}, {5, 5, 4}, {8, 8, 4}));
nn.add(new LayerLeakyRectifier(8 * 8 * 4));
nn.add(new LayerMaxPooling2D(8, 8, 4));
nn.add(new LayerConv({4, 4, 4}, {3, 3, 4, 6}, {4, 4, 1, 6}));
nn.add(new LayerLeakyRectifier(4 * 4 * 6));
nn.add(new LayerMaxPooling2D(4, 4, 1 * 6));
nn.add(new LayerLinear(2 * 2 * 6, 3));

	nn.init_weights();
	Vec in(64);
	for(size_t i = 0; i < in.size(); ++i)
		in[i] = double(i)/10;

	Vec target(3);
	for(size_t i = 0; i < target.size(); ++i)
		target[i] = i /10.0;


		nn.train_incremental(in, target);

		nn.centralFiniteDifferencing(in, target);




	return;
}

int main(int argc, char *argv[])
{
	Rand random(123);
	enableFloatingPointExceptions();
	int ret = 1;


	try
	{

		ret = 0;
	}
	catch(const std::exception& e)
	{
		cerr << "An error occurred: " << e.what() << "\n";
	}
	try
	{
		//cout << 325 << endl;
		//convergenceDetection(random);
		unitTest(random);
		// NeuralNet nn(random);
		// nn.addLayerLinear(1,2);
		// nn.addLayerTanh(2);
		// nn.addLayerLinear(2,1);
		// nn.init_weights();
		// Vec in(1);
		// in.fill(0.3);
		// Vec target(1);
		// target.fill(0.7);
		// for(size_t i = 0; i < 3; ++i)
		// {
		// 	nn.train_incremental(in, target );
    //   nn.refine_weights(0.0);
    //    nn.scale_gradient(0.0);
		// }
		//  nn.get_weights().print();





		ret = 0;
	}
	catch(const std::exception& e)
	{
		cerr << "An error occurred: " << e.what() << "\n";
	}
	cout.flush();
	cerr.flush();
	return ret;
}
