//-------------------------------------------------------------
// Hidden Conditional Random Field Library - DataSequenceRealtime Component
//
// Moved/maintained by Julien-Charles Levesque
// June 14th, 2011

#include "dataset.h"

DataSequenceRealtime::DataSequenceRealtime()
{
	precompFeatures = 0;
	windowSize = 0;
	width = 0;
	height = 0;
	pos = 0;
	alpha = 0;
	ready = false;
	
}

DataSequenceRealtime::DataSequenceRealtime(int windowSize, int bufferLength, int height, int numberofLabels)
{
	init(windowSize, bufferLength,height, numberofLabels);
}

DataSequenceRealtime::~DataSequenceRealtime()
{
	if(alpha != NULL)
		delete alpha;
}

int DataSequenceRealtime::init(int windowSize, int bufferLength, int height, int numberofLabels)
{
	precompFeatures = new dMatrix(windowSize+bufferLength,height);
	this->windowSize = windowSize;
	this->width = windowSize+bufferLength;
	this->height = height;
	pos = 0;
	ready = false;
	alpha = 0;
	//alpha->create(numberofLabels);
	//alpha->set(0);
	return 1;
}

void DataSequenceRealtime::push_back(const dVector* const featureVector)
{
	for(int row=0; row<height; row++)	
		precompFeatures->setValue(row, pos, featureVector->getValue(row));	
	pos++;
	if(pos == width)
	{
		pos = 0;
		ready = true;
	}
}

int DataSequenceRealtime::getWindowSize()
{
	return windowSize;
}

int DataSequenceRealtime::getWidth()
{
	return width;
}

dVector* DataSequenceRealtime::getAlpha()
{
	return alpha;
}

void DataSequenceRealtime::initializeAlpha(int height)
{
	alpha = new dVector(height);
	alpha->set(0);
}

int DataSequenceRealtime:: getPosition()
{
	return pos;
}

bool DataSequenceRealtime::isReady()
{
	return ready;
}
