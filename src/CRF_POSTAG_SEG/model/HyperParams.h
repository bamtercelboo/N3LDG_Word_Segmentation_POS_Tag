#ifndef SRC_HyperParams_H_
#define SRC_HyperParams_H_

#include "N3LDG.h"
#include "Example.h"
#include "Options.h"

using namespace nr;
using namespace std;

struct HyperParams{

	// must assign
	int wordcontext;
	int hiddensize;
	int rnnhiddensize;
	dtype dropOut;


	dtype nnRegular; // for optimization
	dtype adaAlpha;  // for optimization
	dtype adaEps; // for optimization



	//auto generated
	int wordwindow;
	int wordDim;
	vector<int> typeDims;
	int unitsize;
	int inputsize;
	int labelSize;
	int batch;
public:
	HyperParams(){
		bAssigned = false;
	}

public:
	void setRequared(Options& opt){
		wordcontext = opt.wordcontext;
		hiddensize = opt.hiddenSize;
		rnnhiddensize = opt.rnnHiddenSize;
		dropOut = opt.dropProb;
		nnRegular = opt.regParameter;
		adaAlpha = opt.adaAlpha;
		adaEps = opt.adaEps;
		batch = opt.batchSize;

		bAssigned = true;
	}

	void clear(){
		bAssigned = false;
	}

	bool bValid(){
		return bAssigned;
	}

	void saveModel(std::ofstream &os) const {
		//os << nnRegular << std::endl;
		//os << adaAlpha << std::endl;
		//os << adaEps << std::endl;

		//os << hiddenSize << std::endl;
		//os << wordContext << std::endl;
		//os << wordWindow << std::endl;
		//os << windowOutput << std::endl;
		//os << dropProb << std::endl;


		//os << wordDim << std::endl;
		//os << inputSize << std::endl;
		//os << labelSize << std::endl;
	}

	void loadModel(std::ifstream &is) {
		//is >> nnRegular;
		//is >> adaAlpha;
		//is >> adaEps;

		//is >> hiddenSize;
		//is >> wordContext;
		//is >> wordWindow;
		//is >> windowOutput;
		//is >> dropProb;


		//is >> wordDim;
		//is >> inputSize;
		//is >> labelSize;

		//bAssigned = true;
	}

public:

	void print(){

	}

private:
	bool bAssigned;
};


#endif /* SRC_HyperParams_H_ */