#ifndef SRC_GraphBuilder_H_
#define SRC_GraphBuilder_H_

#include "ModelParams.h"


// Each model consists of two parts, building neural graph and defining output losses.
struct GraphBuilder {
public:
	const static int max_sentence_length = 256;

public:
	// node instances
	vector<LookupNode> word_inputs;
	LSTM1Builder lstm_left;
	LSTM1Builder lstm_right;
	vector<ConcatNode> _lstm_concat;
	vector<LinearNode> output;

	int type_num;


	Graph *_pcg;
	// node pointers
public:
	GraphBuilder(){
	}

	~GraphBuilder(){
		clear();
	}

public:
	//allocate enough nodes 
	inline void createNodes(int sent_length){
		//type_num = typeNum;
		//resizeVec(word_inputs, sent_length, type_num + 1);
		word_inputs.resize(sent_length);
		lstm_left.resize(sent_length);
		lstm_right.resize(sent_length);
		_lstm_concat.resize(sent_length);
		output.resize(sent_length);

	}

	inline void clear(){
		//Graph::clear();
		word_inputs.clear();
		lstm_left.clear();
		lstm_right.clear();
		//_lstm_concat.clear();
		output.clear();
	}

public:
	//inline void initial(Graph* _pcg, ModelParams& model, HyperParams& opts, AlignedMemoryPool* mem){
	inline void initial(Graph* _pcg, ModelParams& model, HyperParams& opts){
		this->_pcg = _pcg;
		int maxsize = word_inputs.size();
		for (int idx = 0; idx < maxsize; idx++) {
			word_inputs[idx].setParam(&model.words);
			output[idx].setParam(&model.olayer_linear);

			_lstm_concat[idx].init(opts.rnnhiddensize * 2, -1);
		}

		//lstm.init(&model.lstm_params, opts.dropOut, true, mem);
		lstm_left.init(&model.lstm_left_params, opts.dropOut, true);
		lstm_right.init(&model.lstm_right_params, opts.dropOut, false);

		for (int idx = 0; idx < maxsize; idx++){
			word_inputs[idx].init(opts.wordDim, opts.dropOut);
			output[idx].init(opts.labelSize, -1);
		}
	}


public:
	// some nodes may behave different during training and decode, for example, dropout
	inline void forward(const vector<Feature>& features, bool bTrain = false){
		//first step: clear value
//		clearValue(bTrain); // compute is a must step for train, predict and cost computation

		_pcg->train = bTrain;
		// second step: build graph
		int seq_size = features.size();
		//forward
		// word-level neural networks
		for (int idx = 0; idx < seq_size; idx++) {
			const Feature& feature = features[idx];
			//input
			word_inputs[idx].forward(_pcg, feature.words[0]);
		}
		lstm_left.forward(_pcg, getPNodes(word_inputs, seq_size));
		lstm_right.forward(_pcg, getPNodes(word_inputs, seq_size));

		for (int i = 0; i < seq_size; i++) {
			_lstm_concat[i].forward(_pcg, &lstm_left._hiddens[i], &lstm_right._hiddens[i]);
		}

		for (int idx = 0; idx < seq_size; idx++) {
			//feed-forward
			//output[idx].forward(_pcg, &lstm_left._hiddens[idx]);
			output[idx].forward(_pcg, &_lstm_concat[idx]);
		}
	}

};

#endif /* SRC_GraphBuilder_H_ */
