#include <vector>
#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>
#include "moses/ScoreComponentCollection.h"
#include "moses/Hypothesis.h"
#include "moses/FactorCollection.h"
#include "util/exception.hh"
#include "moses/StaticData.h"
#include "MorphoSubWordLM.h"

using namespace std;

namespace Moses
{
////////////////////////////////////////////////////////////////
MorphoSubWordLM::MorphoSubWordLM(const std::string &line)
:MorphoLM(line)
{
}


const FFState* MorphoSubWordLM::EmptyHypothesisState(const InputType &input) const {
  std::vector<const Factor*> context;
  context.push_back(m_sentenceStart);
  std::vector<const Factor*>  thisIsTheStart;
  thisIsTheStart.push_back(m_sentenceStart);
  std::vector<std::vector<const Factor*> > contextSplit;
  contextSplit.push_back(thisIsTheStart);

  return new MorphoSubWordLMState(context, contextSplit, "", 0.0);
}

void MorphoSubWordLM::Load()
{
	boost::filesystem::path resolved = boost::filesystem::canonical(m_path);
	m_LM.load(resolved.string());
}

FFState* MorphoSubWordLM::EvaluateWhenApplied(
  const Hypothesis& cur_hypo,
  const FFState* prev_state,
  ScoreComponentCollection* accumulator) const
{
  // dense scores
  float score = 0;
  float ngramScore = 0.0;
  size_t targetLen = cur_hypo.GetCurrTargetPhrase().GetSize();

  assert(prev_state);

  const MorphoSubWordLMState *prevMorphState = static_cast<const MorphoSubWordLMState*>(prev_state);

  bool isUnfinished = prevMorphState->IsUnfinished();
  string unfinishedWord = prevMorphState->GetUnfinishedWord();
  float prevScore = prevMorphState->GetPrevScore();

  vector<const Factor*> context = prevMorphState->GetPhrase();
  std::vector<std::vector<const Factor*> > splitContext = prevMorphState->GetPhraseSplit();;

  //vector<string> stringContext;
  //SetContext(stringContext, prevMorphState->GetPhrase());
  FactorCollection &fc = FactorCollection::Instance();



  for (size_t pos = 0; pos < targetLen; ++pos){
	  const Word &word = cur_hypo.GetCurrWord(pos);
	  const Factor *factor = word[m_factorType];
	  string currStr = factor->GetString().as_string();
	  int prefixSuffix = GetMarker(factor->GetString());

	  const Factor *wordStem;
	  const Factor *wordPrefix;
	  const Factor *wordSuffix;

	  std::vector<const Factor*> factorSplits;

	  if (isUnfinished) {

		  switch (prefixSuffix) {
		  case 0:
			  // a+ b. Invalid. Start new word
			  wordStem = fc.AddFactor(currStr, false);
			  factorSplits.push_back(wordStem);
			  unfinishedWord = "";
			  factor = fc.AddFactor(currStr, false);
			  isUnfinished = false;
			  break;
		  case 1:
			  // a+ +b
			  wordPrefix = fc.AddFactor(unfinishedWord, false);
			  wordSuffix = fc.AddFactor(currStr, false);
			  factorSplits.push_back(wordPrefix);
			  factorSplits.push_back(wordSuffix);
        	  unfinishedWord += currStr;
              factor = fc.AddFactor(unfinishedWord, false);
              unfinishedWord = "";
              score -= prevScore;
              isUnfinished = false;
			  break;
		  case 2:
        	  // a+ b+. Invalid. Start new word
			  unfinishedWord = currStr;
			  wordPrefix = fc.AddFactor(unfinishedWord, false);
			  factorSplits.push_back(wordPrefix);
              factor = fc.AddFactor(currStr, false);
              isUnfinished = true;
			  break;
		  case 3:
			  // a+ +b+.
			  wordPrefix = fc.AddFactor(unfinishedWord, false);
			  wordStem = fc.AddFactor(currStr, false);
			  factorSplits.push_back(wordPrefix);
			  factorSplits.push_back(wordStem);
        	  unfinishedWord += currStr;
              factor = fc.AddFactor(unfinishedWord, false);
              score -= prevScore;
              isUnfinished = true;
			  break;
		  default:
			  abort();
		  }
      }
      else {
		  switch (prefixSuffix) {
		  case 0:
			  // a b
			  wordStem = fc.AddFactor(currStr, false);
			  factorSplits.push_back(wordStem);
              factor = fc.AddFactor(currStr, false);
        	  unfinishedWord = currStr;
              isUnfinished = false;
			  break;
		  case 1:
			  // a +b. Invalid. New word
			  wordSuffix = fc.AddFactor(currStr, false);
			  factorSplits.push_back(wordSuffix);
			  factor = fc.AddFactor(currStr, false);
              unfinishedWord = "";
              isUnfinished = false;
			  break;
		  case 2:
        	  // a b+. start new unfinished word
			  unfinishedWord = currStr;
			  wordPrefix = fc.AddFactor(unfinishedWord, false);
			  factorSplits.push_back(wordPrefix);
        	  factor = fc.AddFactor(unfinishedWord, false);
              isUnfinished = true;
              break;
		  case 3:
			  // a +b+. Invalid. Start new word
			  wordStem = fc.AddFactor(currStr, false);
			  factorSplits.push_back(wordStem);
        	  unfinishedWord = currStr;
        	  factor = fc.AddFactor(unfinishedWord, false);
              isUnfinished = true;
			  break;
		  default:
			  abort();
		  }
      }

	  context.push_back(factor);
	  splitContext.push_back(factorSplits);

	  // SCORE
	  if (context.size() > m_order) {
	    context.erase(context.begin());
	    splitContext.erase(splitContext.begin());
	  }

	  assert(context.size() == splitContext.size());
	  ngramScore = Score(splitContext);
      score += ngramScore;

      prevScore = ngramScore;

      //DebugContext(context);
      //cerr << " ngramScore=" << ngramScore << endl;

      if (isUnfinished) {
    	  context.resize(context.size() - 1);
    	  splitContext.resize(splitContext.size() - 1);
      }
  }

  // is it finished?
  if (cur_hypo.GetWordsBitmap().IsComplete()) {
      context.push_back(m_sentenceEnd);
      std::vector<const Factor*>  thisIsTheEnd;
      thisIsTheEnd.push_back(m_sentenceEnd);
      splitContext.push_back(thisIsTheEnd);
      if (context.size() > m_order) {
    	  context.erase(context.begin());
    	  splitContext.erase(splitContext.begin());
      }
      unfinishedWord = "";
      prevScore = 0;

      assert(context.size() == splitContext.size());
      ngramScore = Score(splitContext);

      //DebugContext(context);
      //cerr << "ngramScore=" << ngramScore << endl;

      score += ngramScore;
  }

  // finished scoring. set score
  accumulator->PlusEquals(this, score);

  // TODO: Subtract itermediate?
  if (context.size() >= m_order) {
	  context.erase(context.begin());
	  splitContext.erase(splitContext.begin());
  }
  //cerr << "unfinishedWord=" << unfinishedWord << endl;

  //std::vector<const Factor*>  context;
  //SetContext2(stringContext, context);

  assert(context.size() < m_order);
  assert(context.size() == splitContext.size());
  return new MorphoSubWordLMState(context, splitContext, unfinishedWord, prevScore);
}

size_t MorphoSubWordLM::GetContextOutcome(std::vector<std::vector<const Factor*> > &contextSplit, maxent::MaxentModel::context_type &MEcontext, maxent::MaxentModel::outcome_type &MEoutcome) const
{
	size_t modelOrder = contextSplit.size();
	cerr << "Size of contextSplit = " << contextSplit.size() << endl;
	for (size_t i=0; i<contextSplit.size(); i++) {

		size_t wordIndex = contextSplit.size()-1-i;
		cerr << "Size of contextSplit[" << i << "] =" << contextSplit[i].size() << endl;
		for (size_t k = 0; k < contextSplit[i].size(); k++) {
			//std::string predorder =	boost::lexical_cast<std::string>(wordIndex);
			std::string pred = "::"+contextSplit[i][k]->GetString().as_string();
			cerr << "FEAT: " << wordIndex << pred << " , " << endl;
			//MEcontext.push_back(make_pair(pred, 1));
		}
	}
	//MEoutcome
	//std::string label =	contextSplit.back().back()->GetString().as_string();
	//cerr << "LABEL: " << label << endl;
	return modelOrder;
}

float MorphoSubWordLM::DummyScore(std::vector<std::vector<const Factor*> > &contextSplit) const
{
	return -0.1;
}
float MorphoSubWordLM::Score(std::vector<std::vector<const Factor*> > &contextSplit) const
{
	maxent::MaxentModel::context_type mycontext;
	maxent::MaxentModel::outcome_type myoutcome;
	size_t modelOrder = GetContextOutcome(contextSplit, mycontext, myoutcome);

	//return static_cast<float>(m_LM.eval(mycontext,myoutcome));
	return DummyScore(contextSplit);
}

}
