#include <vector>
#include "MorphoSubWordLM.h"
#include "moses/ScoreComponentCollection.h"
#include "moses/Hypothesis.h"
#include "moses/FactorCollection.h"
#include "util/exception.hh"

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

  std::vector<std::vector<const Factor*> > contextSplit;

  return new MorphoSubWordLMState(context, contextSplit, "", 0.0);
}

void MorphoSubWordLM::Load()
{
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

  std::vector<const Factor*> factorSplits;

  for (size_t pos = 0; pos < targetLen; ++pos){
	  const Word &word = cur_hypo.GetCurrWord(pos);
	  const Factor *factor = word[m_factorType];
	  string currStr = factor->GetString().as_string();
	  int prefixSuffix = GetMarker(factor->GetString());

	  if (prefixSuffix & 1) {
	      currStr.erase(currStr.begin());
	  }
	  if (prefixSuffix & 2) {
          currStr.erase(currStr.end() - 1);
	  }

	  const Factor *wordStem;
	  const Factor *wordPrefix;
	  const Factor *wordSuffix;

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
			  wordPrefix = fc.AddFactor(unfinishedWord, false);
			  factorSplits.push_back(wordPrefix);
        	  unfinishedWord = currStr;
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
			  factorSplits.push_back(wordStem);
			  factor = fc.AddFactor(currStr, false);
              unfinishedWord = "";
              isUnfinished = false;
			  break;
		  case 2:
        	  // a b+. start new unfinished word
			  wordPrefix = fc.AddFactor(unfinishedWord, false);
			  factorSplits.push_back(wordPrefix);
			  unfinishedWord = currStr;
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
	  }

	  ngramScore = Score(splitContext);
      score += ngramScore;

      prevScore = ngramScore;

      //DebugContext(context);
      //cerr << " ngramScore=" << ngramScore << endl;

      if (isUnfinished) {
    	  context.resize(context.size() - 1);
      }
  }

  // is it finished?
  if (cur_hypo.GetWordsBitmap().IsComplete()) {
      context.push_back(m_sentenceEnd);
      if (context.size() > m_order) {
    	  context.erase(context.begin());
      }
      unfinishedWord = "";
      prevScore = 0;

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
  }
  //cerr << "unfinishedWord=" << unfinishedWord << endl;

  //std::vector<const Factor*>  context;
  //SetContext2(stringContext, context);

  assert(context.size() < m_order);
  return new MorphoSubWordLMState(context, splitContext, unfinishedWord, prevScore);
}

float MorphoSubWordLM::Score(std::vector<std::vector<const Factor*> > contextSplit) const
{
	return 1.0;

}

}
