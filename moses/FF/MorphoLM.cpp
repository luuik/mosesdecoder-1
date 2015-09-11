#include <vector>
#include "MorphoLM.h"
#include "moses/ScoreComponentCollection.h"
#include "moses/Hypothesis.h"
#include "moses/FactorCollection.h"
#include "moses/InputFileStream.h"
#include "util/exception.hh"

#include "moses/FF/JoinScore/TrieSearch.h"
#include "moses/FF/MorphoTrie/MorphTrie.h"

using namespace std;

namespace Moses
{
int MorphoLMState::Compare(const FFState& other) const
{
  const MorphoLMState &otherState = static_cast<const MorphoLMState&>(other);

  if (m_lastWords < otherState.m_lastWords) {
	  return -1;
  }
  else if (m_lastWords > otherState.m_lastWords) {
	  return +1;
  }

  // context words equal. Compare last unfinished word
  if (m_unfinishedWord < otherState.m_unfinishedWord) {
	  return -1;
  }
  else if (m_unfinishedWord > otherState.m_unfinishedWord) {
	  return +1;
  }

  return 0;
}

////////////////////////////////////////////////////////////////
MorphoLM::MorphoLM(const std::string &line)
:StatefulFeatureFunction(1, line)
,m_order(0)
,m_factorType(0)
,m_marker("+")
,m_binLM(false)
{
  ReadParameters();

  if (m_order == 0) {
	UTIL_THROW2("Must set order");
  }

  FactorCollection &fc = FactorCollection::Instance();
  m_sentenceStart = fc.AddFactor("<s>", false);
  m_sentenceEnd = fc.AddFactor("</s>", false);
}

const FFState* MorphoLM::EmptyHypothesisState(const InputType &input) const {
  std::vector<const Factor*> context;
  context.push_back(m_sentenceStart);

  return new MorphoLMState(context, "");
}

void MorphoLM::Load()
{
  if (m_binLM) {
	  m_trieSearch = new TrieSearch<LMScores, NGRAM>;
	  m_trieSearch->Create(m_path);

	  // vocab
	  FactorCollection &fc = FactorCollection::Instance();

	  m_vocab = new std::map<const Factor *, uint64_t>;

	  InputFileStream vocabStrme(m_path + ".vocab");
	  string line;
	  while (getline(vocabStrme, line)) {
		  vector<string> toks = Tokenize(line);
		  assert(toks.size() == 2);

		  const Factor *factor = fc.AddFactor(toks[0], false);
		  uint64_t vocabId = Scan<uint64_t>(toks[1]);
		  (*m_vocab)[factor] = vocabId;
	  }
  }
  else {
	  root = new MorphTrie<string, float>;
	  LoadLm(m_path, root);
  }
}

void MorphoLM::EvaluateInIsolation(const Phrase &source
    , const TargetPhrase &targetPhrase
    , ScoreComponentCollection &scoreBreakdown
    , ScoreComponentCollection &estimatedFutureScore) const
{}

void MorphoLM::EvaluateWithSourceContext(const InputType &input
    , const InputPath &inputPath
    , const TargetPhrase &targetPhrase
    , const StackVec *stackVec
    , ScoreComponentCollection &scoreBreakdown
    , ScoreComponentCollection *estimatedFutureScore) const
{}

void MorphoLM::EvaluateTranslationOptionListWithSourceContext(const InputType &input
    , const TranslationOptionList &translationOptionList) const
{}

FFState* MorphoLM::EvaluateWhenApplied(
  const Hypothesis& cur_hypo,
  const FFState* prev_state,
  ScoreComponentCollection* accumulator) const
{
  // dense scores
  float score = 0;
  float bad_score = -10000000000000.0;
  size_t targetLen = cur_hypo.GetCurrTargetPhrase().GetSize();
  const WordsRange &targetRange = cur_hypo.GetCurrTargetWordsRange();

  assert(prev_state);

  const MorphoLMState *prevMorphState = static_cast<const MorphoLMState*>(prev_state);

  bool prevIsMorph = prevMorphState->GetPrevIsMorph();
  string prevMorph = prevMorphState->GetPrevMorph();
  bool wasBad = false;
  std::vector<const Factor*> context(prevMorphState->GetPhrase());

  vector<string> stringContext;
  if (!prevIsMorph)
    stringContext.push_back(prevMorph);
          
  for (size_t pos = targetRange.GetStartPos(); pos < targetLen; ++pos){
	  const Word &word = cur_hypo.GetWord(pos);
	  const Factor *factor = word[m_factorType];
	  string str = factor->GetString().as_string();
        if (str[0] == '+' && prevIsMorph == true) {
            string prevClear = context.back()->GetString().as_string();
            str.erase(str.begin());
            str = prevClear + str;
        }
        else if (str[0] == '+' && prevIsMorph == false) {
          str.erase(str.begin()); //Get rid of starting +
          score += bad_score;
          wasBad = true;
        }
        else if (str[0] != '+' && prevIsMorph == true) {
          score += bad_score;
          wasBad = true;
        }
        else {
          //Yay! Easy ... just words
        }
        
        if (str[str.length() - 1] == '+') {
            str.erase(str.end() - 1);
            prevIsMorph = true;
            prevMorph = str;
            //TODO estimate score of the rest
        }
        else {
          prevIsMorph = false;
          prevMorph = "";
           // TODO: Subtract itermediate?
        }
    
    // If the current hypotheis is null, ignore it (just a +, start of this method, etc.)
    if (str.length() > 0) {
      stringContext.push_back(str);
    }
    if (!wasBad) {
      score += KneserNey(stringContext);
    }
    wasBad = false;

    //TODO: Subtract 

  }

  // finished scoring. set score
  accumulator->PlusEquals(this, score);

  // TODO: Subtract itermediate?

  return new MorphoLMState(context, prevMorph);
}

FFState* MorphoLM::EvaluateWhenApplied(
  const ChartHypothesis& /* cur_hypo */,
  int /* featureID - used to index the state in the previous hypotheses */,
  ScoreComponentCollection* accumulator) const
{
  abort();
  return NULL;
}

float MorphoLM::KneserNey(std::vector<string>& context) const
{
  float oov = -10000000000.0;
  float backoff = 0.0;

  Node<string, float>* result = root->getProb(context);


  if (result) 
    return result->getProb();
  if (context.size() > 1) {
    std::vector<string> backOffContext(context.begin(), context.end() - 1);
    
    result = root->getProb(backOffContext);
    if (result)
      backoff = result->getBackOff();

    context.erase(context.begin());

    return (backoff + MorphoLM::KneserNey(context));
  }
  
  return oov;
}

void MorphoLM::SetParameter(const std::string& key, const std::string& value)
{
  if (key == "path") {
    m_path = value;
  }
  else if (key == "order") {
    m_order = Scan<size_t>(value);
  }
  else if (key == "factor") {
	m_factorType = Scan<FactorType>(value);
  }
  else if (key == "marker") {
	  m_marker = value;
  }
  else if (key == "binary") {
	  m_binLM = Scan<bool>(value);
  }
  else {
    StatefulFeatureFunction::SetParameter(key, value);
  }
}

}

