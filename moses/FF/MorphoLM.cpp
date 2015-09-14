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

  return new MorphoLMState(context, "", 0.0);
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
	  FactorCollection &fc = FactorCollection::Instance();
	  root = new MorphTrie<const Factor*, LMScores>;

	  InputFileStream infile(m_path);
	  size_t lineNum = 0;
	  string line;
	  while (getline(infile, line)) {
		  if (++lineNum % 10000 == 0) {
			  cerr << lineNum << " ";
		  }

		  vector<string> substrings;
		  Tokenize(substrings, line, "\t");

		  if (substrings.size() < 2)
			   continue;

		  assert(substrings.size() == 2 || substrings.size() == 3);

		  float prob = Moses::Scan<float>(substrings[0]);
		  if (substrings[1] == "<unk>") {
			  m_oov = prob;
			  continue;
		  }

		  float backoff = 0.f;
		  if (substrings.size() == 3)
			backoff = Moses::Scan<float>(substrings[2]);

		  // ngram
		  vector<string> key;
		  Tokenize(key, substrings[1], " ");

		  vector<const Factor*> factorKey;
		  for (int i = 0; i < key.size(); ++i)
			  factorKey.push_back(fc.AddFactor(key[i], false));

		  root->insert(factorKey, LMScores(prob, backoff));
	  }
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
  float ngramScore = 0.0;
  size_t targetLen = cur_hypo.GetCurrTargetPhrase().GetSize();

  assert(prev_state);

  const MorphoLMState *prevMorphState = static_cast<const MorphoLMState*>(prev_state);

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

	  ngramScore = Score(context);
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

      ngramScore = Score(context);

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
  return new MorphoLMState(context, unfinishedWord, prevScore);
}

void MorphoLM::DebugContext(const vector<const Factor*> &context) const
{
    cerr << "CONTEXT:";
    for (size_t i = 0; i < context.size(); ++i) {
  	  cerr << context[i]->GetString() << " ";
    }
}

FFState* MorphoLM::EvaluateWhenApplied(
  const ChartHypothesis& /* cur_hypo */,
  int /* featureID - used to index the state in the previous hypotheses */,
  ScoreComponentCollection* accumulator) const
{
  abort();
  return NULL;
}

float MorphoLM::Score(std::vector<const Factor*> context) const
{
  assert(context.size() <= m_order);

  float backoff = 0.0;

  Node<const Factor*, LMScores>* result = root->getNode(context);

  float ret = -99999;
  if (result) {
	ret = result->getValue().prob;
  }
  else if (context.size() > 1) {
	  std::vector<const Factor*> backOffContext(context.begin(), context.end() - 1);
	  result = root->getNode(backOffContext);
	  if (result) {
		  backoff = result->getValue().backoff;
	  }

	  context.erase(context.begin());

	ret = backoff + Score(context);
  }
  else {
	  assert(context.size() == 1);
	  ret = m_oov;
  }
  
  return ret;
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

int MorphoLM::GetMarker(const StringPiece &str) const
{
  int ret = 0;

  if (str.size() > 1) {
    if (str.starts_with("+")) {
      ret += 1;
    }
    if (str.ends_with("+")) {
      ret += 2;
    }
  }

  return ret;
}

}

