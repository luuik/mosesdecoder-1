#pragma once

#include <string>
#include "StatefulFeatureFunction.h"
#include "FFState.h"
#include "moses/FF/JoinScore/TrieSearch.h"
#include "moses/FF/MorphoTrie/utils.h"

namespace Moses
{
struct LMScores
{
	float prob, backoff;
};

class MorphoLMState : public FFState
{
  std::vector<const Factor*> m_lastWords;
  std::string m_unfinishedWord;
public:

  MorphoLMState(const std::vector<const Factor*> &context,
		  	  const std::string &unfinishedWord)
    :m_lastWords(context)
  	,m_unfinishedWord(unfinishedWord)
  {
  }

  int Compare(const FFState& other) const;
  FFState *Clone() const {
    abort();
    return NULL;
  }

  const std::vector<const Factor*> &GetPhrase() const
  { return m_lastWords; }

  bool GetPrevIsMorph() const
  { return !m_unfinishedWord.empty(); }

  const std::string &GetPrevMorph() const
  { return m_unfinishedWord; }
};

class MorphoLM : public StatefulFeatureFunction
{
protected:
	std::string m_path;
	size_t m_order;
    FactorType	m_factorType;
    std::string m_marker;
    bool m_binLM;

    // binary trie
    std::map<const Factor*, uint64_t> *m_vocab;

    typedef std::vector<uint64_t> NGRAM;
    TrieSearch<LMScores, NGRAM> *m_trieSearch;

    // in-mem trie
    MorphTrie<string, float>* root;

    const Factor *m_sentenceStart, *m_sentenceEnd; //! Contains factors which represents the beging and end words for this LM.

public:
  MorphoLM(const std::string &line);

  bool IsUseable(const FactorMask &mask) const {
    return true;
  }

  virtual const FFState* EmptyHypothesisState(const InputType &input) const;

  virtual void Load();

  void EvaluateInIsolation(const Phrase &source
                           , const TargetPhrase &targetPhrase
                           , ScoreComponentCollection &scoreBreakdown
                           , ScoreComponentCollection &estimatedFutureScore) const;
  void EvaluateWithSourceContext(const InputType &input
                                 , const InputPath &inputPath
                                 , const TargetPhrase &targetPhrase
                                 , const StackVec *stackVec
                                 , ScoreComponentCollection &scoreBreakdown
                                 , ScoreComponentCollection *estimatedFutureScore = NULL) const;

  void EvaluateTranslationOptionListWithSourceContext(const InputType &input
      , const TranslationOptionList &translationOptionList) const;

  FFState* EvaluateWhenApplied(
    const Hypothesis& cur_hypo,
    const FFState* prev_state,
    ScoreComponentCollection* accumulator) const;
  FFState* EvaluateWhenApplied(
    const ChartHypothesis& /* cur_hypo */,
    int /* featureID - used to index the state in the previous hypotheses */,
    ScoreComponentCollection* accumulator) const;

  float KneserNey(std::vector<string>& context) const;

  void SetParameter(const std::string& key, const std::string& value);

};


}

