#pragma once

#include <cassert>
#include <string>
#include "StatefulFeatureFunction.h"
#include "FFState.h"
#include "moses/FF/JoinScore/TrieSearch.h"
#include "moses/FF/MorphoTrie/utils.h"

namespace Moses
{
struct LMScores
{
	LMScores()
	{}

	LMScores(const LMScores &copy)
	:prob(copy.prob)
	,backoff(copy.backoff)
	{}

	LMScores(float inProb, float inBackoff)
	:prob(inProb)
	,backoff(inBackoff)
	{}

	float prob, backoff;
};

inline std::ostream& operator<<(std::ostream &out, const LMScores &obj)
{
	out << "(" << obj.prob << "," << obj.backoff << ")" << flush;
	return out;
}

class MorphoLMState : public FFState
{
  std::vector<const Factor*> m_lastWords;
  std::string m_unfinishedWord;
  float m_prevScore;
public:
  MorphoLMState(const std::vector<const Factor*> &context,
		  	  const std::string &unfinished, float prevScore)
    :m_lastWords(context)
  	,m_unfinishedWord(unfinished)
    ,m_prevScore(prevScore)
  {
  }

  int Compare(const FFState& other) const;
  FFState *Clone() const {
    abort();
    return NULL;
  }

  const std::vector<const Factor*> &GetPhrase() const
  { return m_lastWords; }

  bool IsUnfinished() const
  { return !m_unfinishedWord.empty(); }

  const std::string &GetUnfinishedWord() const
  {
	  return m_unfinishedWord;
  }

  float GetPrevScore() const
  { return m_prevScore; }

  void SetPrevScore(float score)
  {
      m_prevScore = score;
  }

};

class MorphoLM : public StatefulFeatureFunction
{
protected:
	std::string m_path;
	size_t m_order;
    FactorType	m_factorType;
    std::string m_marker;
    bool m_binLM;
    float m_oov;
    const Factor *m_sentenceStart, *m_sentenceEnd; //! Contains factors which represents the beging and end words for this LM.

    // binary trie
    std::map<const Factor*, uint64_t> *m_vocab;
    std::map<uint64_t, const Factor*> *m_vocabRev; // for debugging
    uint64_t m_oovId;

    typedef std::vector<uint64_t> NGRAM;
    TrieSearch<LMScores, NGRAM> *m_trieSearch;


    // in-mem trie
    MorphTrie<const Factor*, LMScores> *root;

    float Score(std::vector<const Factor*> context) const;
    float ScoreMem(std::vector<const Factor*> context) const;
    float ScoreBin(std::vector<uint64_t> context) const;

    uint64_t GetBinVocabId(const Factor *factor) const;
    const Factor *GetFactorFromVocabId(uint64_t vocabId) const;

    // +a = 1. a+ = 2. +a+ = 0
    int GetMarker(const StringPiece &str) const;

    void DebugContext(const vector<const Factor*> &context) const;
    void DebugContext(const vector<uint64_t> &context) const;
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

  void SetParameter(const std::string& key, const std::string& value);

};


}

