#pragma once

#include "moses/FF/MorphoLM.h"
#include "maxentmodel.hpp"

namespace Moses
{

class MorphoSubWordLMState : public MorphoLMState
{
	std::vector<std::vector<const Factor*> > m_lastWordSplits;

public:
  MorphoSubWordLMState(const std::vector<const Factor*> &context,
		  const std::vector<std::vector<const Factor*> > &contextSplits,
		  	  const std::string &unfinished, float prevScore)
    :	MorphoLMState(context,unfinished,prevScore)
    	,m_lastWordSplits(contextSplits)
  {
  }

  const std::vector<std::vector<const Factor*> > &GetPhraseSplit() const
   { return m_lastWordSplits; }

};

class MorphoSubWordLM : public MorphoLM
{
protected:
	maxent::MaxentModel m_LM;
	size_t GetContextOutcome(std::vector<std::vector<const Factor*> > &contextSplit, maxent::MaxentModel::context_type &MEcontext, maxent::MaxentModel::outcome_type &MEoutcome) const;
	float Score(std::vector<std::vector<const Factor*> > &contextSplit) const;
public:
	MorphoSubWordLM(const std::string &line);

	bool IsUseable(const FactorMask &mask) const {
		return true;
	}

	virtual const FFState* EmptyHypothesisState(const InputType &input) const;

	virtual void Load();
	FFState* EvaluateWhenApplied(
    const Hypothesis& cur_hypo,
    const FFState* prev_state,
    ScoreComponentCollection* accumulator) const;


};


}

