#include <iostream>
#include <vector>
#include <boost/foreach.hpp>
#include <boost/filesystem.hpp>
#include "moses/FF/JoinScore/Trie.h"
#include "moses/Util.h"

using namespace std;
using namespace Moses;

namespace Moses {
extern bool g_mosesDebug;
}

struct LMScores
{
	float prob, backoff;

	LMScores()
	{
		prob = 4343;
		backoff = 76;
	}
};

std::ostream& operator<<(std::ostream &out, const LMScores &obj)
{
	out << "(" << obj.prob << "," << obj.backoff << ")" << flush;
	return out;
}

map<string, uint64_t> vocab;

typedef vector<uint64_t> NGRAM;
typedef Trie<LMScores, NGRAM> MYTRIE;

int64_t GetVocabId(const string &str)
{
  int64_t ret;
	map<string, uint64_t>::iterator iter = vocab.find(str);

	if (iter == vocab.end()) {
		cerr << "Couldn't find in vocab:" << str << endl;
		abort();
	}

    ret = iter->second;
	return ret;
}

void ParseLine(NGRAM &ngram, LMScores &lmScores, const string &line)
{
	vector<string> toks = Tokenize(line);
	for (size_t i = 0; i < toks.size() - 2; ++i) {
		const string &tok = toks[i];
		int64_t vocabId = GetVocabId(tok);
		ngram.push_back(vocabId);
	}
	
	lmScores.prob = Scan<float>(toks[toks.size() - 2]);
	lmScores.backoff = Scan<float>(toks[toks.size() - 1]);
}

void Save(MYTRIE &trie, const string &inPath, const string &outPath)
{
	MYTRIE::Node &root = trie.m_root;

	MYTRIE::Parameters &params = trie.m_params;
	params.m_inPath = inPath;
	params.m_outPath = outPath;
	//trie.m_params.m_prefixV = ;
	
  InputFileStream inStrme(params.m_inPath);

  std::ofstream outStrme;
  outStrme.open(params.m_outPath.c_str(), std::ios::out | std::ios::in | std::ios::binary | std::ios::ate | std::ios::trunc);
  UTIL_THROW_IF(!outStrme.is_open(),
                util::FileOpenException,
                std::string("Couldn't open file ") + params.m_outPath);

  std::string line;
  size_t lineNum = 0;
  while (getline(inStrme, line)) {
    lineNum++;
    if (lineNum%100000 == 0) std::cerr << lineNum << " " << std::flush;
    //std::cerr << lineNum << " " << line << std::endl;

    if (line.empty() || line.size() > 150) {
      continue;
    }

  	NGRAM ngram;
	LMScores lmScores;
	ParseLine(ngram, lmScores, line);
	//cerr << "lmScores=" << lmScores << endl;
		
    trie.m_root.Save(ngram, outStrme, 0, params, lmScores);
  }

  // write root node;
  size_t numChildren = root.m_children.size();
  root.WriteToDisk(outStrme);
  std::cerr << "Saved root "
            << root.m_filePos << "="
            //<< m_value << "="
            << numChildren << std::endl;

  outStrme.write((char*)&root.m_filePos, sizeof(root.m_filePos));

  outStrme.close();
  inStrme.Close();

}
 
int main(int argc, char* argv[])
{
  cerr << "Starting..." << endl;

  string arpaPath = argv[1];
  string outDir = argv[2];

  string lmPath = outDir + "/lm.dat";
  string vocabPath = outDir + "/vocab.dat";
  string miscPath = outDir + "/misc.dat";

  // load vocab
  InputFileStream vocabStrme(vocabPath);
  string line;
  uint64_t vocabId = 1;
  while (getline(vocabStrme, line)) {
	  vocab[line] = vocabId;
	  //cerr << line << "=" << vocabId << endl;
	  ++vocabId;
  }

  // save trie
  MYTRIE trie;
  Save(trie, arpaPath, lmPath);

  ofstream miscStrme;
  miscStrme.open(miscPath.c_str());
  miscStrme << "Version 1" << endl;
  miscStrme.close();

  cerr << "Finished" << endl;
}
