#include <iostream>
#include <vector>
#include <boost/foreach.hpp>
#include "moses/FF/JoinScore/Trie.h"
#include "moses/Util.h"

using namespace std;
using namespace Moses;

struct LMScores
{
	float prob, backoff;
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
		ret = vocab.size() + 1;
		vocab[str] = ret;
	}
	else {
	  ret = iter->second;
	}
	
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

  MYTRIE trie;
  Save(trie, argv[1], argv[2]);

  // save vocab
  string vocabPath = string(argv[2]) + ".vocab";
  ofstream vocabStrme;
  vocabStrme.open(vocabPath.c_str());

/*
  BOOST_FOREACH(const map<string, uint64_t>::value_type &e, vocab) {
	  cerr << e.first << " " << e.second << endl;
  }
*/
  map<string, uint64_t>::const_iterator iter;
  for (iter = vocab.begin(); iter != vocab.end(); ++iter) {
	  vocabStrme << iter->first << " " << iter->second << endl;
  }
  vocabStrme.close();

  cerr << "Finished" << endl;
}
