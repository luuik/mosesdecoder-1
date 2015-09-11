#include "MorphTrie.h"
#include <fstream>
#include <ostream>
#include <string>
#include <vector>
#include "moses/Util.h"
#include "moses/Factor.h"

using namespace std;

inline void ParseLineByChar(string& line, char c, vector<string>& substrings) {
    size_t i = 0;
    size_t j = line.find(c);

    while (j != string::npos) {
        substrings.push_back(line.substr(i, j-i));
        i = ++j;
        j = line.find(c, j);

        if (j == string::npos)
            substrings.push_back(line.substr(i, line.length()));
    }
}

inline void LoadLm(string lmPath, MorphTrie<string, float>* root) {
  ifstream infile(lmPath.c_str());
    string line;
    while (getline(infile, line)) {
        vector<string> substrings;
        ParseLineByChar(line, '\t', substrings);

        if (substrings.size() < 2)
               continue;

        float weight = Moses::Scan<float>(substrings[0]);

        vector<string> key;
        ParseLineByChar(substrings[1], ' ', key);

        float backoff = 0.f;
        if (substrings.size() == 3)
            backoff = Moses::Scan<float>(substrings[2]);


       //reverse(key.begin(), key.end());
       root->insert(key, weight, backoff);
    }

}
