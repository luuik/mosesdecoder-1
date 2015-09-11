#ifndef MORPHTRIE_H_
#define MORPHTRIE_H_

#include <string>
#include <vector>
#include "Node.h"

using namespace std;

template<class KeyClass, class ValueClass>
class MorphTrie
{
    public:
        MorphTrie() : root(new Node<KeyClass, ValueClass>()) {}
        Node<KeyClass, ValueClass>* insert(const vector<KeyClass>& word, const ValueClass& weight, const ValueClass& backoff);
        Node<KeyClass, ValueClass>* getProb(const vector<KeyClass>& word);
        virtual ~MorphTrie();
    private:
        void deleteNode(Node<KeyClass, ValueClass>* node);
        Node<KeyClass, ValueClass>* root;
};

template<class KeyClass, class ValueClass>
Node<KeyClass, ValueClass>* MorphTrie<KeyClass, ValueClass>::insert(const vector<KeyClass>& word, 
																	const ValueClass& weight, 
																	const ValueClass& backoff)
{
    Node<KeyClass, ValueClass>* cRoot = root;
    for (size_t i = 0; i < word.size(); ++i)
    {
        KeyClass cKey = word[i];
        Node<KeyClass, ValueClass>* cNode = root->findSub(cKey);
        if (cNode == NULL)
        {
            cNode = new Node<KeyClass, ValueClass>(cKey);
            cNode->setProb(weight);
            cNode->setBackOff(backoff);
            cRoot->addSubnode(cNode);
        }
        cRoot = cNode;
    }
    return cRoot;
}

template<class KeyClass, class ValueClass>
Node<KeyClass, ValueClass>* MorphTrie<KeyClass, ValueClass>::getProb(const vector<KeyClass>& word)
{
    Node<KeyClass, ValueClass>* cRoot = root;
    for (size_t i = 0; i < word.size(); ++i)
    {
        KeyClass cKey = word[i];
        Node<KeyClass, ValueClass>* cNode = root->findSub(cKey);
        if (cNode == NULL)
        {
            return NULL;
        }

        cRoot = cNode;
    }
    return cRoot;
}

template<class KeyClass, class ValueClass>
void MorphTrie<KeyClass, ValueClass>::deleteNode(Node<KeyClass, ValueClass>* node)
{
    for (size_t i = 0; i < node->getSubnodes().size(); ++i)
    {
        deleteNode(node->getSubnodes()[i]);
    }
    delete node;
}

template<class KeyClass, class ValueClass>
MorphTrie<KeyClass, ValueClass>::~MorphTrie()
{
    deleteNode(root);
}
#endif /* end of include guard: MORPHTRIE_H_ */
