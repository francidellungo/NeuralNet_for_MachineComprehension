# Neural Networks for Machine Reading Comprehension
Machine Comprehension / Machine Reading Comprehension models enable computers
to read a document and answer general questions against it.
 While this is a relatively elementary task for a human,
  it's not that straightforward for AI models. 

[Interactive demo](http://allgood.cs.washington.edu:1995/) by the authors of  the paper [[2]](#RNN).
  
## Dataset
Dataset used [[1]](#TriviaQA). \
Create new directory: `mkdir dataset`\
The data can be downloaded from the [TriviaQA website](http://nlp.cs.washington.edu/triviaqa/) or 
with: `wget https://nlp.cs.washington.edu/triviaqa/data/triviaqa-rc.tar.gz`

and extract with: `tar -xf triviaqa-rc.tar.gz -C dataset`


### SQuAD dataset (1.1 version)
 create new dir : `mkdir dataset/squad`\
 download the data with:  `wget https://www.wolframcloud.com/objects/6b06e230-f56a-4244-8f23-382e74440a15` \
 oppure (meglio, ma riguarda path dataset nel codice): \
 train `wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json -O dataset/squad/train-v1.1.json` \
 dev `wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json -O dataset/squad/dev-v1.1.json`
 
 
## Dependencies
* tensorflow-gpu 2.0.1
* gensim 3.8.0
* numpy 1.18.1

Dependencies can be installed with:
`pip install -r requirements.txt`

Create new directory: `mkdir glove` \
Get glove pretrained: `wget https://nlp.stanford.edu/data/glove.6B.zip` \
And extract it: `unzip glove.6B.zip -d ./glove`


## References
<a id="TriviaQA">[1]</a> 
Mandar Joshi, Eunsol Choi, Daniel S. Weld, Luke Zettlemoyer (2017). 
**TriviaQA: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension**.
 Association for Computational Linguistics (ACL). Vancouver, Canada.
 
 <a id="RNN">[2]</a> 
Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, Hananneh Hajishirzi (2017). **Bidirectional Attention Flow for Machine Comprehension**.
CoRR.

 <a id="GloVe">[3]</a> 
Jeffrey Pennington,  Richard Socher, Christopher D. Manning (2014)
**GloVe: Global Vectors for Word Representation**.
Empirical Methods in Natural Language Processing (EMNLP).

 <a id="high">[4]</a> 
 Rupesh Kumar Srivastava, Klaus Greff, Jürgen Schmidhuber (2015).
**Highway Networks**. CoRR.

 <a id="squad">[5]</a> 
 Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev and Percy Liang (2016)
 **SQuAD: 100,000+ Questions for Machine Comprehension of Text**.  Empirical Methods in Natural Language Processing (EMNLP).