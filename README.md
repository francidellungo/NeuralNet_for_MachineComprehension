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

## Dependencies
* tensorflow-gpu 2.0.0
* gensim 3.8.0
* numpy 1.18.1

Dependencies can be installed with:
`pip install -r requirements.txt`

Create new directory: `mkdir glove` \
Get glove pretrained: `wget https://nlp.stanford.edu/data/glove.6B.zip` \
And extract it: `unzip glove.6B.zip -d ./glove`


## References
<a id="TriviaQA">[1]</a> 
Joshi, Mandar and Choi, Eunsol and Weld, Daniel S. and Zettlemoyer, Luke (2017). 
**TriviaQA: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension**.
 Association for Computational Linguistics (ACL). Vancouver, Canada.
 
 <a id="RNN">[2]</a> 
Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi and Hananneh Hajishirzi (2017). **Bidirectional Attention Flow for Machine Comprehension**.
CoRR.

 <a id="...">[3]</a> 
...

- glove paper: https://nlp.stanford.edu/pubs/glove.pdf
