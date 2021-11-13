# equal.txt

---

---

## 1. First Reflection

### Title: Equal.txt

### Who
**amukhe12**: Adwith Mukherjee &&
**kteramot**: Kira Clarke

### Introduction
Representations on the screen reflect and amplify the problems we face in the broader societal landscape. This is also the case in terms of film reviews — according to a [study](https://womenintvfilm.sdsu.edu/wp-content/uploads/2020/08/2020-Thumbs-Down-Report.pdf) from the  San Diego State University’s Center for the Study of Women Film and Television, men comprise 65%, and women 35%, of all film reviewers . 
 
This got us to thinking: If representations of women in film are already skewed, how much more bias might be detected in movie reviews that already reflect those imbalances… if most are written by males? 
 
With this project, we are attempting to identify gender biases written into movie reviews by looking at a substantial corpus. This is a classification deep learning project, because correlations are established for the given input through word embeddings. We hope to explore, by looking at the vector embeddings, what kind of language is associated with each gender. 


### Related Work
Although not NLP, we were inspired by a related work that can be found here: "[Using Deep Learning to Analyse Movie Posters for Gender Bias](https://medium.com/analytics-vidhya/using-deep-learning-to-analyse-movie-posters-for-gender-bias-4c0f1557a051)". These researchers ran these posters through 2 AI models for facial recognition to identify the gender of characters on the posters. Using a dataset of 21,000 movie posters and metadata (IMDb scores, movie credits), they found that while more women have become featured as the years have progressed, most posters predominantly feature the male characters. 
We were of course influenced by the 
"[Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings](https://proceedings.neurips.cc/paper/2016/file/a486cd07e4ac3d270571622f4f316ec5-Paper.pdf)"(2016) paper on which the debiasing lab was based. From this, we learned that word embeddings learn covertly and overtly coded “gendered” meanings of terms, but that debiasing these were possible. 
The paper for which the dataset we are using was "[Learning Word Vectors for Sentiment Analysis](http://ai.stanford.edu/~amaas/papers/wvSent_acl2011.pdf)" (2011), and it used movie reviews because the polarity of the reviews, as well as their abundance, provides a robust benchmark for sentiment analysis.


- Ghosh, R. (2020, May 27). Using deep learning to analyse movie posters for gender bias. Medium. Retrieved November 12, 2021, from https://medium.com/analytics-vidhya/using-deep-learning-to-analyse-movie-posters-for-gender-bias-4c0f1557a051. 

- Tolga Bolukbasi, Kai-Wei Chang, James Zou, Venkatesh Saligrama, and Adam Kalai. 2016. "[Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings](https://proceedings.neurips.cc/paper/2016/file/a486cd07e4ac3d270571622f4f316ec5-Paper.pdf)". In Proceedings of the 30th International Conference on Neural Information Processing Systems (NIPS'16). Curran Associates Inc., Red Hook, NY, USA, 4356–4364.

- Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. (2011). [Learning Word Vectors for Sentiment Analysis](http://ai.stanford.edu/~amaas/papers/wvSent_acl2011.pdf). The 49th Annual Meeting of the Association for Computational Linguistics (ACL 2011).



### Data
We have found a Large Movie Review Dataset that was compiled by researchers at Stanford University. It was initially used for a sentiment analysis study, linked below. It pointed out that many unsupervised models are unable to “capture sentiment information that is central to many word meanings and important for a wide range of NLP tasks.” They used the data to create a model that performs sentiment classification to capture relational structures of language. As such, the dataset is for binary sentiment classification containing a set of 25,000 movie reviews for training, and 25,000 for testing.
In terms of preprocessing the txt files, it should be fairly simple. We will have to remove punctuation in the process, but we have an understanding of how to alter them to the word-embeddings through classwork. The main challenge and feature of exploration will be the debiasing.
The dataset can be easily downloaded at this dataset [link](http://ai.stanford.edu/~amaas/data/sentiment/).

- Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. (2011). [Learning Word Vectors for Sentiment Analysis](http://ai.stanford.edu/~amaas/papers/wvSent_acl2011.pdf). The 49th Annual Meeting of the Association for Computational Linguistics (ACL 2011).



### Methodology
#### Inputs
We plan to tokenize each word in the corpus of movie review data. Then we will create skipgrams of the whole corpus with a window size of 3 to start. 

#### Training initial models
We plan to train two models to start. The first is a Continuous Bag of Words model that predicts a target word from a number of context words. For this implementation, the input to the model is the sum of the vector representations of all other words in the window apart from the target center word, which acts as the label. The Skipgram model will work similarly but will take in individual skipgram inputs. 
The model architecture for each model will look similar, with an embedding matrix of size 64 and two dense layers which will return logits over the entire vocab size. We will use sparse softmax cross entropy loss to evaluate the function, alongside an Adam optimizer with a learning rate of 0.001 (subject to change)

#### Debiasing 
We then plan to evaluate the gender-based bias of each model by querying the learned vector embeddings of different words. First, we'll create an equality set of gendered words ("man", "woman"), and compare the cosine similarities of these words against words with potentially negative gender-based connotations. We can also use these gendered words to create a gender subspace that will be used to debias the model. 

#### Stretch: Training deeper contextual models
Once we have built the two models and debiased them, we plan to implement a bidirectional language model similar to the one implemented in [this paper](https://arxiv.org/pdf/1802.05365.pdf). This is a combination of a forward LM, which passes past tokens to an LSTM to predict the next token, as well as a backward LM, which passes future tokens to an LSTM to predict the current token. Combined, the two use the power of LSTMs take an input of some context words (+/- window size) to predict the center target word. The result is a deeper context-based representation of words that will better capture any bias present in the corpus.  We can then attempt to debias this model as mentioned above. 


### Metrics
We aim to have both qualitative and quantitative metrics. Qualitatively, we want to observe the associations between the words and see how certain connotations can be neutralized after debiasing. Quantitatively, we want to calculate the ​​variable cosine similarities between the biased words and the gender-paired words (utilizing the gender subspace). First, we hope to identify some examples and analyze the patterns. Next, we hope to use a novel debiasing model (we will experiment with the architecture, as indicated above), and record the differences, attempting through various modifications to progressively minimize the cosine similarities. Success entails both a thorough assessment of the initial results and a gradual amelioration of the neutralization process of the biased words. 
> Goal
1.  **Base Goal**: Analyze skipgram model using Stanford Movie Review dataset
2.  **Target Goal**: Analyze skipgram and CBOW model using Stanford Movie Review Dataset and attempt to debias model.
3.  **Stretch Goal**: Using another Movie Review dataset to further test the correlations between gender and polarity of the movie reviews.

### Ethics
Deep learning is an effective way to assess this problem, because it would be a time-consuming process if it were approached manually. To reveal the gender biases that persist through tens of thousands of reviews is difficult without the ability to quantify the subtle and apparent differences in word associations. Then again, in so doing, we are resorting to using a benchmark of very binary boundaries of gender. This presents limitations, but is nonetheless a start to exploring a way to monitor our own habits within the context and confines of language, as well as considerations we have to make in modifying models to account for the existence of bias in the datasets we feed them. 
With deep neural networks, the word embeddings reflect back to us the patterns which we have become accustomed to, and are therefore less immediately evident to us. A broader societal issue that is relevant to this is the lopsided statistics of gender in writing. Over 70% of best-selling fiction books are written by men, and, as we discussed above, a majority of movie critics are male. This is not just a matter of numbers, but about who is allowed to command the respectabiltiy or stature that affords importance in this sphere. We do not speculate that the use of less biased language falls along gender lines; However, a more diverse group of scribes could only enrich and broaden the scope of language, even potentially resolving some of these issues. In assessing movie reviews, we are also considering the subject that this corpus discusses; Popular movies disproportionately highlight certain experiences and glorify characters with particular traits. It is no wonder that texts discussing these films relay the same (sexist, or other dubious) characterizations the films exhibit. As we wrote previously, this assessment of patterns begins with a binary gender framework, but can reach beyond to reveal further biases that we may have become blind to, and prevent their amplification through machine learning models.

### Division of Labor
Since we have a small team, a lot of the work will be shared. We intend to work together on preprocessing, word2vec, skipgrams, and the debiasing, contributing in ways that are tailored to each of our strengths and weaknesses, respectively. We will also both work on the video, poster, and write-ups, alternating between the initial drafting and polishing.
We expect the collaboration to be successful – both of us will contribute an equal amount of time to this project.


