# equal.txt

---
### [Colab Notebook link](https://colab.research.google.com/drive/1bvk_XejjAHMtGi0ccrIF1s_v6V2aoAL1?usp=sharing)

---

##  Final Reflection

### Title: Equal.txt

### Who
**amukhe12**: Adwith Mukherjee &&
**kteramot**: Kira Clarke


### Introduction
Representations on the screen reflect and amplify the problems we face in the broader societal landscape. This is also the case in terms of film reviews — according to a [study](https://womenintvfilm.sdsu.edu/wp-content/uploads/2020/08/2020-Thumbs-Down-Report.pdf) from the  San Diego State University’s Center for the Study of Women Film and Television, men comprise 65%, and women 35%, of all film reviewers . 
This got us to thinking: If representations of women in film are already skewed, how much more bias might be detected in movie reviews that already reflect those imbalances… if most are written by males? 
 
With this project, we are attempting to identify gender biases written into movie reviews by looking at a substantial corpus. This is a classification deep learning project, because correlations are established for the given input through word embeddings. We hope to explore, by looking at the vector embeddings, what kind of language is associated with each gender. 

Although not NLP, we were inspired by a related work that can be found here: "[Using Deep Learning to Analyse Movie Posters for Gender Bias](https://medium.com/analytics-vidhya/using-deep-learning-to-analyse-movie-posters-for-gender-bias-4c0f1557a051)". These researchers ran these posters through 2 AI models for facial recognition to identify the gender of characters on the posters. Using a dataset of 21,000 movie posters and metadata (IMDb scores, movie credits), they found that while more women have become featured as the years have progressed, most posters predominantly feature the male characters. 
We were of course influenced by the 
"[Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings](https://proceedings.neurips.cc/paper/2016/file/a486cd07e4ac3d270571622f4f316ec5-Paper.pdf)"(2016) paper on which the debiasing lab was based. From this, we learned that word embeddings learn covertly and overtly coded “gendered” meanings of terms, but that debiasing these were possible. 
The paper for which the dataset we are using was "[Learning Word Vectors for Sentiment Analysis](http://ai.stanford.edu/~amaas/papers/wvSent_acl2011.pdf)" (2011), and it used movie reviews because the polarity of the reviews, as well as their abundance, provides a robust benchmark for sentiment analysis.

### Methodology

#### Preprocessing
For the data, we found a Large Movie Review Dataset*compiled by researchers at Stanford University. The dataset is for binary sentiment classification containing a set of 25,000 movie reviews for training, and 25,000 for testing. In terms of preprocessing the txt files, we removed punctuation, standardized case, stemmed words, and substituted uncommon words with an <UNK> token to limit the model’s vocab size. This was all done with the nltk library in Python. Then, each word was tokenized and a word2id vocabulary map constructed. 

#### Model 1: word2vec with Skipgrams

The first model trained was a simple word2vec diagram that used skipgrams with a window size of two and an embedding size of 20. The model architecture included an embedding layer and an output dense layer. 

#### Model 2: Vanilla LSTM

The second model trained used a basic LSTM architecture. Given a sequence of window size 20, the model attempted to predict the next word. The model was composed of an embedding layer (embedding size 100), a LSTM layer, and a single dense output layer.

#### Model 3: Bidirectional LSTM (context2vec)

The third model used two LSTM layers operating in opposite directions to infer a central context word. This architecture, dubbed context2vec, is drawn from this paper**. Using a window size of 31, the first 15 words are passed to the left-to-right LSTM while the last 15 words are passed to the right-to-left LSTM. Their outputs are then concatenated and passed to a multi-layer perceptron to infer the central word. This modified architecture is valuable because more context is provided for each central word, and the meaning of a word is often dependent on words before and after it. This pair of LSTMs reading in opposite directions is often called a Bidirectional LSTM (BiLSTM). 

#### Bias Evaluation
See results for details.


 
### Results

We successfully trained three models using the dataset of movie reviews. They were, in order, a word2vec model using Skipgrams, a LSTM model, and a Bidirectional LSTM. With each model, we saw notable improvements in performance, which we measured using training perplexity. Models with lower perplexity better predict samples and it is therefore more likely for their embeddings to be robust. 

| Model      | Final Perplexity Values |
| ----------- | ----------- |
| word2vec     | 107       |
| Vanilla LSTM     | 44       |
| BiLSTM     | 9.0       |

After each model was trained, we took the best-performing model, the BiLSTM, and used its embeddings to evaluate the bias learned from the movie reviews. We tested the cosine similarity of certain ungendered words to a number of gendered words (‘he’/ ‘she’, ‘man’/ ‘woman’, etc) and noted the average values below. 


**Cosine Similarity Values to Gendered Words**

```
test_bias(bilstm_model, Word, equality_sets)
```

| Word      | Similarity to Masculine Words | Similarity to Feminine Words  |
| :---        |    :----:   |          ---: |
| star     | 0.0750       | 0.0564   |
| great   | 0.0479        | 0.0259   |
| success | 0.0595  | -0.0348  |
| supporting | -0.0297 | 0.115 |
| original | 0.0376 | 0.012 |
| love | 0.056  |  0.061 | 
| dull  |  -0.065 |  0.0215 | 
| lead | 0.024 |  -0.070 | 
|  weak |  0.02 |  0.075 | 


One can see, with these examples alone, that the seemingly un-gendered words skew towards one or the other gender significantly. These indicate that the “un-gendered” words used in the film reviews (~25,000) of the dataset are biased towards males. Notably, “weak” has an average similarity to male coded words of a measly 0.002 as opposed to 0.075 for feminine words. It is almost entirely orthogonal in its cosine similarity to masculine words. This model shows that the film critics behind the reviews do demonstrate bias in the reviews they author.

### Challenges

Initially, the BiLSTM model took a very long time to train with the vocabulary size — to combat this, this model was retrained with application of the nltk library (as elaborated below). The Skipgrams model had a high loss value and was very noisy, so efforts to generate a gender subspace to begin debiasing have been unsuccessful. We reasoned that applying a smaller, denoised vocabulary with a smaller embedding matrix would improve these loss values and make it easier to identify bias. This would then allow us to begin debiasing. 
 
The hardest part of the project was preprocessing the raw movie review data for the model. Initial runs without significant preprocessing led to extremely large vocab sizes and unwieldy word2id maps, and the highly varied vocab made extracting meaningful word embeddings difficult. Additionally, large vocab sizes led to a large embedding matrix, which resulted in extremely long training times. 
 
Therefore it was necessary to <UNK> words with low frequency (obscure words, proper nouns, etc.) so that the model could better learn embeddings of more commonly used language and lower the overall vocab size. To facilitate this, we used the python nltk library to pare down the vocabulary and tokenize words. 


### Reflection
We feel that ultimately, our project was successful in exploring the question we sought to answer. The results unfortunately did confirm our initial hypothesis — the language used to describe each gender (from a purely binary perspective) is quite biased, particularly from the findings extracted from our most effective model (BiLSTM). Words such as “lead”, “great”, “star”, and “original” were male-coded, while “supporting”, “love”, “dull” were female-coded. While we reached our target goal of training the BiLSTM model and conducted the bias analysis, we did not start the debiasing process, which was our reach goal. However, taking the results from this program and the literature we found initially concerning the lack of female film critics, we believe that problems such as these must be addressed not through algorithmic processes after the words have been written.
Rather, nullifying the gender imbalance within the journalistic domain is imperative. As mentioned in the introduction, representations of women in film have already been shown to be skewed (e.g. Bechdel, Jane Tests) and the homogeneity of the critics who review them (as well as their language, as the model found,) only amplify these biases.
Had we more time, it would have been exciting to explore the cosine similarity values to gendered words based on the gender of the movie reviewer. We might even be able to explore the nuances of language that arise between gender mappings in positive versus negative movie reviews, since our dataset was already divided into those two (as the original paper was a sentiment analysis).




- Ghosh, R. (2020, May 27). Using deep learning to analyse movie posters for gender bias. Medium. Retrieved November 12, 2021, from https://medium.com/analytics-vidhya/using-deep-learning-to-analyse-movie-posters-for-gender-bias-4c0f1557a051. 

- Tolga Bolukbasi, Kai-Wei Chang, James Zou, Venkatesh Saligrama, and Adam Kalai. 2016. "[Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings](https://proceedings.neurips.cc/paper/2016/file/a486cd07e4ac3d270571622f4f316ec5-Paper.pdf)". In Proceedings of the 30th International Conference on Neural Information Processing Systems (NIPS'16). Curran Associates Inc., Red Hook, NY, USA, 4356–4364.

- *Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. (2011). [Learning Word Vectors for Sentiment Analysis](http://ai.stanford.edu/~amaas/papers/wvSent_acl2011.pdf). The 49th Annual Meeting of the Association for Computational Linguistics (ACL 2011).

- **Melamud, O., Goldberger, J., & Dagan, I. (2016). Context2vec: Learning generic context embedding with Bidirectional LSTM. Proceedings of The 20th SIGNLL Conference on Computational Natural Language Learning. https://doi.org/10.18653/v1/k16-1006 

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


