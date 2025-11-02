___

## Speech Model Pre-training for End-to-End Spoken Language Understanding

https://arxiv.org/abs/1904.03670 [Submitted on 7 Apr 2019, last revised 25 Jul 2019]

> Loren Lugosch, Mirco Ravanelli, Patrick Ignoto, Vikrant Singh Tomar, Yoshua Bengio

Whereas conventional spoken language understanding (SLU) systems map speech to text, and then text to intent, end-to-end
SLU systems map speech directly to intent through a single trainable model. Achieving high accuracy with these
end-to-end models without a large amount of training data is difficult. We propose a method to reduce the data
requirements of end-to-end SLU in which the model is first pre-trained to predict words and phonemes, thus learning good
features for SLU. We introduce a new SLU dataset, Fluent Speech Commands, and show that our method improves performance
both when the full dataset is used for training and when only a small subset is used. We also describe preliminary
experiments to gauge the model's ability to generalize to new phrases not heard during training.


---

## Using Speech Synthesis to Train End-to-End Spoken Language Understanding Models

https://arxiv.org/abs/1910.09463  [Submitted on 21 Oct 2019]

> Loren Lugosch, Brett Meyer, Derek Nowrouzezahrai, Mirco Ravanelli

End-to-end models are an attractive new approach to spoken language understanding (SLU) in which the meaning of an
utterance is inferred directly from the raw audio without employing the standard pipeline composed of a separately
trained speech recognizer and natural language understanding module. The downside of end-to-end SLU is that in-domain
speech data must be recorded to train the model. In this paper, we propose a strategy for overcoming this requirement in
which speech synthesis is used to generate a large synthetic training dataset from several artificial speakers.
Experiments on two open-source SLU datasets confirm the effectiveness of our approach, both as a sole source of training
data and as a form of data augmentation.


---

## Training Keyword Spotters with Limited and Synthesized Speech Data

https://arxiv.org/abs/2002.01322  [Submitted on 31 Jan 2020]

> James Lin, Kevin Kilgour, Dominik Roblek, Matthew Sharifi

With the rise of low power speech-enabled devices, there is a growing demand to quickly produce models for recognizing
arbitrary sets of keywords. As with many machine learning tasks, one of the most challenging parts in the model creation
process is obtaining a sufficient amount of training data. In this paper, we explore the effectiveness of synthesized
speech data in training small, spoken term detection models of around 400k parameters. Instead of training such models
directly on the audio or low level features such as MFCCs, we use a pre-trained speech embedding model trained to
extract useful features for keyword spotting models. Using this speech embedding, we show that a model which detects 10
keywords when trained on only synthetic speech is equivalent to a model trained on over 500 real examples. We also show
that a model without our speech embeddings would need to be trained on over 4000 real examples to reach the same
accuracy.

---

## Mining Effective Negative Training Samples for Keyword Spotting

https://ieeexplore.ieee.org/document/9053009

> Jingyong Hou; Yangyang Shi; Mari Ostendorf; Mei-Yuh Hwang; Lei Xie [Added to IEEE Xplore: 09 April 2020 ]

Max-pooling neural network architectures have been proven to be useful for keyword spotting (KWS), but standard training
methods suffer from a class-imbalance problem when using all frames from negative utterances. To address the problem, we
propose an innovative algorithm, Regional Hard-Example (RHE) mining, to find effective negative training samples, in
order to control the ratio of negative vs. positive data. To maintain the diversity of the negative samples, multiple
non-contiguous difficult frames per negative training utterance are dynamically selected during training, based on the
model statistics at each training epoch. Further, to improve model learning, we introduce a weakly constrained
max-pooling method for positive training utterances, which constrains max-pooling over the keyword ending frames only at
early stages of training. Finally, data augmentation is combined to bring further improvement. We assess the algorithms
by conducting experiments on wake-up word detection tasks with two different neural network architectures. The
experiments consistently show that the proposed methods provide significant improvements compared to a strong baseline.
At a false alarm rate of once per hour, our methods achieve 45-58% relative reduction in false rejection rates over a
strong baseline.

---

## HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units

https://arxiv.org/abs/2106.07447 [Submitted on 14 Jun 2021]

> Wei-Ning Hsu, Benjamin Bolte, Yao-Hung Hubert Tsai, Kushal Lakhotia, Ruslan Salakhutdinov, Abdelrahman Mohamed

Self-supervised approaches for speech representation learning are challenged by three unique problems: (1) there are
multiple sound units in each input utterance, (2) there is no lexicon of input sound units during the pre-training
phase, and (3) sound units have variable lengths with no explicit segmentation. To deal with these three problems, we
propose the Hidden-Unit BERT (HuBERT) approach for self-supervised speech representation learning, which utilizes an
offline clustering step to provide aligned target labels for a BERT-like prediction loss. A key ingredient of our
approach is applying the prediction loss over the masked regions only, which forces the model to learn a combined
acoustic and language model over the continuous inputs. HuBERT relies primarily on the consistency of the unsupervised
clustering step rather than the intrinsic quality of the assigned cluster labels. Starting with a simple k-means teacher
of 100 clusters, and using two iterations of clustering, the HuBERT model either matches or improves upon the
state-of-the-art wav2vec 2.0 performance on the Librispeech (960h) and Libri-light (60,000h) benchmarks with 10min, 1h,
10h, 100h, and 960h fine-tuning subsets. Using a 1B parameter model, HuBERT shows up to 19% and 13% relative WER
reduction on the more challenging dev-other and test-other evaluation subsets.

---

## DistilHuBERT: Speech Representation Learning by Layer-wise Distillation of Hidden-unit BERT

https://arxiv.org/abs/2110.01900 [Submitted on 5 Oct 2021, last revised 28 Apr 2022]

> Heng-Jui Chang, Shu-wen Yang, Hung-yi Lee

Self-supervised speech representation learning methods like wav2vec 2.0 and Hidden-unit BERT (HuBERT) leverage unlabeled
speech data for pre-training and offer good representations for numerous speech processing tasks. Despite the success of
these methods, they require large memory and high pre-training costs, making them inaccessible for researchers in
academia and small companies. Therefore, this paper introduces DistilHuBERT, a novel multi-task learning framework to
distill hidden representations from a HuBERT model directly. This method reduces HuBERT's size by 75% and 73% faster
while retaining most performance in ten different tasks. Moreover, DistilHuBERT required little training time and data,
opening the possibilities of pre-training personal and on-device SSL models for speech.


---

## GraphemeAug: A Systematic Approach to Synthesized Hard Negative Keyword Spotting Examples

https://arxiv.org/abs/2505.14814  [Submitted on 20 May 2025, last revised 25 May 2025]

> Harry Zhang, Kurt Partridge, Pai Zhu, Neng Chen, Hyun Jin Park, Dhruuv Agarwal, Quan Wang

Spoken Keyword Spotting (KWS) is the task of distinguishing between the presence and absence of a keyword in audio. The
accuracy of a KWS model hinges on its ability to correctly classify examples close to the keyword and non-keyword
boundary. These boundary examples are often scarce in training data, limiting model performance. In this paper, we
propose a method to systematically generate adversarial examples close to the decision boundary by making
insertion/deletion/substitution edits on the keyword's graphemes. We evaluate this technique on held-out data for a
popular keyword and show that the technique improves AUC on a dataset of synthetic hard negatives by 61% while
maintaining quality on positives and ambient negative audio data. 

