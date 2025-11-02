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

## Efficient corpus design for wake-word detection

https://ieeexplore.ieee.org/document/9383569 [January 2021]

> Delowar Hossain; Yoshinao Sato

Wake-word detection is an indispensable technology for preventing virtual voice agents from being unintentionally triggered. Although various neural networks were proposed for wake-word detection, less attention has been paid to efficient corpus design, which we address in this study. For this purpose, we collected speech data via a crowdsourcing platform and evaluated the performance of several neural networks when different subsets of the corpus were used for training. The results reveal the following requirements for efficient corpus design to produce a lower misdetection rate: (1) short segments of continuous speech can be used as negative samples, but they are not as effective as random words; (2) utterances of "adversarial" words, i.e., phonetically similar words to a wake-word, contribute to improving performance significantly when they are used as negative samples; (3) it is preferable for individual speakers to provide both positive and negative samples; (4) increasing the number of speakers is better than increasing the number of repetitions of a wake-word by each speaker.

---

## A Novel Loss Function and Training Strategy for Noise-Robust Keyword Spotting

https://ieeexplore.ieee.org/document/9465680 [25 June 2021]

> Espejo, Iván López; Tan, Zheng-Hua; Jensen, Jesper

The development of keyword spotting (KWS) systems that are accurate in noisy conditions remains a challenge. Towards this goal, in this paper we propose a novel training strategy relying on multi-condition training for noise-robust KWS. By this strategy, we think of the state-of-the-art KWS models as the composition of a keyword embedding extractor and a linear classifier that are successively trained. To train the keyword embedding extractor, we also propose a new (CN,2+1)-pair loss function extending the concept behind related loss functions like triplet and N-pair losses to reach larger inter-class and smaller intra-class variation. Experimental results on a noisy version of the Google Speech Commands Dataset show that our proposal achieves around 12% KWS accuracy relative improvement with respect to standard end-to-end multi-condition training when speech is distorted by unseen noises. This performance improvement is achieved without increasing the computational complexity of the KWS model.


---

## Exploring the application of synthetic audio in training keyword spotters

https://ieeexplore.ieee.org/document/9413448 [June 2021]

> Andrew Werchniak; Roberto Barra Chicote; Yuriy Mishchenko; Jasha Droppo; Jeff Condal; Peng Liu

The study of keyword spotting, a subfield within the broader field of speech recognition that centers around identifying individual keywords in speech audio, has gained particular importance in recent years with the rise of personal voice assistants such as Alexa. As voice assistants aim to rapidly expand to support new languages, keywords, and use cases, stakeholders face the issue of limited training data for these unseen scenarios. This paper details some initial exploration into the application of Text-To-Speech (TTS) audio as a "helper" tool for training keyword spotters in these low-resource scenarios. In the experiments studied in this paper, the careful mixing of TTS audio with human speech audio during training led to a reduction of over 11% in the detection-error-tradeoff (DET) area under the curve (AUC) metric.

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

##  Generating TTS Based Adversarial Samples for Training Wake-Up Word Detection Systems Against Confusing Words

https://www.isca-archive.org/odyssey_2022/wang22c_odyssey.html

> Haoxu Wang, Yan Jia, Zeqing Zhao, Xuyang Wang, Junjie Wang, Ming Li

Wake-up word detection models are widely used in real life, but suffer from severe performance degradation when encountering adversarial samples. In this paper we discuss the concept of confusing words in adversarial samples. Confusing words are commonly encountered, which are various kinds of words that sound similar to the predefined keywords. To enhance the robustness of the wake-up word detection system against confusing words, we propose several methods to generate the adversarial confusing samples for simulating real confusing words scenarios in which we usually do not have any real confusing samples in the training set. The generated samples include concatenated audio, synthesized data, and partially masked keywords. Moreover, we use a domain embedding concatenated system to improve the performance. Experimental results show that the adversarial samples generated in our approach help improve the system’s robustness in both the common scenario and the confusing words scenario. In addition, we release the confusing words testing database called HI-MIA-CW for future research.


---

## On the Efficiency of Integrating Self-Supervised Learning and Meta-Learning for User-Defined Few-Shot Keyword Spotting

> Wei-Tsung Kao; Yuan-Kuei Wu; Chia-Ping Chen; Zhi-Sheng Chen; Yu-Pao Tsai; Hung-Yi Lee

https://ieeexplore.ieee.org/document/10022697 [ 27 January 2023 ]

User-defined keyword spotting is a task to detect new spoken terms defined by users. This can be viewed as a few-shot learning problem since it is unreasonable for users to define their desired keywords by providing many examples. To solve this problem, previous works try to incorporate self-supervised learning models or apply meta-learning algorithms. But it is unclear whether self-supervised learning and meta-learning are complementary and which combination of the two types of approaches is most effective for few-shot keyword discovery. In this work, we systematically study these questions by utilizing various self-supervised learning models and combining them with a wide variety of meta-learning algorithms. Our result shows that HuBERT combined with Matching network achieves the best result and is robust to the changes of few-shot examples.

---

## Synth4Kws: Synthesized Speech for User Defined Keyword Spotting in Low Resource Environments
 
https://arxiv.org/abs/2407.16840 [Submitted on 23 Jul 2024]

> Pai Zhu, Dhruuv Agarwal, Jacob W. Bartel, Kurt Partridge, Hyun Jin Park, Quan Wang

One of the challenges in developing a high quality custom keyword spotting (KWS) model is the lengthy and expensive process of collecting training data covering a wide range of languages, phrases and speaking styles. We introduce Synth4Kws - a framework to leverage Text to Speech (TTS) synthesized data for custom KWS in different resource settings. With no real data, we found increasing TTS phrase diversity and utterance sampling monotonically improves model performance, as evaluated by EER and AUC metrics over 11k utterances of the speech command dataset. In low resource settings, with 50k real utterances as a baseline, we found using optimal amounts of TTS data can improve EER by 30.1% and AUC by 46.7%. Furthermore, we mix TTS data with varying amounts of real data and interpolate the real data needed to achieve various quality targets. Our experiments are based on English and single word utterances but the findings generalize to i18n languages and other keyword types. 


---

## Utilizing TTS Synthesized Data for Efficient Development of Keyword Spotting Model
 
https://arxiv.org/abs/2407.18879  [Submitted on 26 Jul 2024]

> Hyun Jin Park, Dhruuv Agarwal, Neng Chen, Rentao Sun, Kurt Partridge, Justin Chen, Harry Zhang, Pai Zhu, Jacob Bartel, Kyle Kastner, Gary Wang, Andrew Rosenberg, Quan Wang

This paper explores the use of TTS synthesized training data for KWS (keyword spotting) task while minimizing development cost and time. Keyword spotting models require a huge amount of training data to be accurate, and obtaining such training data can be costly. In the current state of the art, TTS models can generate large amounts of natural-sounding data, which can help reducing cost and time for KWS model development. Still, TTS generated data can be lacking diversity compared to real data. To pursue maximizing KWS model accuracy under the constraint of limited resources and current TTS capability, we explored various strategies to mix TTS data and real human speech data, with a focus on minimizing real data use and maximizing diversity of TTS output. Our experimental results indicate that relatively small amounts of real audio data with speaker diversity (100 speakers, 2k utterances) and large amounts of TTS synthesized data can achieve reasonably high accuracy (within 3x error rate of baseline), compared to the baseline (trained with 3.8M real positive utterances). 
  

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

---

## LLM-Synth4KWS: Scalable Automatic Generation and Synthesis of Confusable Data for Custom Keyword Spotting

https://arxiv.org/abs/2505.22995 [Submitted on 29 May 2025]

> Pai Zhu, Quan Wang, Dhruuv Agarwal, Kurt Partridge

Custom keyword spotting (KWS) allows detecting user-defined spoken keywords from streaming audio. This is achieved by comparing the embeddings from voice enrollments and input audio. State-of-the-art custom KWS models are typically trained contrastively using utterances whose keywords are randomly sampled from training dataset. These KWS models often struggle with confusing keywords, such as "blue" versus "glue". This paper introduces an effective way to augment the training with confusable utterances where keywords are generated and grouped from large language models (LLMs), and speech signals are synthesized with diverse speaking styles from text-to-speech (TTS) engines. To better measure user experience on confusable KWS, we define a new northstar metric using the average area under DET curve from confusable groups (c-AUC). Featuring high scalability and zero labor cost, the proposed method improves AUC by 3.7% and c-AUC by 11.3% on the Speech Commands testing set. 


---

##  Device Playback Augmentation with Echo Cancellation for Keyword Spotting

https://www.isca-archive.org/interspeech_2021/opatka21_interspeech.html

> Kuba Łopatka, Katarzyna Kaszuba-Miotke, Piotr Klinke, Paweł Trella

Keyword spotting (KWS) is required to operate in device playback conditions in which the device itself plays interfering signals. We propose a new method to augment the training set and adapt the acoustic model to the playback environment. It is based on acoustic simulation which models the coupling between the device’s loudspeakers and microphones. The employed model involves frequency response of the device, as well as room impulse response and nonlinear distortions introduced in the playback path. Finally, we pass the simulated signals through Acoustic Echo Cancellation (AEC) to model the artifacts introduced by AEC algorithm. The proposed method reduces False Rejection Rate in device playback noise by 25–60% for a Time-Delay Neural Network-based KWS engine. It is shown that the introduction of device characteristics and nonlinear filtration is necessary to achieve improvement in playback conditions. The augmentation scheme is highly independent of the architecture of the KWS system.

---

##  Few-Shot Keyword Spotting in Any Language

https://www.isca-archive.org/interspeech_2021/mazumder21_interspeech.html

> Mark Mazumder, Colby Banbury, Josh Meyer, Pete Warden, Vijay Janapa Reddi

We introduce a few-shot transfer learning method for keyword spotting in any language. Leveraging open speech corpora in nine languages, we automate the extraction of a large multilingual keyword bank and use it to train an embedding model. With just five training examples, we fine-tune the embedding model for keyword spotting and achieve an average F1 score of 0.75 on keyword classification for 180 new keywords unseen by the embedding model in these nine languages. This embedding model also generalizes to new languages. We achieve an average F1 score of 0.65 on 5-shot models for 260 keywords sampled across 13 new languages unseen by the embedding model. We investigate streaming accuracy for our 5-shot models in two contexts: keyword spotting and keyword search. Across 440 keywords in 22 languages, we achieve an average streaming keyword spotting accuracy of 87.4% with a false acceptance rate of 4.3%, and observe promising initial results on keyword search.

---

##  MatchboxNet: 1D Time-Channel Separable Convolutional Neural Network Architecture for Speech Commands Recognition

https://www.isca-archive.org/interspeech_2020/majumdar20_interspeech.html

> Somshubra Majumdar, Boris Ginsburg

We present MatchboxNet — an end-to-end neural network for speech command recognition. MatchboxNet is a deep residual network composed from blocks of 1D time-channel separable convolution, batch-normalization, ReLU and dropout layers. MatchboxNet reaches state-of-the art accuracy on the Google Speech Commands dataset while having significantly fewer parameters than similar models. The small footprint of MatchboxNet makes it an attractive candidate for devices with limited computational resources. The model is highly scalable, so model accuracy can be improved with modest additional memory and compute. Finally, we show how intensive data augmentation using an auxiliary noise dataset improves robustness in the presence of background noise.

---

##  Keyword Spotting with Synthetic Data using Heterogeneous Knowledge Distillation

https://www.isca-archive.org/interspeech_2022/lee22_interspeech.html

> Yuna Lee, Seung Jun Baek

It is crucial that Keyword Spotting (KWS) systems learn to understand new classes of user-defined keywords, which however is a challenging task requiring high-quality audio datasets. We propose KWS with Heterogeneous Embedding Knowledge Distillation (HEKD) which uses only synthetic data of unseen keyword classes. In HEKD, a reference model transfers the heterogeneous knowledge on seen classes to the student model for classifying keywords of unseen classes. By mimicking the embedding function of reference model trained on real data via a contrastive learning approach, we show that student model can learn to discriminate unseen keyword classes guided by synthetic data. In addition, we propose to maximize the dispersion of embedding clusters of unseen keywords with approximation guarantees in order to enhance the inter-class variability. Experiments show that HEKD outperforms baseline schemes using few-shot learning and those pre-trained on a large volume of data, demonstrating its effectiveness and efficiency.

---

## Keyword Transformer: A Self-Attention Model for Keyword Spotting

https://www.isca-archive.org/interspeech_2021/cai21c_interspeech.html

> Axel Berg, Mark O’Connor, Miguel Tairum Cruz

The Transformer architecture has been successful across many domains, including natural language processing, computer vision and speech recognition. In keyword spotting, self-attention has primarily been used on top of convolutional or recurrent encoders. We investigate a range of ways to adapt the Transformer architecture to keyword spotting and introduce the Keyword Transformer (KWT), a fully self-attentional architecture that exceeds state-of-the-art performance across multiple tasks without any pre-training or additional data. Surprisingly, this simple architecture outperforms more complex models that mix convolutional, recurrent and attentive layers. KWT can be used as a drop-in replacement for these models, setting two new benchmark records on the Google Speech Commands dataset with 98.6% and 97.7% accuracy on the 12 and 35-command tasks respectively.

---

## Iphonmatchnet: Zero-Shot User-Defined Keyword Spotting Using Implicit Acoustic Echo Cancellation

https://ieeexplore.ieee.org/document/10447696

> https://ieeexplore.ieee.org/document/10447696

In response to the increasing interest in human–machine communication across various domains, this paper introduces a novel approach called iPhonMatchNet, which addresses the challenge of barge-in scenarios, wherein user speech overlaps with device playback audio, thereby creating a self-referencing problem. The proposed model leverages implicit acoustic echo cancellation (iAEC) techniques to increase the efficiency of user-defined keyword spotting models, achieving a remarkable 95% reduction in mean absolute error with a minimal increase in model size (0.13%) compared to the baseline model, PhonMatchNet. We also present an efficient model structure and demonstrate its capability to learn iAEC functionality without requiring a clean signal. The findings of our study indicate that the proposed model achieves competitive performance in real-world deployment conditions of smart devices.
