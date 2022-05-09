---
layout: page
title: Results and Discussion
permalink: /results/
---

We present our baseline results for unimodal, and multimodal models for SAVSR in Table 1.

<figure>
    <center>
    <img src="/images/pic8.png">
    </center>
</figure>

# Baseline results
From AV-HuBERT baseline metric results, we can observe that combining both vision and speech features help improve performance compared to the unimodal models. For unimodal models, we observe that working with the speech only features AV-HuBERT already achieves good performance based on their extensive pre-training, but is also able to leverage vision features from the mouth of the speaker to resolve similarly pronounced words such as "relative" and "belated". However, providing only vision features severely harms the performance, which is an expected result since humans are also inaccurate in predicting words only from reading lips.

We also experimented on Conformer encoder - transformer decoder (Conf-trans) model and Contextualized blockwise conformer encoder - blockwise Streaming decoder (Stream-block) model for audio-only baselines. Although we were not able to meet word error rate results comparable to Av-HuBERT which uses extensive pre-training techniques, we were able to achieve word error rate where generated results are human understandable. Our streaming model performed worse compared to Conf-trans model since it limits the window size of context during encoding and decoding. The locality aspect of Streaming model resulted in lower performance in word error rate compared to Av-HuBERT and Conformer-transformer model.

However, compared to Av-HuBERT or Conformer transformer model, our contextualized streaming model was able to achieve much faster latency due to block-wise decoding. We find that the performance trade-off for latency is not huge since the majority of samples are human understandable for Streaming models. 

# Vision Integrated Models
We have experimented on early fusion of vision features with audio features before the encoder-decoder framework. For each of the models we experienced a increase of performance in accuracy. Based on qualitative results, we have observed that addition of vision features help distinguish words with similar pronunciation through visual features of lip readings. For example, errors such as confusing "workforce" to "work first" were fixed by adding mouth roi features. However, the addition of features result into lower performance in latency due to increased parameters of the model and 

We also experienced that providing the entire visual input deteriorates performance, where the model has trouble to focusing on the mouth features of the captured image that should be used for the task. Moreover, we observed that fusing visual features and audio features tend to harm convergence during the training process, which commonly lead to training failures such as overfitting to the model. 


# Qualitative Analysis
We provide failure example results for the same utterance for our models in Table 2. From the qualitative results, we can identify that adding ROI features help fix word errors due to similar pronunciation. Also we observe that streaming models lack context and suffer from more grammatical errors. 

<figure>
    <center>
    <img src="/images/pic9.png">
    </center>
</figure>