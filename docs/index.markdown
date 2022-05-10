---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: page
---

Karthik Ganesan  (karthikg@andrew.cmu.edu) <br>
Aditya Gaddipati (agaddipa@andrew.cmu.edu)

[Streaming Audio Visual ASR](https://github.com/AdityaGaddipati/VLR-Audio-Visual-Streaming/tree/main/streaming_audio_visual_asr)

- [Abstract](#1-abstract)
- [Introduction](#2-introduction)
- [Related Work](#3-related-work)
- [Problem Statement](#4-problem-statement)
- [Proposed Model](#5-proposed-model)
- [Experimentation Methodology](#6-experimentation-methodology)
- [Results and Discussion](#7-results-and-discussion)
- [Conclusion and Future Work](#8-conclusion-and-future-work)
- [References](#9-references)
- [Code](#code)

## 1. Abstract

Streaming audio-visual speech recognition (SAVSR) introduces an online setting to audio-visual speech recognition (AVSR), which frees the full utterance requirement prior to decoding that traditional speech recognition models are limited to. Streaming audio-visual speech recognition further challenges the model leaving itself to decide how much the model should wait to have retrieved enough information to start decoding. While transformer based models such as AvHuBERT [1] have been successful in AVSR tasks through pretraining and cross-modal interactions, these models suffer in achieving reasonable Real-Time Factor (RTF) which is necessary for communication agents. We propose [Streaming Audio Visual ASR](https://github.com/AdityaGaddipati/VLR-Audio-Visual-Streaming/tree/main/streaming_audio_visual_asr), a multimodal framework integrated to ESPnet [2], and provide baseline results for the task SAVSR. We also propose a streaming transformer [3]. and multimodal fusion based model for SAVSR. Through ESPnet Mulitmodal, we expect to facilitate research in the field of audio-visual tasks including SAVSR.

## 2. Introduction

Recently, advances in deep learning and transformer models have dramatically improved performance of end-to-end automatic speech recognition (ASR). Specifically, models with joint transformer with Connectionist Temporal Classification (CTC) decoding technique provide impressive performance results for acoustic speech recognition tasks ([1] [4]). Moreover, models tend to use multi-modal fusion techniques to aid missing or unreliable information from one modality [5].

However, human dialogue happens in an online fashion where the listener instantaneously interprets the speaker's intent before the speaker finishes and parallelly starts preparing for a response. For powerful transformer models to be applied to communication agents in the real world, it should be able to start making decisions before the entire input signal is even available to produce an instant response to the speaker. 
Therefore, we assert that studying streaming audio-visual speech recognition is crucial in mainly two aspects:

1. Streaming audio-visual speech recognition models human communication more accurately, whereby studying models for this task will provide insight on how machines should interpret and process language in real-life settings.

2. Studying application of powerful transformer models in a streaming environment will unveil hidden caveats for real-life applications that have not been studied, and further facilitate the development of truly human-like, real-time communication agents.

Non-autoregressive (NAR) models have been studied for streaming sequence-to-sequence tasks, but have been unsuccessful due to strong conditional independence assumptions of CTC. Mask-CTC models [6] and approaches using blockwise-attention [7] have been proposed to mitigate this problem. However, these techniques have been only introduced to tasks such as machine translation and speech recognition, leaving the area of streaming audio-visual speech recognition unexplored. The main technical challenges of SAVSR are of the following:
<!-- Identify Main Technical Challenges -->

1. Efficiency: Models should process data streams in an online fashion resulting in low-latency to be considered "real time".

2. Context: For streaming AVSR models, there exist a challenge of define (detecting) context range, and how to transfer long-term context across different decoding blocks.

3. Limited future temporal information: Since models are required to start prediction before availability of the full sequence, future time-step signals are limited. We expect multi-modal techniques would help mitigate noisy, missing future information that is challenged by SAVSR in nature. 

<!-- Impact on research community, society -->
In this project, we plan to explore multi-modal fusion techniques that would help populate the embedding space in an online streaming environment, and how we can apply multi-modal fusion techniques to uni-modal streaming ASR techniques for SAVSR.

## 3. Related Work

# 3.1 Audio-Visual Speech Recognition Models

Audio-Visual speech recognition is one of the first well-studied multimodal fields because of the McGurk effect.
The release of LRS2 and LRS3 dataset led to significant increase in the research progress in this field. But the streaming research as compared off-line models has been limited. 

[1] learns audio visual representation via multimodal masking to learn a joint representation in a self-supervised fashion. AV-HuBERT is currently considered as the SoTA model for LRS3. AV-HuBERT encodes masked audio and image sequences into audio-visual features via a hybrid ResNet-transformer architecture to predict the predetermined sequence of discrete cluster assignments.

[7] proposes a streaming approach for transformer based speech recognition models. 
The main contributions of the paper are as follows: 
1) A blockwise synchronous beam search algorithm using BBD is proposed, which is incorporated with the contextual block processing of the encoder in CTC/attention hybrid decoding scheme. 2) Knowledge distillation is performed on the streaming Transformer, guided by the original batch Transformer. 3) The proposed streaming Transformer algorithm is compared with conventional approaches including MoChA. The results indicate our approach outperforms them in the HKUST and AISHELL-1 Mandarin, LibriSpeech English, and CSJ Japanese tasks. 4) The impact of each factor in the proposed blockwise synchronous beam search on latency is evaluated through an ablation study

[4] proposes a sequence to sequence model with attention for audio visual speech recognition. The authors compare two papers one with visual adaption CTC and the other with global attention.

[5] proposes a new fusion strategy, incorporating reliability information in a decision fusion net that considers the temporal effects of the attention mechanism

# 3.2 Blockwise Non-Autoregressive Models

[7] combines blockwise-attention and connectionist temporal classification with Mask-CTC for non-autoregressive speech recognition. 

3.2.1 Mask-CTC

Mask-CTC is a non-autoregressive model trained with both CTC objective and mask-prediction objective, where mask-predict decoder predicts the masked tokens based on CTC output tokens and encoder output. Since CTC lacks the ability of modeling correlations between output tokens due to its conditional independence assumption, Mask-CTC adopts an attention-based decoder as a masked language model (MLM) by iterative refining the output of CTC greedy decoding. [7]

![image](assets/images/pic1.png)

3.2.2 Blockwise Attention and Mask-CTC

![image](assets/images/pic12.png)

# 3.3 Fusing Information Streams

![image](assets/images/pic13.png)

## 4. Problem Statement

Streaming Audio-Visual Speech Recognition (SAVSR)

![image](assets/images/pic11.png)

## 5. Proposed Model

![image](assets/images/model_structure.png)

A diagram of our proposed model structure is shown in Figure 1. We plan to integrate multimodal fusion to audio features and resnet-18 vision features and perform contextualized block-wise encoding. For audio feature extraction we plan to experiment on raw MFCC features and Wav2Vec features, and for vision features we plan to use ResNET-18 vision features. We also plan to experiment on a modified version of ResNET-18 by changing the first layer to 3D conv layer as done in AV-HuBERT. For the encoder, we use contextualized block-wise encoder such that it chunks the fused input signal in a streaming manner, and keeps global information in a contextual vector to future encoding blocks. For the decoder, we choose to experiment on transformer decoder, Streaming Blockwise Decoder, and RNN Transducer model designed for unimodal speech only model designed for online settings. We expect including a multimodal fusion layer would help populate the missing information due to block-wise chunking. We also plan to experiment on modality drop-out, a technique used by AV-HuBERT such that we could generate a robust model form noise and missing modalities. 

![image](assets/images/streaming_transformer.png)

Although Transformers have gained success in several sequence processing tasks like Machine translaition and as an effective encoder for all the modalities speech,text,vision also for time-series data,but achieving online processing while keeping competitive performance is still essential for real-world interaction. In this course project, we take the first step on streaming audio visual speech recognition using a blockwise streaming Conformer which is an augmented transformer with convolution layers, which is based on contextual block processing and blockwise synchronous beam search. In addition, the CTC translation output is also used to refine the search space with CTC prefix score, achieving joint CTC/attention simultaneous translation via multi-tasking of CTC and attention loss. 

## 6. Experimentation Methodology

# 6.1 Dataset

We use LRS3-TED [8] as our main dataset for SAVSR task. LRS3
is the one of largest publicly available sentence-level lip reading dataset to date. The dataset contains over 400 hours of video, extracted from 5594 TED and TEDx talsk in English, downloaded from YouTube. Cropped face tracks of the speaker are provided with corresponding audio and text transcripts as well as the alignment boundaries of every word are included in plain text files. 

# 6.2 Training Details

We train on Adam optimizer with learning rate 0.002, weight decay 0.000001, gradual warmup scheduler with 15000 warmup steps for 30 epochs on a NVIDIA GeFORCE RTX-2080 device. The model with best validation accuracy from token prediction cross-entropy loss is chosen as the best performing model for decoding. 

# 6.3 Evaluation Metrics
Word Error Rate (WER) is the most commonly used accuracy measure for ASR where it is calculated by dividing the number of errors (substitution, insertion, deletion) by the total number words in a sequence. We additionally use Latency as our evaluation metric from keeping track of the timestamp of "end of utterance" and "end of prediction" to evaluate the streaming ability of the model. 

![image](assets/images/pic7.png)

# 6.4 Vision Features

We extract vision features through a ResNET-18 pretrained model from mouth roi cropped vision files. To facilitate the training process, we preprocess the LRS3 dataset to extract roi cropped mp4 files in the same directory and used the cropped video files as vision data inputs. We have also experimented on using the full (non-cropped) version of video files, but observed that the model suffers from extracting valuable information from a diverse set of vision features deteriorating the performance further from only using audio signals. After visual feature extraction, we use a projection layer to project fused audio-visual features to the multimodal embedding space.

# 6.5 Audio Vision Alignment
Since audio and video have different frame rates available in the feature extraction step, alignment of vision and audio features is a challenge for multimodal AVSR. For LRS3 dataset, we collect audio features at a rate of 16000 fps and vision features at a rate of 25 fps. To align the vision features and audio features, we repeat the paired vision features corresponding to the audio feature at a timestamp and assume that the following audio-features will be having the same vision features until the next paired timestamp. We assume that a small window (0.04 seconds for our case) of audio features share the same vision features, and this small window is fine-grained enough to capture the movement of lips of speakers. 

## 7. Results and Discussion

We present our baseline results for unimodal, and multimodal models for SAVSR in Table 1.

![image](assets/images/pic8.png)

# 7.1 Baseline results
From AV-HuBERT baseline metric results, we can observe that combining both vision and speech features help improve performance compared to the unimodal models. For unimodal models, we observe that working with the speech only features AV-HuBERT already achieves good performance based on their extensive pre-training, but is also able to leverage vision features from the mouth of the speaker to resolve similarly pronounced words such as "relative" and "belated". However, providing only vision features severely harms the performance, which is an expected result since humans are also inaccurate in predicting words only from reading lips.

We also experimented on Conformer encoder - transformer decoder (Conf-trans) model and Contextualized blockwise conformer encoder - blockwise Streaming decoder (Stream-block) model for audio-only baselines. Although we were not able to meet word error rate results comparable to Av-HuBERT which uses extensive pre-training techniques, we were able to achieve word error rate where generated results are human understandable. Our streaming model performed worse compared to Conf-trans model since it limits the window size of context during encoding and decoding. The locality aspect of Streaming model resulted in lower performance in word error rate compared to Av-HuBERT and Conformer-transformer model.

However, compared to Av-HuBERT or Conformer transformer model, our contextualized streaming model was able to achieve much faster latency due to block-wise decoding. We find that the performance trade-off for latency is not huge since the majority of samples are human understandable for Streaming models. 

# 7.2 Vision Integrated Models
We have experimented on early fusion of vision features with audio features before the encoder-decoder framework. For each of the models we experienced a increase of performance in accuracy. Based on qualitative results, we have observed that addition of vision features help distinguish words with similar pronunciation through visual features of lip readings. For example, errors such as confusing "workforce" to "work first" were fixed by adding mouth roi features. However, the addition of features result into lower performance in latency due to increased parameters of the model.

We also experienced that providing the entire visual input deteriorates performance, where the model has trouble to focusing on the mouth features of the captured image that should be used for the task. Moreover, we observed that fusing visual features and audio features tend to harm convergence during the training process, which commonly lead to training failures such as overfitting to the model. 


# 7.3 Qualitative Analysis
We provide failure example results for the same utterance for our models in Table 2. From the qualitative results, we can identify that adding ROI features help fix word errors due to similar pronunciation. Also we observe that streaming models lack context and suffer from more grammatical errors. 

![image](assets/images/pic9.png)

## 8. Conclusion and Future Work

In this course project, we take the first step on streaming audio visual speech recognition which can pave way to multiple interaction related tasks using a block-wise streaming Conformer based on contextual block processing and block-wise synchronous beam search. We also propose a technique to combine pre-trained vision and speech models by matching their frame-rates via an alignment aware upsampling technique for the visual features so that they are fused along the time axis. This fusion along time axis allows to extend techniques that work for unimodal speech recognition to multimodal audio-visual speech recognition. 

## 9. References

[1] Shi, B., W.-N. Hsu, K. Lakhotia, et al. Learning audio-visual speech representation by masked219
multimodal cluster prediction, 2022.220

[2] Watanabe, S., T. Hori, S. Karita, et al. Espnet: End-to-end speech processing toolkit, 2018.221

[3] Tsunoo, E., Y. Kashiwagi, S. Watanabe. Streaming transformer asr with blockwise synchronous222
beam search, 2020.223

[4] Palaskar, S., R. Sanabria, F. Metze. End-to-end multimodal speech recognition. In 2018224
IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages225
5774–5778. 2018.226

[5] Yu, W., S. Zeiler, D. Kolossa. Fusing information streams in end-to-end audio-visual speech227
recognition, 2021.228

[6] Ghazvininejad, M., O. Levy, Y. Liu, et al. Mask-predict: Parallel decoding of conditional229
masked language models. In Proceedings of the 2019 Conference on Empirical Methods in230
Natural Language Processing and the 9th International Joint Conference on Natural Language231
Processing (EMNLP-IJCNLP), pages 6112–6121. Association for Computational Linguistics,232
Hong Kong, China, 2019.233

[7] Wang, T., Y. Fujita, X. Chang, et al. Streaming end-to-end asr based on blockwise non-234
autoregressive models, 2021.235

[8] Afouras, T., J. S. Chung, A. Zisserman. Lrs3-ted: a large-scale dataset for visual speech236
recognition, 2018.

## Code

[VLR-Audio-Visual-Streaming](https://github.com/AdityaGaddipati/VLR-Audio-Visual-Streaming)
