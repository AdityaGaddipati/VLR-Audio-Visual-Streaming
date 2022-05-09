---
layout: page
title: Proposed Model
permalink: /proposed_model/
---

![image](assets/images/model_structure.png)

A diagram of our proposed model structure is shown in Figure 1. We plan to integrate multimodal fusion to audio features and resnet-18 vision features and perform contextualized block-wise encoding. For audio feature extraction we plan to experiment on raw MFCC features and Wav2Vec features, and for vision features we plan to use ResNET-18 vision features. We also plan to experiment on a modified version of ResNET-18 by changing the first layer to 3D conv layer as done in AV-HuBERT. For the encoder, we use contextualized block-wise encoder such that it chunks the fused input signal in a streaming manner, and keeps global information in a contextual vector to future encoding blocks. For the decoder, we choose to experiment on transformer decoder, Streaming Blockwise Decoder, and RNN Transducer model designed for unimodal speech only model designed for online settings. We expect including a multimodal fusion layer would help populate the missing information due to block-wise chunking. We also plan to experiment on modality drop-out, a technique used by AV-HuBERT such that we could generate a robust model form noise and missing modalities. 

![image](assets/images/streaming_transformer.png)

Although Transformers have gained success in several sequence processing tasks like Machine translaition and as an effective encoder for all the modalities speech,text,vision also for time-series data,but achieving online processing while keeping competitive performance is still essential for real-world interaction. In this course project, we take the first step on streaming audio visual speech recognition using a blockwise streaming Conformer which is an augmented transformer with convolution layers, which is based on contextual block processing and blockwise synchronous beam search. In addition, the CTC translation output is also used to refine the search space with CTC prefix score, achieving joint CTC/attention simultaneous translation via multi-tasking of CTC and attention loss. 