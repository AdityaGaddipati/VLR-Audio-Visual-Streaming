---
layout: page
title: Introduction
permalink: /introduction/
---

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