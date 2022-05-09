---
layout: page
title: Abstract
permalink: /abstract/
---

Streaming audio-visual speech recognition (SAVSR) introduces an online setting to audio-visual speech recognition (AVSR), which frees the full utterance requirement prior to decoding that traditional speech recognition models are limited to. Streaming audio-visual speech recognition further challenges the model leaving itself to decide how much the model should wait to have retrieved enough information to start decoding. While transformer based models such as AvHuBERT [1] have been successful in AVSR tasks through pretraining and cross-modal interactions, these models suffer in achieving reasonable Real-Time Factor (RTF) which is necessary for communication agents. We propose [ESPnet Mulimodal](https://github.com/chorongi/espnet_multimodal), a multimodal framework integrated to ESPnet [2], and provide baseline results for the task SAVSR. We also propose a streaming transformer [3]. and multimodal fusion based model for SAVSR. Through ESPnet Mulitmodal, we expect to facilitate research in the field of audio-visual tasks including SAVSR.