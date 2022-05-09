---
layout: page
title: Problem Statement
permalink: /problem_tatement/
use_math: true
---

# Streaming Audio-Visual Speech Recognition (SAVSR)

Given a video utterance of time duration $$ T $$ with frame rate $$ F $$, we extract aligned sequence of visual features $v_1, v_2, v_3, \cdot \cdot \cdot v_{k_t}$ and audio features $a_1, a_2, a_3, \cdot\cdot\cdot a_{k_t}$, where $k_t \in \{1, \cdot \cdot \cdot T\times F\}$. We aim to accurately predict the text-script $c_1, c_2, ..., c_s$ corresponding to the utterance only available at time-step $t$. 

$ v_1 $

$$
K(a,b) = \int \mathcal{D}x(t) \exp(2\pi i S[x]/\hbar)
$$

<!--figure>
    <center>
    <img src="/images/pic4.png">
    </center>
</figure-->

Thus, using CTC decoding as a decoding strategy where we maximize the negative log likelihood of the entire sequence, we predict corresponding text $\hat{c}_{1:s}$ by,

<figure>
    <center>
    <img src="/images/pic5.png">
    </center>
</figure>

Note that for an ordinary AVSR task, $k_t = T \times F$ where we would always have the full-length utterance available at decoding time. 


<!--
% \subsection{Continuous Emotion Recognition (CER)}
% Belfast Naturalistic Database contains 10 to 60 seconds–
% long audiovisuals taken from English television chat shows,
% current affairs programmes and interviews. It features 125
% subjects, of which 31 are male, and 94 are females. Out
% of 298 clips, 100 videos totalling 86 minutes in duration
% have been labelled with continuous-valued emotion labels
% 2. http://sspnet.eu/2010/02/belfast-naturalistic/
% for activation and evaluation dimensions, with additionally
% 16 basic classifying emotion labels
% A video clip is composed of various episodes where each episode is labeled by a continuous-valued emotion labels of N categories. 
-->

