---
layout: page
title: Related Work
permalink: /related_work/
use_math: true
---

## Audio-Visual Speech Recognition Models

Audio-Visual speech recognition is one of the first well-studied multimodal fields because of the McGurk effect.
The release of LRS2 and LRS3 dataset led to significant increase in the research progress in this field. But the streaming research as compared off-line models has been limited. 

[1] learns audio visual representation via multimodal masking to learn a joint representation in a self-supervised fashion. AV-HuBERT is currently considered as the SoTA model for LRS3. AV-HuBERT encodes masked audio and image sequences into audio-visual features via a hybrid ResNet-transformer architecture to predict the predetermined sequence of discrete cluster assignments.

[7] proposes a streaming approach for transformer based speech recognition models. 
The main contributions of the paper are as follows: 
1) A blockwise synchronous beam search algorithm using BBD is proposed, which is incorporated with the contextual block processing of the encoder in CTC/attention hybrid decoding scheme. 2) Knowledge distillation [27–29] is performed on the streaming Transformer, guided by the original batch Transformer. 3) The proposed streaming Transformer algorithm is compared with conventional approaches including MoChA. The results indicate our approach outperforms them in the HKUST [30] and AISHELL-1 [31] Mandarin, LibriSpeech [32] English, and CSJ [33] Japanese tasks. 4) The impact of each factor in the proposed blockwise synchronous beam search on latency is evaluated through an ablation study


[4] proposes a sequence to sequence model with attention for audio visual speech recognition. The authors compare two papers one with visual adaption CTC and the other with global attention.

[5] proposes a new fusion strategy, incorporating reliability information in a decision fusion net that considers the temporal effects of the attention mechanism


<!-- % % Category 2
% \subsection{Streaming audio-visual speech recognition models}
% Our work is the first in the field of streaming multimodal fusion models 
% Paper Detail 1 -->

## Blockwise Non-Autoregressive Models

[7] combines blockwise-attention and connectionist temporal classification with Mask-CTC for non-autoregressive speech recognition. 

# Mask-CTC

Mask-CTC is a non-autoregressive model trained with both CTC objective and mask-prediction objective, where mask-predict decoder predicts the masked tokens based on CTC output tokens and encoder output. Since CTC lacks the ability of modeling correlations between output tokens due to its conditional independence assumption, Mask-CTC adopts an attention-based decoder as a masked language model (MLM) by iterative refining the output of CTC greedy decoding. [7]

<figure>
    <center>
    <img src="/images/pic1.png">
    </center>
</figure>


# Blockwise Attention and Mask-CTC

The core idea of blockwise-attention (BA) is that it simulates the environment when the encoder is only allowed to access limited future context. Each block only attends to the former layer's output within the current block and previous block. They define blockwise attention key, query, values as $[Z^{i-1}_{b-1}, Z^{i-1}_{b}]$, $Z^{i-1}_b$, and $[Z^{i-1}_{b-1}, Z^{i-1}_{b}]$ respectively where $Z^{i}_{b}$ represents the output of encoder layer i at the b-th block. Following the encoder, a 1D depthwise convolution followed by a Conformer structure. Due to the blockwise attention output, the paper performs greedy decoding of each block for blockwise Mask-CTC defined as the following:

<figure>
    <center>
    <img src="/images/pic2.png">
    </center>
</figure>

where $y_t_{i,b}$ denotes the label at the i-th time step of the b-th block. Note that blocks are consecutive blocks throughout the utterance with small overlaps. The paper further improves inference time through dynamic mapping approach for overlapping inference recovering erroneous output at the boundaries. 


<!--% Paper Detail 2-->
## Fusing Information Streams

[5] builds upon ESPnet and TM-CTC by incorporating reliability information in a decision fusion net that considers the temporal effects of the attention mechanism. Similar to Masked-CTC loss we defined above, ESPnet improves convergence by using a linear combination of sequence-to-sequence (S2S) transformer objective and CTC objective as the following. 

<figure>
    <center>
    <img src="/images/pic3.png">
    </center>
</figure>

The TM-CTC framework has two decoders for each stream, (audio, video), where CTC decoder contains 6 multi-head attention blocks and S2S decoder contains 6 decoder blocks. [5] adds another reliability measure encoder, with a sumbsampling layer which is fed into a BLSTM based decision fusion network (DFN). The main difference between TM-CTC is that given the output of TM-CTC decoders, $p_{ctc}(s|o)$ and $p_{s2s}(s|o)$, they concatenate these with reliability embedding vector $\xi_a$ and $\xi_v$ produced by reliability measure encoder. Then this concated vector is feeded into BLSTM based $DFN_{ctc}$ and $DFN_{s2s}$ to produce log probabilities $\log p_{ctc}(s|o)$ and $\log p_{s2s}(s|o)$.