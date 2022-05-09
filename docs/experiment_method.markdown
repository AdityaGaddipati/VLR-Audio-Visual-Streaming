---
layout: page
title: Experiment Methodology
permalink: /experiment_method/
---

# Dataset

We use LRS3-TED [8] as our main dataset for SAVSR task. LRS3
is the one of largest publicly available sentence-level lip reading dataset to date. The dataset contains over 400 hours of video, extracted from 5594 TED and TEDx talsk in English, downloaded from YouTube. Cropped face tracks of the speaker are provided with corresponding audio and text transcripts as well as the alignment boundaries of every word are included in plain text files. 

# Training Details

We train on Adam optimizer with learning rate 0.002, weight decay 0.000001, gradual warmup scheduler with 15000 warmup steps for 30 epochs on a NVIDIA GeFORCE RTX-2080 device. The model with best validation accuracy from token prediction cross-entropy loss is chosen as the best performing model for decoding. 

# Evaluation Metrics
Word Error Rate (WER) is the most commonly used accuracy measure for ASR where it is calculated by dividing the number of errors (substitution, insertion, deletion) by the total number words in a sequence. We additionally use Latency as our evaluation metric from keeping track of the timestamp of "end of utterance" and "end of prediction" to evaluate the streaming ability of the model. 

<figure>
    <center>
    <img src="/images/pic7.png">
    </center>
</figure>

# Vision Features

We extract vision features through a ResNET-18 pretrained model from mouth roi cropped vision files. To facilitate the training process, we preprocess the LRS3 dataset to extract roi cropped mp4 files in the same directory and used the cropped video files as vision data inputs. We have also experimented on using the full (non-cropped) version of video files, but observed that the model suffers from extracting valuable information from a diverse set of vision features deteriorating the performance further from only using audio signals. After visual feature extraction, we use a projection layer to project fused audio-visual features to the multimodal embedding space.

# Audio Vision Alignment
Since audio and video have different frame rates available in the feature extraction step, alignment of vision and audio features is a challenge for multimodal AVSR. For LRS3 dataset, we collect audio features at a rate of 16000 fps and vision features at a rate of 25 fps. To align the vision features and audio features, we repeat the paired vision features corresponding to the audio feature at a timestamp and assume that the following audio-features will be having the same vision features until the next paired timestamp. We assume that a small window (0.04 seconds for our case) of audio features share the same vision features, and this small window is fine-grained enough to capture the movement of lips of speakers. 