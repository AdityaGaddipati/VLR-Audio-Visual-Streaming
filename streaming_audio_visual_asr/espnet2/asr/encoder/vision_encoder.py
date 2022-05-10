# Copyright 2022 Hyukjae Kwark
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Encoder definition."""
import contextlib
import copy
from filelock import FileLock
import logging
import os
from typing import Optional
from typing import Tuple

import torch
from typeguard import check_argument_types

from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from torchvision import models, transforms
import torch.nn as nn

class ResNet(AbsEncoder):
    """FairSeq Wav2Vec2 encoder module.

    Args:
        input_size: input dim
        output_size: dimension of attention
        w2v_url: url to Wav2Vec2.0 pretrained model
        w2v_dir_path: directory to download the Wav2Vec2.0 pretrained model.
        normalize_before: whether to use layer_norm before the first block
        finetune_last_n_layers: last n layers to be finetuned in Wav2Vec2.0
                                0 means to finetune every layer if freeze_w2v=False.
    """

    def __init__(
        self,
        input_size: int = 224,
        output_size: int = 512,        
    ):
        assert check_argument_types()
        super().__init__()

        model = models.resnet18(pretrained=True)
        model = nn.Sequential(*list(model.children())[:-1])

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.encoders = model.to(self.device)
        # self.pretrained_params = copy.deepcopy(model.state_dict())
        self.transform = transforms.Compose([
                transforms.Resize(input_size),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        self.output_size = output_size


    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        x: torch.Tensor,
        ilens: torch.Tensor,
        prev_states: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Forward FairSeqWav2Vec2 Encoder.

        Args:
            x: input tensor (B, L, D, D, C)
            ilens: input length (B)
        Returns:
            position embedded tensor and mask
        """
        self.encoders.eval()
        batch_size = x.size(0)
        utt_length = x.size(1)
        if x.size(-1) == 3:
            x = x.permute(0,1,4,2,3)
        c = x.size(2)
        h = x.size(3)
        w = x.size(4)
        x = x.reshape(batch_size * utt_length, c, h, w)
        with torch.no_grad():
            x = self.transform(x)
            enc_output = self.encoders(x)
        enc_output = enc_output.squeeze()
        assert(enc_output.size(-1) == self.output_size)
        enc_output = enc_output.reshape(batch_size, utt_length,  self.output_size)
        return enc_output, ilens, None

    # def reload_pretrained_parameters(self):
    #     self.encoders.load_state_dict(self.pretrained_params)
    #     logging.info("Pretrained ResNet-18 model parameters reloaded!")




class VisionTransformer(AbsEncoder):
    """TODO: Vision Transformer feature extraction.

    Args:
        input_size: input dim
        output_size: dimension of attention

    """

    def __init__(
        self,
        input_size: int,
        output_size: int,

    ):
        assert check_argument_types()
        super().__init__()

        if w2v_url != "":
            try:
                import fairseq
                from fairseq.models.wav2vec.wav2vec2 import Wav2Vec2Model
            except Exception as e:
                print("Error: FairSeq is not properly installed.")
                print(
                    "Please install FairSeq: cd ${MAIN_ROOT}/tools && make fairseq.done"
                )
                raise e

        self.w2v_model_path = download_w2v(w2v_url, w2v_dir_path)

        self._output_size = output_size

        models, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            [self.w2v_model_path],
            arg_overrides={"data": w2v_dir_path},
        )
        model = models[0]

        if not isinstance(model, Wav2Vec2Model):
            try:
                model = model.w2v_encoder.w2v_model
            except Exception as e:
                print(
                    "Error: pretrained models should be within: "
                    "'Wav2Vec2Model, Wav2VecCTC' classes, etc."
                )
                raise e

        self.encoders = model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.pretrained_params = copy.deepcopy(model.state_dict())

        self.normalize_before = normalize_before
        if self.normalize_before:
            self.after_norm = LayerNorm(output_size)


        self.freeze_finetune_updates = freeze_finetune_updates
        self.register_buffer("num_updates", torch.LongTensor([0]))

    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        x: torch.Tensor,
        ilens: torch.Tensor,
        prev_states: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Forward FairSeqWav2Vec2 Encoder.

        Args:
            x: input tensor (B, L, D)
            ilens: input length (B)
        Returns:
            position embedded tensor and mask
        """
        print("------------------------------------")
        print(x.size())

        with torch.no_grad():
            enc_output = self.encoders(x.to(self.device))

        return enc_output, ilens, None

    def reload_pretrained_parameters(self):
        self.encoders.load_state_dict(self.pretrained_params)
        logging.info("Pretrained Wav2Vec model parameters reloaded!")
