# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************

import numpy as np
from scipy.io.wavfile import read
import torch
import os


def get_mask_from_lengths(lengths):
    max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, out=torch.cuda.LongTensor(max_len))
    mask = (ids < lengths.unsqueeze(1)).byte()
    return mask


def load_wav_to_torch(full_path):
    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_filepaths_and_text(filename, use_emotions=False, split="|"):
    with open(filename, encoding='utf-8') as f:
        def split_line(line):
            parts = line.strip().split(split)
            if len(parts) > 4:
                raise Exception(
                    "incorrect line format for file: {}".format(filename))

            path = parts[0]
            text = parts[1]
            speaker_id = int(parts[2])
            emotion_id = int(parts[3]) if use_emotions else None

            return path, text, speaker_id, emotion_id
        filepaths_and_text = [split_line(line) for line in f]
    return filepaths_and_text


def to_gpu(x):
    x = x.contiguous()

    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return torch.autograd.Variable(x)


def remove_crackle(d, h, s):
    """
    Args:
        d: audio time series, np.array
        h: threshold for median
        s: length of slices to apply

    Returns: filtered audio time series

    """
    n = int(np.floor(d.shape[0] / h))
    for i in np.arange(1, n):
        slice_i = d[i * h: (i + 1) * h]
        median_i = np.median(np.abs(slice_i))
        if np.abs(median_i) < s:
            d[i * h: (i + 1) * h] = d[i * h: (i + 1) * h] * 0
        j = i + 0.5
        slice_j = d[int((j) * h): int((j + 1) * h)]
        median_j = np.median(np.abs(slice_j))
        if np.abs(median_j) < s:
            d[int(j * h): int((j + 1) * h)] = d[int(j * h): int((j + 1) * h)] * 0

    d[int(d.shape[0] - h/4):] = d[int(d.shape[0] - h/4):] * 0

    return d