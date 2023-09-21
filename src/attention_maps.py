import argparse
import os
import glob
import time
import re
from typing import Tuple
from tqdm import tqdm
from numpy.random import default_rng
import numpy as np
import torch
import torch.nn.functional as F

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

LANG_ID = 'de'
MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-german"


def get_all_attns(src_sent):
    feed_dict = model.make_feed_dict([(src_sent, "")])
    tf.get_default_graph().clear_collection("AttnWeights")
    model.transformer.encode(feed_dict['inp'], feed_dict['inp_len'], False)
    info = tf.get_collection("AttnWeights")
    attns = [info[i][0].eval() for i in range(model.transformer.num_layers_enc)]
    return attns


def get_attns(src_sent, layer, head):
    all_attns = get_all_attns(src_sent)
    return all_attns[layer][head]


if __name__ == '__main__':

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(8)

    # load pretrained model
    processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID).to(device)
