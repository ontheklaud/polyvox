from io import BytesIO
import copy

from scipy.io import wavfile
import numpy as np
import torch
import torch.nn.functional as F
import intel_extension_for_pytorch as ipex

#from libs.ide8.tacotron2.tacotron2.model import Tacotron2 as Tacotron2MS
from libs.ide8.tacotron2.tacotron2.model2 import Tacotron2 as Tacotron2MS
from libs.nvstock.torch.SpeechSynthesis.Tacotron2.waveglow import model as waveglow
from run_tacotron2_nvstock import load_nv_hub_models_pt_explicit, nvidia_tts_utils_for_xpu

import streamlit as st


def interpolate_tensor(src_tensor, target_shape, mode: str = "bilinear"):

    assert len(src_tensor.shape) == 2, "This method only works for 2D tensors"

    tensor = src_tensor.unsqueeze(0).unsqueeze(0)
    resized_tensor = F.interpolate(tensor, size=target_shape, mode=mode)
    return resized_tensor.squeeze(0).squeeze(0)


def interpolate_2_tensor(src_tensor, target_shape, mode: str = "linear"):

    assert len(src_tensor.shape) == 2, "This method only works for 2D tensors"

    tensor = src_tensor.unsqueeze(1)

    resized_tensor = F.interpolate(tensor, size=target_shape, mode=mode)
    return resized_tensor.squeeze(1)


def interpolate_tensor_shape_aware(src_tensor, dim_fix, mode: str = "linear"):

    src_shape = src_tensor.shape
    # target_shape = (src_shape[0], src_shape[1] + dim_fix)
    # return interpolate_tensor(src_tensor, target_shape, mode=mode)

    target_shape = (src_shape[1] + dim_fix)
    return interpolate_2_tensor(src_tensor, target_shape, mode=mode)


def fix_tacotron2_stock_pt(state_dict: dict,
                           model_configs: dict):

    state_dict_fix = copy.deepcopy(state_dict)

    # errors before fix
    # RuntimeError: Error(s) in loading state_dict for Tacotron2:
    # 	Missing key(s) in state_dict: "symbols_embedding.weight", "speakers_embedding.weight".
    # 	Unexpected key(s) in state_dict: "embedding.weight".
    # 	size mismatch for decoder.attention_rnn.weight_ih: copying a param with shape torch.Size([4096, 768]) from checkpoint, the shape in current model is torch.Size([4096, 784]).
    # 	size mismatch for decoder.attention_layer.memory_layer.linear_layer.weight: copying a param with shape torch.Size([128, 512]) from checkpoint, the shape in current model is torch.Size([128, 528]).
    # 	size mismatch for decoder.decoder_rnn.weight_ih: copying a param with shape torch.Size([4096, 1536]) from checkpoint, the shape in current model is torch.Size([4096, 1552]).
    # 	size mismatch for decoder.linear_projection.linear_layer.weight: copying a param with shape torch.Size([80, 1536]) from checkpoint, the shape in current model is torch.Size([80, 1552]).
    # 	size mismatch for decoder.gate_layer.linear_layer.weight: copying a param with shape torch.Size([1, 1536]) from checkpoint, the shape in current model is torch.Size([1, 1552]).

    list_to_fix = ["decoder.attention_rnn.weight_ih",
                   "decoder.attention_layer.memory_layer.linear_layer.weight",
                   "decoder.decoder_rnn.weight_ih",
                   "decoder.linear_projection.linear_layer.weight",
                   "decoder.gate_layer.linear_layer.weight"]

    # step 1: rename
    state_dict_fix["symbols_embedding.weight"] = state_dict_fix["embedding.weight"]
    state_dict_fix.pop("embedding.weight")

    # configs
    use_ms = model_configs["use_ms"]
    speakers_embedding_dim = model_configs["speakers_embedding_dim"] if use_ms else 0
    use_emotions = model_configs["use_emotions"]
    emotions_embedding_dim = model_configs["emotions_embedding_dim"] if use_ms else 0

    # step 2: manual init required tensors
    if model_configs["use_ms"]:
        state_dict_fix["speakers_embedding.weight"] = \
            torch.nn.init.xavier_uniform_(torch.empty(1, speakers_embedding_dim))
    else:
        pass

    if use_ms or use_emotions:
        encoder_dim = 512
        fit_in_before = encoder_dim + speakers_embedding_dim + emotions_embedding_dim
        state_dict_fix["fit_latent.linear_layer.weight"] = \
            torch.nn.init.xavier_uniform_(torch.empty(encoder_dim, fit_in_before))

        state_dict_fix["fit_latent.linear_layer.bias"] = \
            torch.nn.init.zeros_(torch.empty(encoder_dim))

        # step 3b: key-based interpolation
        for key in list_to_fix:
            state_dict_fix[key] = interpolate_tensor_shape_aware(state_dict_fix[key], fit_in_before)
    else:
        pass

    # step 3a: key-based pop
    #for key in list_to_fix:
    #    state_dict_fix.pop(key)

    #for item in state_dict_fix.keys():
    #    print(f"{item} => {state_dict_fix[item].shape}")
    #print('=' * 20)

    # fin
    return state_dict_fix


def main():

    # multi-speaker awareness requires more arguments:
    # TypeError: Tacotron2.__init__() missing 5 required positional arguments: 'n_speakers', 'speakers_embedding_dim', 'use_emotions', 'n_emotions', and 'emotions_embedding_dim'

    # ide8/tacotron2 expects n_speakers: 128 / speakers_embedding_dim: 16 for tacotron2
    configs_aux_tacotron2: dict = {
        "use_ms": False, "n_speakers": 1, "speakers_embedding_dim": 16,
        "use_emotions": False, "n_emotions": 15,  "emotions_embedding_dim": 8,
    }
    tacotron2_m: torch.nn.Module = load_nv_hub_models_pt_explicit(
        "assets/nvidia_tacotron2pyt_fp32_20190427", Tacotron2MS,
        aux_configs=configs_aux_tacotron2,
        aux_state_dict_fix=fix_tacotron2_stock_pt,
        mode="eval", final_map_device="xpu",
        mismatch_relax=False
    )

    waveglow_m = load_nv_hub_models_pt_explicit(
        "assets/nvidia_waveglowpyt_fp32_20190427", waveglow.WaveGlow,
        mode="eval", final_map_device="xpu"
    )

    # text utils
    utils = nvidia_tts_utils_for_xpu()

    # Streamlit UI - title and description
    st.title("Tacotron 2 TTS")
    st.write("Enter text below to generate speech:")

    # Input text from the user
    input_text = st.text_input("Input Text", "Hello, welcome to Tacotron 2!")

    speaker_id = st.slider("Speaker:", 0, 128-1, value=0, step=1)

    # Check if input text is provided and ready for processing
    if st.button("Generate"):
        if input_text.strip():
            st.success(f"Processing text: '{input_text}'")

            sequences, lengths = utils.prepare_input_sequence([input_text.strip()], device='xpu')

            with torch.no_grad():

                speaker_id = torch.IntTensor([speaker_id]).to("xpu").long()
                outputs = tacotron2_m.infer(sequences, speaker_id)
                mel = outputs[1]

                audio = waveglow_m.infer(mel)

                audio_post = audio.squeeze().cpu().numpy()

                # Generate WAV in memory
                audio_buffer = BytesIO()
                wavfile.write(audio_buffer,
                              rate=22050,
                              data=(audio_post * 32767).astype(np.int16))  # Scale and convert to 16-bit PCM
                audio_buffer.seek(0)

                # Pass to Streamlit
                st.audio(audio_buffer, format="audio/wav")
                st.success("Audio generated successfully!")

        else:
            st.error("Please provide some text for synthesis.")

    # fin
    return


if __name__ == "__main__":
    main()
