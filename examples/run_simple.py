# /usr/bin/env python
# coding=utf-8
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import json
import time

import numpy as np
import whisper

# model_name_or_path = "tiny"
# model_name_or_path = "large"
# model_name_or_path = "models/bofenghuang-whisper_large_v2_cv11_french.pt"
# model_name_or_path = "/projects/bhuang/models/asr/public/whisper-large-v3-french-distil-dec16/original_model.pt"
# model_name_or_path = "/projects/bhuang/models/asr/bofenghuang-whisper_large_v3_french_dec2_init_ft_ep16_bs256_lr1e4_preprend/original_model.pt"
model_name_or_path = "/projects/bhuang/models/asr/public/whisper-large-v3-french/original_model.pt"
# model_name_or_path = "/home/bhuang/asr/distil-whisper/training/outputs/models/bofenghuang-whisper_large_v3_french_dec16_init_ft_ep16_bs256_lr1e4_preprend/original_model.pt"

model = whisper.load_model(model_name_or_path)
print(
    f"Model is {'multilingual' if model.is_multilingual else 'English-only'} "
    f"and has {sum(np.prod(p.shape) for p in model.parameters()):,} parameters."
)

# audio_file_path = "/home/bhuang/tmp/0130368055-eurexo-2021_04_20_10_49_56-TR_1618908583.3351428-out-000.wav"
# audio_file_path = "/home/bhuang/tmp/La_French_Best_scene_Cyril_Lecomte.wav"
# audio_file_path = "/home/bhuang/tmp/Pokora-Juste_Une_Photo_De_Toi.wav"
# audio_file_path = "/home/bhuang/tmp/lalala.wav"
# audio_file_path = "/home/bhuang/tmp/Real_French_Dialogue.wav"
# audio_file_path = "/home/bhuang/asr/whisper/ls_test.flac"
audio_file_path = "/home/bhuang/asr/audio_samples/audio-dikkenek.wav"

gen_kwargs = {
    # "task": "transcribe",
    "language": "fr",
    # "word_timestamps": True,
    # "without_timestamps": True,
    # decode options
    # "beam_size": 5,
    # "patience": 2,
    # disable fallback
    # "compression_ratio_threshold": None,
    # "logprob_threshold": None,
    # vad threshold
    # "no_speech_threshold": None,
    # "initial_prompt": "MAJUSCULE",
}

start_time = time.perf_counter()

result = model.transcribe(audio_file_path, **gen_kwargs)

# print("\n\n".join([f'Segment {segment["id"]+1} from {segment["start"]:.2f}s to {segment["end"]:.2f}s:\n{segment["text"].strip()}' for segment in result["segments"]]))

print(f'Inference time: {time.strftime("%Hh%Mm%Ss", time.gmtime(time.perf_counter() - start_time))}')

print(result["text"])

# print(json.dumps(result, indent=2))
for segment in result["segments"]:
    del segment["tokens"]
    print(json.dumps(segment, indent=2, ensure_ascii=False))
