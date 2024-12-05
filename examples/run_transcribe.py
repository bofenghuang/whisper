#! /usr/bin/env python
# coding=utf-8
# Copyright 2022 Bofeng Huang

import os
import time
from pathlib import Path
from typing import Optional, Union
import psutil
import fire
import jiwer
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import whisper
from tqdm import tqdm

from myscripts.audio.audio_utils import get_waveform

# from myscripts.data.text.normalize.french_asr import FrenchTextNormalizer
# from myscripts.data.text.wer.wer import stats_wer


MODLE_SAMPLING_RATE = 16_000


def _print_memory_info():
    memory = psutil.virtual_memory()
    print(
        f"Memory info - Free: {memory.available / (1024 ** 3):.2f} Gb, used: {memory.percent}%, total: {memory.total / (1024 ** 3):.2f} Gb"
    )


def _print_cuda_memory_info():
    used_mem, tot_mem = torch.cuda.mem_get_info()
    print(
        f"CUDA memory info - Free: {used_mem / 1024 ** 3:.2f} Gb, used: {(tot_mem - used_mem) / 1024 ** 3:.2f} Gb, total: {tot_mem / 1024 ** 3:.2f} Gb"
    )


def print_memory_info():
    _print_memory_info()
    _print_cuda_memory_info()


def infer(
    model,
    audio_file,
    start_on_second=None,
    duration_on_second=None,
    normalize_volume=False,
    with_timestamps=False,
    task="transcribe",
    language="fr",
    **gen_kwargs,
):
    # todo: add
    # gen_kwargs = {
    #     "task": "transcribe",
    #     "language": "fr",
    #     # "without_timestamps": True,
    #     # decode options
    #     # "beam_size": 5,
    #     # "patience": 2,
    #     # disable fallback
    #     # "compression_ratio_threshold": None,
    #     # "logprob_threshold": None,
    #     # vad threshold
    #     # "no_speech_threshold": None,
    # }

    if start_on_second is not None and duration_on_second is not None:
        # todo: read two times
        sample_rate = sf.info(audio_file).samplerate
        start = int(start_on_second * sample_rate)
        frames = int(duration_on_second * sample_rate)
    else:
        start = 0
        frames = -1

    # todo: doesn't work w/ mp3
    waveform, _ = get_waveform(
        audio_file,
        start=start,
        frames=frames,
        output_sample_rate=MODLE_SAMPLING_RATE,
        normalize_volume=normalize_volume,
        always_2d=False,
    )

    model_outputs = model.transcribe(
        waveform, task=task, language=language, without_timestamps=not with_timestamps, **gen_kwargs
    )
    # or read audio file inside transcribe func
    # model_outputs = model.transcribe(audio_file, without_timestamps=not with_timestamps, **gen_kwargs)

    return model_outputs


def postprocess_results(model_outputs, with_timestamps, prediction_column_name):
    if with_timestamps:
        return [
            {"start": seg["start"], "end": seg["end"], prediction_column_name: seg["text"].strip()}
            for seg in model_outputs["segments"]
        ]
    else:
        return [{prediction_column_name: model_outputs["text"].strip()}]


def main(
    model_name: str = "large",
    input_file_path: Optional[str] = None,
    input_file_sep: Optional[str] = ",",
    audio_file: Optional[str] = None,
    audio_dir: Optional[str] = None,
    out_file_path: Optional[str] = None,
    normalize_volume: bool = False,
    with_timestamps: bool = False,
    id_column_name: str = "ID",
    audio_column_name: str = "wav",
    start_column_name: str = "start",
    end_column_name: str = "end",
    duration_column_name: str = "duration",
    prediction_column_name: str = "prediction",
    **kwargs,
):
    if input_file_path is None and audio_file is None and audio_dir is None:
        raise ValueError("You have to either specify the input_file_path, audio_file or audio_dir")

    # device = 0 if torch.cuda.is_available() else "cpu"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Model will be loaded on device `{device}`")

    model = whisper.load_model(model_name, device=device)
    print(
        f"Model is {'multilingual' if model.is_multilingual else 'English-only'} "
        f"and has {sum(np.prod(p.shape) for p in model.parameters()):,} parameters."
    )

    # check mem and gpu info
    print_memory_info()

    start_time = time.perf_counter()

    if input_file_path is not None:
        outputs_df = pd.read_csv(input_file_path, sep=input_file_sep)
        output_text = []
        # todo: decode with batch, might need to extract the feature part into dataloader
        # process each segment row
        for segment in tqdm(outputs_df.to_dict("records")):
            # todo: tmp fix
            if duration_column_name not in segment:
                segment[duration_column_name] = segment[end_column_name] - segment[start_column_name]

            model_outputs = infer(
                model,
                segment[audio_column_name],
                start_on_second=segment[start_column_name],
                # end_on_second=segment[end_column_name],
                duration_on_second=segment[duration_column_name],
                normalize_volume=normalize_volume,
                with_timestamps=with_timestamps,
                **kwargs,
            )
            output_text.append(model_outputs["text"])
        outputs_df[prediction_column_name] = output_text
    elif audio_file is not None:
        model_outputs = infer(model, audio_file, normalize_volume=normalize_volume, with_timestamps=with_timestamps, **kwargs)
        processed_outputs = postprocess_results(
            model_outputs, with_timestamps=with_timestamps, prediction_column_name=prediction_column_name
        )
        outputs_df = pd.DataFrame(processed_outputs)
        outputs_df[id_column_name] = Path(audio_file).stem
    elif audio_dir is not None:
        outputs_data = []
        for p in tqdm(Path(audio_dir).rglob("*.wav")):
            model_output = infer(
                model, p.as_posix(), normalize_volume=normalize_volume, with_timestamps=with_timestamps, **kwargs
            )
            processed_output = postprocess_results(
                model_output, with_timestamps=with_timestamps, prediction_column_name=prediction_column_name
            )
            processed_output = [{**processed_output_, id_column_name: p.stem} for processed_output_ in processed_output]
            outputs_data.extend(processed_output)
            # break
        outputs_df = pd.DataFrame(outputs_data)

    print(f'Runtime: {time.strftime("%Hh%Mm%Ss", time.gmtime(time.perf_counter() - start_time))}')

    # print(outputs_df.head())
    # print(outputs_df.shape)

    if out_file_path is not None:
        out_file_dir = os.path.dirname(out_file_path)
        if not os.path.exists(out_file_dir):
            os.makedirs(out_file_dir)

        outputs_df.to_csv(out_file_path, index=False, sep="\t")
        print(f"Saved the results into {out_file_path}")


if __name__ == "__main__":
    fire.Fire(main)
