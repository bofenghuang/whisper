#!/usr/bin/env bash
# Copyright 2023  Bofeng Huang

myscriptspath="/home/bhuang/myscripts"
export PYTHONPATH="${PYTHONPATH:-}:$myscriptspath"
export PYTHONUNBUFFERED=1

export CUDA_VISIBLE_DEVICES=2

# model_name_or_path="/home/bhuang/transformers/examples/pytorch/speech-recognition/outputs/general/whisper-large-v2-ft-lr4e6-bs256-punct-augment/checkpoint_openai.pt"
# model_name_or_path="/home/bhuang/transformers/examples/pytorch/speech-recognition/outputs/general/whisper-large-v2-ft-french-lr4e6-bs256-augment/checkpoint_openai.pt"
# model_name_or_path="/home/bhuang/transformers/examples/pytorch/speech-recognition/outputs/hmhm_merged_and_raw/openai-whisper_large_v2-ft-ep2-bs256-lr4e6-wd1e2-aug/checkpoint_openai.pt"
# model_name_or_path="/home/bhuang/transformers/examples/pytorch/speech-recognition/outputs/hmhm_merged_and_raw/bofenghuang-whisper_large_v2_french-ft-ep2-bs256-lr4e6-wd1e2-aug/checkpoint_openai.pt"
model_name_or_path="/home/bhuang/transformers/examples/pytorch/speech-recognition/outputs/hmhm_merged_and_raw/openai-whisper_large_v2-ft-ep2-bs256-lr4e6-wd1e2-aug-specaug/checkpoint_openai.pt"
# model_name_or_path="/home/bhuang/transformers/examples/pytorch/speech-recognition/outputs/hmhm_merged_and_raw/bofenghuang-whisper_large_v2_french-ft-ep2-bs256-lr4e6-wd1e2-aug-specaug/checkpoint_openai.pt"
# outdir="outputs/bofenghuang-whisper_large_v2_cv11_french"
# outdir="outputs/bofenghuang-whisper_large_v2_french"
# outdir="outputs/openai-whisper_large_v2-ft-ep2-bs256-lr4e6-wd1e2-aug"
# outdir="outputs/bofenghuang-whisper_large_v2_french-ft-ep2-bs256-lr4e6-wd1e2-aug"
outdir="outputs/openai-whisper_large_v2-ft-ep2-bs256-lr4e6-wd1e2-aug-specaug"
# outdir="outputs/bofenghuang-whisper_large_v2_french-ft-ep2-bs256-lr4e6-wd1e2-aug-specaug"

# --audio_file "/home/bhuang/asr/audio_samples/LBPA-WL_1676395452.7031390.wav" \

# python examples_new/run_transcribe.py \
#     --model_name $model_name_or_path \
#     --audio_file "/rd_storage/Audio/LBPA-WL/LBPA-WL_1676387329.7337023.wav" \
#     --with_timestamps True  \
#     --normalize_volume True  \
#     --out_file_path "outputs/LBPA-WL_1676387329.7337023_whisper-large-v2-cv11_greedy_normalized.csv"

# --normalize_volume True  \

# tmp
# python examples_new/run_transcribe.py \
#     --model_name $model_name_or_path \
#     --input_file_path "/home/bhuang/corpus/speech/internal/hm_hm/test_hmhm_10h.csv" \
#     --id_column_name "ID" \
#     --audio_column_name "wav" \
#     --start_column_name "start" \
#     --duration_column_name "duration" \
#     --prediction_column_name "prediction" \
#     --with_timestamps True \
#     --out_file_path "$outdir/test_hmhm_greedy_with_timestamps/predition.csv"

# python examples_new/eval.py \
#     --input_file_path "$outdir/test_hmhm_greedy_with_timestamps/predition.csv" \
#     --id_column_name "ID" \
#     --target_column_name "wrd" \
#     --prediction_column_name "prediction" \
#     --outdir "$outdir/test_hmhm_greedy_with_timestamps"

# python examples_new/run_transcribe.py \
#     --model_name $model_name_or_path \
#     --input_file_path "/home/bhuang/corpus/speech/internal/hm_hm/test_hmhm_10h.csv" \
#     --id_column_name "ID" \
#     --audio_column_name "wav" \
#     --start_column_name "start" \
#     --duration_column_name "duration" \
#     --prediction_column_name "prediction" \
#     --out_file_path "$outdir/test_hmhm_greedy/predition.csv"

# python examples_new/eval.py \
#     --input_file_path "$outdir/test_hmhm_greedy/predition.csv" \
#     --id_column_name "ID" \
#     --target_column_name "wrd" \
#     --prediction_column_name "prediction" \
#     --outdir "$outdir/test_hmhm_greedy"

# python examples_new/run_transcribe.py \
#     --model_name $model_name_or_path \
#     --input_file_path "/projects/corpus/voice/zaion/carglass/data/20220111/data_wo_incomplet_words.csv" \
#     --id_column_name "utt" \
#     --audio_column_name "path" \
#     --start_column_name "start" \
#     --duration_column_name "dur" \
#     --prediction_column_name "prediction" \
#     --out_file_path "$outdir/test_carglass_greedy/predition.csv"

# python examples_new/eval.py \
#     --input_file_path "$outdir/test_carglass_greedy/predition.csv" \
#     --id_column_name "utt" \
#     --target_column_name "text" \
#     --prediction_column_name "prediction" \
#     --outdir "$outdir/test_carglass_greedy"

# python examples_new/run_transcribe.py \
#     --model_name $model_name_or_path \
#     --input_file_path "/projects/corpus/voice/zaion/dekuple/2022_10_17/output/data/data_clean_without_words_with_dash.csv" \
#     --id_column_name "ID" \
#     --audio_column_name "wav" \
#     --start_column_name "start" \
#     --duration_column_name "duration" \
#     --prediction_column_name "prediction" \
#     --out_file_path "$outdir/test_dekuple_greedy/predition.csv"

# python examples_new/eval.py \
#     --input_file_path "$outdir/test_dekuple_greedy/predition.csv" \
#     --id_column_name "ID" \
#     --target_column_name "wrd" \
#     --prediction_column_name "prediction" \
#     --outdir "$outdir/test_dekuple_greedy"

# python examples_new/run_transcribe.py \
#     --model_name $model_name_or_path \
#     --input_file_path "/projects/corpus/voice/zaion/lbpa/2023-02-21/data/data_without_partial_words.csv" \
#     --id_column_name "ID" \
#     --audio_column_name "wav" \
#     --start_column_name "start" \
#     --duration_column_name "duration" \
#     --prediction_column_name "prediction" \
#     --out_file_path "$outdir/test_lbpa_greedy/predition.csv"

# python examples_new/eval.py \
#     --input_file_path "$outdir/test_lbpa_greedy/predition.csv" \
#     --id_column_name "ID" \
#     --target_column_name "wrd" \
#     --prediction_column_name "prediction" \
#     --outdir "$outdir/test_lbpa_greedy"


python examples_new/run_transcribe.py \
    --model_name $model_name_or_path \
    --input_file_path "/projects/corpus/voice/zaion/edenred/2023-05-10/data.tsv" \
    --input_file_sep "\t" \
    --id_column_name "ID" \
    --audio_column_name "wav" \
    --start_column_name "start" \
    --end_column_name "end" \
    --duration_column_name "duration" \
    --prediction_column_name "text" \
    --with_timestamps true \
    --temperature '[0.0]' \
    --out_file_path "/projects/corpus/voice/zaion/edenred/2023-05-10/data_whisper_greedy.tsv"