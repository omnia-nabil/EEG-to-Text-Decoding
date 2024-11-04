python /kaggle/working/EEG-to-Text-Decoding/eval_decoding_raw.py \
    --checkpoint_path /kaggle/input/train-eeg-t-text/checkpoints/decoding_raw/best/task1_task2_taskNRv2_finetune_BrainTranslator_skipstep1_b20_2_2_5e-05_5e-05_unique_sent.pt \
    --config_path /kaggle/input/train-eeg-t-text/config/decoding_raw/task1_task2_taskNRv2_finetune_BrainTranslator_skipstep1_b20_2_2_5e-05_5e-05_unique_sent.json \
    -cuda cuda:0

