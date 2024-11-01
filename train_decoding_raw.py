import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import pickle
import json
from glob import glob
import time
from tqdm import tqdm
from transformers import BartTokenizer, BartForConditionalGeneration, BertTokenizer, BertConfig, BertForSequenceClassification, RobertaTokenizer, RobertaForSequenceClassification
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import sys
sys.path.insert(1, '/kaggle/input/data_raw/pytorch/default/1')
sys.path.insert(1, '/kaggle/input/model_decoding_raw/pytorch/default/1')
sys.path.insert(1, '/kaggle/input/config/pytorch/default/1')
for path in sys.path:
    print(path)

import data_raw
import config
import model_decoding_raw
from torch.nn.utils.rnn import pad_sequence

from nltk.translate.bleu_score import corpus_bleu
from rouge import Rouge
from bert_score import score

import warnings
warnings.filterwarnings('ignore')
from transformers import logging
logging.set_verbosity_error()
torch.autograd.set_detect_anomaly(True)

from torch.utils.tensorboard import SummaryWriter
LOG_DIR = "runs_h"
train_writer = SummaryWriter(os.path.join(LOG_DIR, "train"))
val_writer = SummaryWriter(os.path.join(LOG_DIR, "train_full"))
dev_writer = SummaryWriter(os.path.join(LOG_DIR, "dev_full"))


SUBJECTS = ['ZAB', 'ZDM', 'ZDN', 'ZGW', 'ZJM', 'ZJN', 'ZJS', 'ZKB', 'ZKH', 'ZKW', 'ZMG', 'ZPH', 
            'YSD', 'YFS', 'YMD', 'YAC', 'YFR', 'YHS', 'YLS', 'YDG', 'YRH', 'YRK', 'YMS', 'YIS', 'YTL', 'YSL', 'YRP', 'YAG', 'YDR', 'YAK']

def train_model(dataloaders, device, model, criterion, optimizer, scheduler, num_epochs=25, checkpoint_path_best='/kaggle/working/checkpoints/decoding_raw/best/temp_decoding.pt', checkpoint_path_last='/kaggle/working/checkpoints/decoding_raw/last/temp_decoding.pt', stepone=False, accumulation_steps=4, max_grad_norm=1.0):
    since = time.time()

    best_loss = float('inf')
    train_losses = []
    val_losses = []

    index_plot = 0
    index_plot_dev = 0

    scaler = torch.cuda.amp.GradScaler()  # For mixed precision training

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print(f"lr: {scheduler.get_lr()}")
        print('-' * 10)

        for phase in ['train', 'dev', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0

            if phase == 'test':
                target_tokens_list = []
                target_string_list = []
                pred_tokens_list = []
                pred_string_list = []

            with tqdm(dataloaders[phase], unit="batch") as tepoch:
                for batch_idx, (_, seq_len, input_masks, input_mask_invert, target_ids, target_mask, sentiment_labels, sent_level_EEG, input_raw_embeddings, input_raw_embeddings_lengths, word_contents, word_contents_attn, subject_batch) in enumerate(tepoch):
                    optimizer.zero_grad(set_to_none=True)

                    input_embeddings_batch = input_raw_embeddings.float().to(device, non_blocking=True)
                    input_embeddings_lengths_batch = torch.stack([torch.tensor(a.clone().detach()) for a in input_raw_embeddings_lengths], 0).to(device, non_blocking=True)
                    input_masks_batch = torch.stack(input_masks, 0).to(device, non_blocking=True)
                    input_mask_invert_batch = torch.stack(input_mask_invert, 0).to(device, non_blocking=True)
                    target_ids_batch = torch.stack(target_ids, 0).to(device, non_blocking=True)
                    word_contents_batch = torch.stack(word_contents, 0).to(device, non_blocking=True)
                    word_contents_attn_batch = torch.stack(word_contents_attn, 0).to(device, non_blocking=True)

                    target_ids_batch[target_ids_batch == tokenizer.pad_token_id] = -100

                    with torch.cuda.amp.autocast():  # Enable mixed precision
                        seq2seqLMoutput = model(
                            input_embeddings_batch, input_masks_batch, input_mask_invert_batch, target_ids_batch, input_embeddings_lengths_batch, word_contents_batch, word_contents_attn_batch, stepone, subject_batch, device)

                        if stepone:
                            loss = seq2seqLMoutput
                        else:
                            loss = criterion(seq2seqLMoutput.permute(0, 2, 1), target_ids_batch.long()) / accumulation_steps

                    if torch.isnan(loss):
                        print("NaN detected in loss, skipping backward pass.")
                        continue

                    if phase == 'train':
                        scaler.scale(loss).backward()

                        # Gradient clipping
                        if max_grad_norm > 0:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                        if (batch_idx + 1) % accumulation_steps == 0 or batch_idx == len(dataloaders[phase]) - 1:
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad(set_to_none=True)

                    running_loss += loss.item() * input_embeddings_batch.size(0)
                    tepoch.set_postfix(loss=loss.item(), lr=scheduler.get_lr())

                    del input_embeddings_batch, input_embeddings_lengths_batch, input_masks_batch, input_mask_invert_batch, target_ids_batch, word_contents_batch, word_contents_attn_batch
                    torch.cuda.empty_cache()  # Free up unused memory

                    if phase == 'train':
                        val_writer.add_scalar("train_full", loss.item(), index_plot)
                        index_plot += 1
                    if phase == 'dev':
                        dev_writer.add_scalar("dev_full", loss.item(), index_plot_dev)
                        index_plot_dev += 1

                    if phase == 'train':
                        scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]

            if phase == 'train':
                train_losses.append(epoch_loss)
                torch.save(model.state_dict(), checkpoint_path_last)
            elif phase == 'dev':
                val_losses.append(epoch_loss)

            train_writer.add_scalar(phase, epoch_loss, epoch)

            print('{} Loss: {:.4f}'.format(phase, epoch_loss))

            if phase == 'dev' and epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(model.state_dict(), checkpoint_path_best)
                print(f'Updated best checkpoint: {checkpoint_path_best}')

            if phase == 'test' and stepone == False:
                # Evaluation metrics code
                pass

        print()

    print(f"Train losses: {train_losses}")
    print(f"Val losses: {val_losses}")

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))
    torch.save(model.state_dict(), checkpoint_path_last)
    print(f'Updated last checkpoint: {checkpoint_path_last}')

    return model




def show_require_grad_layers(model):
    print()
    print(' require_grad layers:')
    # sanity check
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(' ', name)


if __name__ == '__main__':
    args = config.get_config('train_decoding')

    ''' config param'''
    dataset_setting = 'unique_sent'

    num_epochs_step1 = args['num_epoch_step1']
    num_epochs_step2 = args['num_epoch_step2']
    step1_lr = args['learning_rate_step1']
    step2_lr = args['learning_rate_step2']

    batch_size = args['batch_size']

    model_name = args['model_name']
    task_name = args['task_name']

    save_path = args['save_path']

    skip_step_one = args['skip_step_one']
    load_step1_checkpoint = args['load_step1_checkpoint']
    use_random_init = False

    if use_random_init and skip_step_one:
        step2_lr = 5*1e-4

    print(f'[INFO]using model: {model_name}')
    print(f'[INFO]using use_random_init: {use_random_init}')

    if skip_step_one:
        save_name = f'{task_name}_finetune_{model_name}_skipstep1_b{batch_size}_{num_epochs_step1}_{num_epochs_step2}_{step1_lr}_{step2_lr}_{dataset_setting}'
    else:
        save_name = f'{task_name}_finetune_{model_name}_2steptraining_b{batch_size}_{num_epochs_step1}_{num_epochs_step2}_{step1_lr}_{step2_lr}_{dataset_setting}'

    if use_random_init:
        save_name = 'randinit_' + save_name

    output_checkpoint_name_best = f'/kaggle/working/best/{save_name}.pt'
    output_checkpoint_name_last = f'/kaggle/working/last/{save_name}.pt'

    subject_choice = args['subjects']
    print(f'![Debug]using {subject_choice}')
    eeg_type_choice = args['eeg_type']
    print(f'[INFO]eeg type {eeg_type_choice}')
    bands_choice = args['eeg_bands']
    print(f'[INFO]using bands {bands_choice}')

    ''' set random seeds '''
    seed_val = 312
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    ''' set up device '''
    # use cuda
    if torch.cuda.is_available():
        # dev = "cuda:3"
        dev = args['cuda']
    else:
        dev = "cpu"
    device = torch.device(dev)
    print(f'[INFO]using device {dev}')
    print()

    ''' set up dataloader '''
    whole_dataset_dicts = []
    if 'task1' in task_name:
        dataset_path_task1 = '/kaggle/input/dataset/ZuCo/task1-SR/pickle/task1-SR-dataset_wRaw.pickle'
        with open(dataset_path_task1, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))
    if 'task2' in task_name:
        dataset_path_task2 = '/kaggle/input/dataset2/task2-NR-dataset_wRaw.pickle'
        with open(dataset_path_task2, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))

    print()

    if model_name in ['BrainTranslator', 'BrainTranslatorNaive']:
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

    # train dataset
    train_set = data_raw.ZuCo_dataset(whole_dataset_dicts, 'train', tokenizer, subject=subject_choice,
                             eeg_type=eeg_type_choice, bands=bands_choice, setting=dataset_setting, raweeg=True)
    # dev dataset
    dev_set = data_raw.ZuCo_dataset(whole_dataset_dicts, 'dev', tokenizer, subject=subject_choice,
                           eeg_type=eeg_type_choice, bands=bands_choice, setting=dataset_setting, raweeg=True)
    # test dataset
    test_set = data_raw.ZuCo_dataset(whole_dataset_dicts, 'test', tokenizer, subject=subject_choice,
                            eeg_type=eeg_type_choice, bands=bands_choice, setting=dataset_setting, raweeg=True)

    dataset_sizes = {'train': len(train_set), 'dev': len(
        dev_set), 'test': len(test_set)}
    print('[INFO]train_set size: ', len(train_set))
    print('[INFO]dev_set size: ', len(dev_set))
    print('[INFO]test_set size: ', len(test_set))

    # Allows to pad and get real size of eeg vectors
    def pad_and_sort_batch(data_loader_batch):
        """
        data_loader_batch should be a list of (sequence, target, length) tuples...
        Returns a padded tensor of sequences sorted from longest to shortest,
        """
        input_embeddings, seq_len, input_masks, input_mask_invert, target_ids, target_mask, sentiment_labels, sent_level_EEG, input_raw_embeddings, word_contents, word_contents_attn, subject = tuple(
            zip(*data_loader_batch))

        raw_eeg = []
        input_raw_embeddings_lenghts = []
        for sentence in input_raw_embeddings:
            input_raw_embeddings_lenghts.append(
                torch.Tensor([a.size(0) for a in sentence]))
            raw_eeg.append(pad_sequence(
                sentence, batch_first=True, padding_value=0).permute(1, 0, 2))

        input_raw_embeddings = pad_sequence(
            raw_eeg, batch_first=True, padding_value=0).permute(0, 2, 1, 3)

        return input_embeddings, seq_len, input_masks, input_mask_invert, target_ids, target_mask, sentiment_labels, sent_level_EEG, input_raw_embeddings, input_raw_embeddings_lenghts, word_contents, word_contents_attn, subject  # lengths


    # train dataloader
    train_dataloader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=pad_and_sort_batch)  # 4
    # dev dataloader
    val_dataloader = DataLoader(
        dev_set, batch_size=1, shuffle=False, num_workers=0, collate_fn=pad_and_sort_batch)  # 4
    # dev dataloader
    test_dataloader = DataLoader(
        test_set, batch_size=1, shuffle=False, num_workers=0, collate_fn=pad_and_sort_batch)  # 4
    # dataloaders
    dataloaders = {'train': train_dataloader,
                   'dev': val_dataloader, 'test': test_dataloader}

    ''' set up model '''
    if model_name == 'BrainTranslator':
        pretrained = BartForConditionalGeneration.from_pretrained(
            'facebook/bart-large')

        model = model_decoding_raw.BrainTranslator(pretrained, in_feature=1024, decoder_embedding_size=1024,
                                additional_encoder_nhead=8, additional_encoder_dim_feedforward=4096)

    model.to(device)

    ''' training loop '''

    ######################################################
    '''step one trainig'''
    ######################################################

    # closely follow BART paper
    if model_name in ['BrainTranslator']:
        for name, param in model.named_parameters():
            if param.requires_grad and 'pretrained' in name:
                if ('shared' in name) or ('embed_positions' in name) or ('encoder.layers.0' in name):
                    continue
                else:
                    param.requires_grad = False

    if skip_step_one:
        if load_step1_checkpoint:
            stepone_checkpoint = 'path_to_step_1_checkpoint.pt'
            print(f'skip step one, load checkpoint: {stepone_checkpoint}')
            model.load_state_dict(torch.load(stepone_checkpoint))
        else:
            print('skip step one, start from scratch at step two')
    else:
        model.to(device)

        ''' set up optimizer and scheduler'''
        optimizer_step1 = optim.SGD(filter(
            lambda p: p.requires_grad, model.parameters()), lr=step1_lr, momentum=0.9)

        exp_lr_scheduler_step1 = lr_scheduler.CyclicLR(optimizer_step1, 
                     base_lr = step1_lr, # Initial learning rate which is the lower boundary in the cycle for each parameter group
                     max_lr = 5e-3, # Upper learning rate boundaries in the cycle for each parameter group
                     mode = "triangular2") #triangular2

        ''' set up loss function '''
        criterion = nn.MSELoss()
        model.freeze_pretrained_bart()

        print('=== start Step1 training ... ===')
        # print training layers
        show_require_grad_layers(model)
        # return best loss model from step1 training
        model = train_model(dataloaders, device, model, criterion, optimizer_step1, exp_lr_scheduler_step1, num_epochs=num_epochs_step1,
                            checkpoint_path_best=output_checkpoint_name_best, checkpoint_path_last=output_checkpoint_name_last, stepone=True)
        
        train_writer.flush()
        train_writer.close()
        val_writer.flush()
        val_writer.close()
        dev_writer.flush()
        dev_writer.close()        
    
    ######################################################
    '''step two trainig'''
    ######################################################

    #model.load_state_dict(torch.load("./checkpoints/decoding_raw_104_h/last/task1_task2_taskNRv2_finetune_BrainTranslator_skipstep1_b4_25_100_5e-05_5e-05_unique_sent.pt"))
    model.freeze_pretrained_brain()

    ''' set up optimizer and scheduler'''
    optimizer_step2 = optim.SGD(filter(
            lambda p: p.requires_grad, model.parameters()), lr=step2_lr, momentum=0.9)

    exp_lr_scheduler_step2 = lr_scheduler.CyclicLR(optimizer_step2, 
                     base_lr = 0.0000005, # Initial learning rate which is the lower boundary in the cycle for each parameter group
                     max_lr =  0.00005, # Upper learning rate boundaries in the cycle for each parameter group
                     mode = "triangular2") #triangular2'''

    ''' set up loss function '''
    criterion = nn.CrossEntropyLoss()

    print()
    print('=== start Step2 training ... ===')
    # print training layers
    show_require_grad_layers(model)

    model.to(device)

    '''main loop'''
    trained_model = train_model(dataloaders, device, model, criterion, optimizer_step2, exp_lr_scheduler_step2, num_epochs=num_epochs_step2,
                                checkpoint_path_best=output_checkpoint_name_best, checkpoint_path_last=output_checkpoint_name_last, stepone=False)

    train_writer.flush()
    train_writer.close()
    val_writer.flush()
    val_writer.close()
    dev_writer.flush()
    dev_writer.close()

