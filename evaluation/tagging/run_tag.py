# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors,
# The HuggingFace Inc. team, and The XTREME Benchmark Authors.
# Copyright (c) 2018, NVIDIA CORPORATION.    All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Fine-tuning models for NER and POS tagging."""

from __future__ import absolute_import, division, print_function

import sys
import argparse
import glob
import logging
import os
import random
import re

import numpy as np
import torch
from seqeval.metrics import precision_score, recall_score, f1_score
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import RandomSampler, SequentialSampler
from tqdm import tqdm, trange
from evaluation.tagging.utils_tag import convert_examples_to_features
from evaluation.tagging.utils_tag import get_labels
from evaluation.tagging.utils_tag import read_examples_from_file

from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
    WEIGHTS_NAME,
    XLMRobertaConfig,
    XLMRobertaTokenizer,
    XLMRobertaForTokenClassification,
    #updated below
    RobertaConfig,
    RobertaTokenizer,
    RobertaForTokenClassification
)

# update below
import copy
from modeling_xlmr_extra import XLMRobertaAssembledForTokenClassification
from modeling_roberta_extra import RobertaAssembledForTokenClassification
from model_loader_extra import load_assembled_model


logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "xlmr": (XLMRobertaConfig, XLMRobertaForTokenClassification, XLMRobertaTokenizer, XLMRobertaAssembledForTokenClassification),
    # update below
    "roberta": (RobertaConfig, RobertaForTokenClassification, RobertaTokenizer, RobertaAssembledForTokenClassification)
}

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def calculate_mean_metrics(file_path):
    metrics = {
        'f1': [],
        'loss': [],
        'precision': [],
        'recall': []
    }
    
    with open(file_path, 'r') as file:
        for line in file:
            # Extract metric values using regex
            for key in metrics.keys():
                match = re.search(rf"{key} = ([\d.]+)", line)
                if match:
                    metrics[key].append(float(match.group(1)))
    
    # Calculate mean for each metric
    mean_metrics = {key: sum(values) / len(values) if values else None for key, values in metrics.items()}
    return mean_metrics

def train(args, train_dataset, model, tokenizer, labels, pad_token_label_id, lang2id=None):
    """Train the model."""
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("    Num examples = %d", len(train_dataset))
    logger.info("    Num Epochs = %d", args.num_train_epochs)
    logger.info("    Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("    Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("    Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("    Total optimization steps = %d", t_total)

    best_score = 0.0
    best_checkpoint = None
    patience = 0
    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args) # Add here for reproductibility (even between python 2 and 3)

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch if t is not None)
            inputs = {"input_ids": batch[0],
                        "attention_mask": batch[1],
                        "labels": batch[3]}

            if args.model_type != "distilbert":
                # XLM and RoBERTa don"t use segment_ids
                inputs["token_type_ids"] = batch[2] if args.model_type in ["bert", "xlnet"] else None

            outputs = model(**inputs)
            loss = outputs[0]

            if args.n_gpu > 1:
                # mean() to average on multi-gpu parallel training
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                scheduler.step()    # Update learning rate schedule
                optimizer.step()
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if args.local_rank == -1 and args.evaluate_during_training:
                        # Only evaluate on single GPU otherwise metrics may not average well
                        results, _ = evaluate(args, model, tokenizer, labels, pad_token_label_id, mode="dev", lang=args.train_langs, lang2id=lang2id)
                    logging_loss = tr_loss

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    if args.save_only_best_checkpoint:
                        result, _ = evaluate(args, model, tokenizer, labels, pad_token_label_id, mode="dev", prefix=global_step, lang=args.train_langs, lang2id=lang2id)
                        if result["f1"] > best_score:
                            logger.info("result['f1']={} > best_score={}".format(result["f1"], best_score))
                            best_score = result["f1"]
                            # Save the best model checkpoint
                            output_dir = os.path.join(args.output_dir, "checkpoint-best")
                            best_checkpoint = output_dir
                            if not os.path.exists(output_dir):
                                os.makedirs(output_dir)
                            # Take care of distributed/parallel training
                            model_to_save = model.module if hasattr(model, "module") else model
                            # model_to_save.save_pretrained(output_dir)
                            torch.save(model_to_save, os.path.join(output_dir, "pytorch_model.bin"))
                            torch.save(args, os.path.join(output_dir, "training_args.bin"))
                            logger.info("Saving the best model checkpoint to %s", output_dir)
                            logger.info("Reset patience to 0")
                            patience = 0
                        else:
                            patience += 1
                            logger.info("Hit patience={}".format(patience))
                            if args.eval_patience > 0 and patience > args.eval_patience:
                                logger.info("early stop! patience={}".format(patience))
                                epoch_iterator.close()
                                train_iterator.close()
                                return global_step, tr_loss / global_step
                    else:
                        # Save model checkpoint
                        output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        # Take care of distributed/parallel training
                        model_to_save = model.module if hasattr(model, "module") else model
                        # model_to_save.save_pretrained(output_dir)
                        torch.save(model_to_save, os.path.join(output_dir, "pytorch_model.bin"))
                        torch.save(args, os.path.join(output_dir, "training_args.bin"))
                        logger.info("Saving model checkpoint to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, labels, pad_token_label_id, mode, prefix="", lang="eng_Latn", lang2id=None, print_result=True):
    eval_dataset = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode=mode, lang=lang, lang2id=lang2id)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation %s in %s *****" % (prefix, lang))
    logger.info("    Num examples = %d", len(eval_dataset))
    logger.info("    Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    model.eval()
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {"input_ids": batch[0],
                        "attention_mask": batch[1],
                        "labels": batch[3]}
            if args.model_type != "distilbert":
                # XLM and RoBERTa don"t use segment_ids
                inputs["token_type_ids"] = batch[2] if args.model_type in ["bert", "xlnet"] else None
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            if args.n_gpu > 1:
                # mean() to average on multi-gpu parallel evaluating
                tmp_eval_loss = tmp_eval_loss.mean()

            eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

    if nb_eval_steps == 0:
        results = {k: 0 for k in ["loss", "precision", "recall", "f1"]}
    else:
        eval_loss = eval_loss / nb_eval_steps
        preds = np.argmax(preds, axis=2)

        label_map = {i: label for i, label in enumerate(labels)}

        out_label_list = [[] for _ in range(out_label_ids.shape[0])]
        preds_list = [[] for _ in range(out_label_ids.shape[0])]

        for i in range(out_label_ids.shape[0]):
            for j in range(out_label_ids.shape[1]):
                if out_label_ids[i, j] != pad_token_label_id:
                    out_label_list[i].append(label_map[out_label_ids[i][j]])
                    preds_list[i].append(label_map[preds[i][j]])

        results = {
            "loss": eval_loss,
            "precision": precision_score(out_label_list, preds_list),
            "recall": recall_score(out_label_list, preds_list),
            "f1": f1_score(out_label_list, preds_list)
        }

    if print_result:
        logger.info("***** Evaluation result %s in %s *****" % (prefix, lang))
        for key in sorted(results.keys()):
            logger.info("    %s = %s", key, str(results[key]))

    return results, preds_list


def load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode, lang, lang2id=None, few_shot=-1):
    # Make sure only the first process in distributed training process
    # the dataset, and the others will use the cache
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()

    # Load data features from cache or dataset file
    print(args.tokenized_dir)
    cached_features_file = os.path.join(args.tokenized_dir, "cached_{}_{}_{}_{}".format(mode, lang,
        list(filter(None, args.model_save_name.split("/"))).pop(),
        str(args.max_seq_len)))
    print(cached_features_file)
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        langs = lang.split(',')
        logger.info("all languages = {}".format(lang))
        features = []
        for lg in langs:
            data_file = os.path.join(args.tokenized_dir, lg, "{}".format(mode))
            logger.info("Creating features from dataset file at {} in language {}".format(data_file, lg))
            examples = read_examples_from_file(data_file, lg, lang2id)
            features_lg = convert_examples_to_features(examples, labels, args.max_seq_len, tokenizer,
                                                    cls_token_at_end=bool(args.model_type in ["xlnet"]),
                                                    cls_token=tokenizer.cls_token,
                                                    cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
                                                    sep_token=tokenizer.sep_token,
                                                    sep_token_extra=True,
                                                    pad_on_left=bool(args.model_type in ["xlnet"]),
                                                    pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                    pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
                                                    pad_token_label_id=pad_token_label_id,
                                                    lang=lg
                                                    )
            features.extend(features_lg)
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file {}, len(features)={}".format(cached_features_file, len(features)))
            torch.save(features, cached_features_file)

    # Make sure only the first process in distributed training process
    # the dataset, and the others will use the cache
    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()

    if few_shot > 0 and mode == 'train':
        logger.info("Original no. of examples = {}".format(len(features)))
        features = features[: few_shot]
        logger.info('Using few-shot learning on {} examples'.format(len(features)))

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
    if args.model_type == 'xlm' and features[0].langs is not None:
        all_langs = torch.tensor([f.langs for f in features], dtype=torch.long)
        logger.info('all_langs[0] = {}'.format(all_langs[0]))
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_langs)
    else:
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    return dataset


def run(args):
    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:
    # Initializes the distributed backend which sychronizes nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(handlers = [logging.FileHandler(args.log_file), logging.StreamHandler()],
                                            format = '%(asctime)s - %(levelname)s - %(name)s -     %(message)s',
                                            datefmt = '%m/%d/%Y %H:%M:%S',
                                            level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logging.info("Input args: %r" % args)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                     args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    # Prepare NER/POS task
    labels = get_labels(args.labels)
    num_labels = len(labels)
    # Use cross entropy ignore index as padding label id
    # so that only real label ids contribute to the loss later
    pad_token_label_id = CrossEntropyLoss().ignore_index

    # Load pretrained model and tokenizer
    # Make sure only the first process in distributed training loads model/vocab
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    # the below is the model loading part

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class, assembled_model_class = MODEL_CLASSES[args.model_type]

    # if the model is not initialized from a stored checkpoint
    if not args.init_checkpoint:
        config = config_class.from_pretrained(args.model_name_or_path, num_labels=num_labels)
        tokenizer = tokenizer_class.from_pretrained(
            args.model_name_or_path, do_lower_case=args.do_lower_case)
        model = model_class.from_pretrained(
            args.model_name_or_path, config=config)
    else:
        tokenizer_class = XLMRobertaTokenizer
        if args.checkpoint_num == 0 and args.use_initialization:
            config = config_class.from_pretrained(args.model_name_or_path, num_labels=num_labels)
            logger.info(f"Loading model from {args.embedding_dir} without continue pretraining!")

            # the masked language model
            loaded_model = load_assembled_model(args.model_name_or_path, args.num_primitive, args.embedding_dir, args.random_initialization)
            config.num_primitive = args.num_primitive
            config.vocab_size = loaded_model.config.vocab_size

            model = assembled_model_class(config)
            model.roberta = loaded_model.roberta
            config = model.config
            # the following is the glot500 tokenizer
            tokenizer = XLMRobertaTokenizer.from_pretrained('cis-lmu/glot500-base')
            assert config.vocab_size == len(tokenizer)
            # change the model class for later using
            if args.num_primitive != 768:
                model_class = assembled_model_class
        else:
            config = config_class.from_pretrained(args.init_checkpoint)
            config.num_labels = num_labels
            tokenizer = XLMRobertaTokenizer.from_pretrained(
                args.init_checkpoint, do_lower_case=args.do_lower_case)
            if args.use_initialization and args.num_primitive != 768:
                model = assembled_model_class(config)
                logger.info(f"Loading model from {args.init_checkpoint}/pytorch_model.bin")
                state_dict = torch.load(args.init_checkpoint + '/pytorch_model.bin')
                model.load_state_dict(state_dict, strict=False)
                uninitialized_params = {k: v for k, v in model.state_dict().items() if k not in state_dict}
                for k, v in uninitialized_params.items():
                    logger.info(f"{k} is randomly initialized!")
                model_class = assembled_model_class
            else:
                model = model_class.from_pretrained(args.init_checkpoint, config=config)
    print(model)

    # config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
    #                                         num_labels=num_labels,
    #                                         cache_dir=args.cache_dir if args.cache_dir else None)
    # tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
    #                                             do_lower_case=args.do_lower_case,
    #                                             cache_dir=args.cache_dir if args.cache_dir else None)

    # if args.init_checkpoint:
    #     logger.info("loading from init_checkpoint={}".format(args.init_checkpoint))
    #     model = model_class.from_pretrained(args.init_checkpoint,
    #                                                                             config=config,
    #                                                                             cache_dir=args.init_checkpoint)
    # else:
    #     logger.info("loading from cached model = {}".format(args.model_name_or_path))
    #     model = model_class.from_pretrained(args.model_name_or_path,
    #                                         from_tf=bool(".ckpt" in args.model_name_or_path),
    #                                         config=config,
    #                                         cache_dir=args.cache_dir if args.cache_dir else None)

    # the above is the model loading part
    lang2id = config.lang2id if args.model_type == "xlm" else None
    logger.info("Using lang2id = {}".format(lang2id))

    # Make sure only the first process in distributed training loads model/vocab
    if args.local_rank == 0:
        torch.distributed.barrier()
    model.to(args.device)
    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode="train", lang=args.train_langs, lang2id=lang2id, few_shot=args.few_shot)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer, labels, pad_token_label_id, lang2id)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use default names for the model,
    # you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        # Save model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        # Take care of distributed/parallel training
        logger.info("Saving model checkpoint to %s", args.output_dir)
        model_to_save = model.module if hasattr(model, "module") else model
        # model_to_save.save_pretrained(args.output_dir)
        torch.save(model_to_save, os.path.join(args.output_dir, "pytorch_model.bin"))
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

    # Initialization for evaluation
    results = {}
    if os.path.exists(os.path.join(args.output_dir, 'checkpoint-best')):
        best_checkpoint = os.path.join(args.output_dir, 'checkpoint-best')
    else:
        best_checkpoint = args.output_dir
    best_f1 = 0

    # Evaluation
    if args.do_eval and args.local_rank in [-1, 0]:
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)
        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            # model = model_class.from_pretrained(checkpoint)
            model = torch.load(os.path.join(checkpoint, "pytorch_model.bin"))
            model.to(args.device)
            result, _ = evaluate(args, model, tokenizer, labels, pad_token_label_id, mode="dev", prefix=global_step, lang=args.train_langs, lang2id=lang2id)
            if result["f1"] > best_f1:
                best_checkpoint = checkpoint
                best_f1 = result["f1"]
            if global_step:
                result = {"{}_{}".format(global_step, k): v for k, v in result.items()}
            results.update(result)

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            for key in sorted(results.keys()):
                writer.write("{} = {}\n".format(key, str(results[key])))
            writer.write("best checkpoint = {}, best f1 = {}\n".format(best_checkpoint, best_f1))

    # Prediction
    if args.do_predict and args.local_rank in [-1, 0]:
        logger.info("Loading the best checkpoint from {}\n".format(best_checkpoint))
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)

        # TODO
        # make sure the model class (if it is an assembled model can use save_pretrained or )
        # model = model_class.from_pretrained(best_checkpoint)
        model = torch.load(os.path.join(best_checkpoint, "pytorch_model.bin"))
        model.to(args.device)

        output_test_results_file = os.path.join(args.output_dir, "test_results.txt")
        with open(output_test_results_file, "w") as result_writer:
            for lang in args.predict_langs:
                if not os.path.exists(os.path.join(args.tokenized_dir, lang, 'test')):
                    logger.info("Language {} for test predition does not exist".format(lang))
                    continue
                result, predictions = evaluate(args, model, tokenizer, labels, pad_token_label_id, mode="test", lang=lang, lang2id=lang2id)

                # Save results
                result_writer.write("=====================\nlanguage={}\n".format(lang))
                for key in sorted(result.keys()):
                    result_writer.write("{} = {}\n".format(key, str(result[key])))
                # Save predictions
                output_test_predictions_file = os.path.join(args.output_dir, "test_{}_predictions.txt".format(lang))
                infile = os.path.join(args.tokenized_dir, lang, "test")
                idxfile = infile + '.idx'
                save_predictions(args, predictions, output_test_predictions_file, infile, idxfile)
            
            # Calculate and write mean metrics at the end
            mean_metrics = calculate_mean_metrics(output_test_results_file)
            result_writer.write("\n=====================\nAVERAGE METRICS\n")
            for key, value in sorted(mean_metrics.items()):
                if value is not None:
                    result_writer.write("{} = {:.4f}\n".format(key, value))
                    logger.info("Average {} = {:.4f}".format(key, value))

    # Predict dev set
    if args.do_predict_dev and args.local_rank in [-1, 0]:
        logger.info("Loading the best checkpoint from {}\n".format(best_checkpoint))
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        # model = model_class.from_pretrained(best_checkpoint)
        model = torch.load(os.path.join(best_checkpoint, "pytorch_model.bin"))
        model.to(args.device)

        output_test_results_file = os.path.join(args.output_dir, "dev_results.txt")
        with open(output_test_results_file, "w") as result_writer:
            for lang in args.predict_langs:
                if not os.path.exists(os.path.join(args.tokenized_dir, lang, 'dev')):
                    logger.info("Language {} for dev predition does not exist".format(lang))
                    continue
                result, predictions = evaluate(args, model, tokenizer, labels, pad_token_label_id, mode="dev", lang=lang, lang2id=lang2id)

                # Save results
                result_writer.write("=====================\nlanguage={}\n".format(lang))
                for key in sorted(result.keys()):
                    result_writer.write("{} = {}\n".format(key, str(result[key])))
                # Save predictions
                output_test_predictions_file = os.path.join(args.output_dir, "dev_{}_predictions.txt".format(lang))
                infile = os.path.join(args.tokenized_dir, lang, "dev")
                idxfile = infile + '.idx'
                save_predictions(args, predictions, output_test_predictions_file, infile, idxfile)

def save_predictions(args, predictions, output_file, text_file, idx_file, output_word_prediction=False):
    # Save predictions
    with open(text_file, "r") as text_reader, open(idx_file, "r") as idx_reader:
        text = text_reader.readlines()
        index = idx_reader.readlines()
        assert len(text) == len(index)

    # Sanity check on the predictions
    with open(output_file, "w") as writer:
        example_id = 0
        prev_id = int(index[0])
        for line, idx in zip(text, index):
            if line == "" or line == "\n":
                example_id += 1
            else:
                cur_id = int(idx)
                output_line = '\n' if cur_id != prev_id else ''
                if output_word_prediction:
                    output_line += line.split()[0] + '\t'
                output_line += predictions[example_id].pop(0) + '\n'
                writer.write(output_line)
                prev_id = cur_id

