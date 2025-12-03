"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
import os, json
from typing import List
import numpy as np
import torch as th
from transformers import set_seed
from src.utils import dist_util, logger

from src.utils.args_utils import *
from train_infer.factory_methods import create_model_and_diffusion
from src.utils.args_utils import create_argparser, args_to_dict, model_and_diffusion_defaults
from src.utils.custom_tokenizer import create_tokenizer
from functools import partial





def main():

    args = create_argparser().parse_args()

    set_seed(args.seed)
    dist_util.setup_dist()
    logger.configure()

    # load configurations.
    args.checkpoint_path = os.path.split(args.model_name_or_path)[0]

    config_path = os.path.join(args.checkpoint_path, "training_args.json")
    training_args = read_training_args(config_path)
    training_args["batch_size"] = args.batch_size
    training_args["diffusion_steps"] = args.diffusion_steps
    training_args['model_name_or_path'] = args.model_name_or_path
    training_args["clamp"] = args.clamp
    training_args['out_dir'] = args.out_dir
    training_args['num_samples'] = args.num_samples
    
    args.__dict__.update(training_args)
    args.sigma_small = True

        
    logger.info(f"Init pretrained = {args.init_pretrained}")
    logger.info(f"Freeze embeddings = {args.freeze_embeddings}")
    logger.info(f"Use pretrained embeddings = {args.use_pretrained_embeddings}")
    logger.info(f"Use bert tokenizer (from training) = {args.use_bert_tokenizer}")
    
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(dist_util.load_state_dict(args.model_name_or_path, map_location="cpu"))
    model.eval()

    # Determine which tokenizer to use based on training args
    # If model uses pretrained embeddings (BERT), we should use BERT tokenizer
    # But respect the training args if they specified otherwise
    use_bert_tok = args.use_pretrained_embeddings and (args.use_bert_tokenizer == "yes" or args.use_bert_tokenizer == True)
    if hasattr(args, 'use_bert_tokenizer') and isinstance(args.use_bert_tokenizer, str):
        use_bert_tok = args.use_bert_tokenizer.lower() in ["yes", "true", "1"]
    
    logger.info(f"Using BERT tokenizer: {use_bert_tok}")
    tokenizer = create_tokenizer(return_pretokenized=use_bert_tok, path=f"data/{args.dataset}/")
    
    # Verify vocab size matches
    model_vocab_size = model.word_embedding.weight.shape[0]
    tokenizer_vocab_size = getattr(tokenizer, 'vocab_size', len(tokenizer) if hasattr(tokenizer, '__len__') else None)
    logger.info(f"Model vocab size: {model_vocab_size}, Tokenizer vocab size: {tokenizer_vocab_size}")
    if tokenizer_vocab_size and model_vocab_size != tokenizer_vocab_size:
        logger.warning(f"WARNING: Vocab size mismatch! This may cause decoding issues.")
    
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    logger.log(f"the parameter count is {pytorch_total_params}")

    diffusion.rescale_timesteps = True

    model.to(dist_util.dev())
    model.eval()  # DEBUG


    logger.log("sampling...")
    logger.log(f"Clamping is set to {args.clamp}")
    
    # Create a denoised function that projects embeddings back to valid token embeddings
    # This prevents embeddings from drifting away during sampling
    def denoised_fn_round(text_emb, t):
        """Project continuous embeddings to nearest valid token embeddings."""
        word_emb = model.word_embedding.weight  # [vocab_size, emb_dim]
        old_shape = text_emb.shape
        old_device = text_emb.device
        
        # Flatten for efficient computation
        text_emb_flat = text_emb.reshape(-1, text_emb.size(-1))  # [bsz*seqlen, emb_dim]
        
        # Efficient L2 distance computation: ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a*b
        emb_norm = (word_emb ** 2).sum(-1).view(-1, 1)  # [vocab_size, 1]
        text_emb_t = text_emb_flat.transpose(0, 1)  # [emb_dim, bsz*seqlen]
        arr_norm = (text_emb_flat ** 2).sum(-1).view(-1, 1)  # [bsz*seqlen, 1]
        dist = emb_norm + arr_norm.transpose(0, 1) - 2.0 * th.mm(word_emb, text_emb_t)  # [vocab_size, bsz*seqlen]
        dist = th.clamp(dist, 0.0, float('inf'))
        
        # Find nearest token for each embedding
        topk_out = th.topk(-dist, k=1, dim=0)  # [1, bsz*seqlen]
        rounded_tokens = topk_out.indices[0]  # [bsz*seqlen]
        
        # Get the actual token embeddings
        new_embeds = model.word_embedding(rounded_tokens).view(old_shape).to(old_device)
        return new_embeds
    
    all_samples = []
    while len(all_samples) * args.batch_size < args.num_samples:
        model_kwargs = {}
        sample_shape = (args.batch_size, args.sequence_len, model.word_embedding.weight.shape[1])
        sample = diffusion.p_sample_loop(
            model,
            sample_shape,
            clip_denoised=args.clip_denoised,
            denoised_fn=denoised_fn_round,  # Always use rounding to prevent embedding drift
            model_kwargs=model_kwargs,
            top_p=args.top_p,
            progress=True,
            tokenizer=tokenizer,
            log_verbose=True
        )


        # Keep on GPU for potential reuse
        generated = sample.detach()  # stays on GPU
        # Append the batch as a whole array, don't extend (which would flatten)
        all_samples.append(generated.cpu().numpy())

        logger.log(f"created {len(all_samples) * args.batch_size} total samples so far")

    # Concatenate all batches along the batch dimension
    arr = np.concatenate(all_samples, axis=0)
    num_to_keep = args.num_samples * args.mbr_sample
    arr = arr[:num_to_keep]
    
    logger.log(f"Total samples collected: {len(all_samples)}, keeping: {num_to_keep}, arr shape: {arr.shape}")

    x_t = th.tensor(arr).cuda()

    # Use logits to get token IDs - the model was trained to predict via lm_head,
    # so this should be more accurate than direct rounding to word_embedding
    # (which may have drifted during training)
    with th.no_grad():
        logits = model.get_logits(x_t)  # bsz, seqlen, vocab
        # Use argmax to get the most likely token for each position
        token_ids = th.argmax(logits, dim=-1)  # [batch, seq_len]
    
    decoded_sentences = []
    
    logger.log(f"token_ids shape after squeeze: {token_ids.shape}")
    
    # Ensure token_ids is 2D: [batch, seq_len]
    if token_ids.dim() == 1:
        token_ids = token_ids.unsqueeze(0)
    
    # Clamp token IDs to valid range
    vocab_size = getattr(tokenizer, 'vocab_size', len(tokenizer) if hasattr(tokenizer, '__len__') else 30522)
    token_ids = th.clamp(token_ids, 0, vocab_size - 1)
    
    logger.log("sampling complete")
    logger.log(f"Decoding {token_ids.shape[0]} samples")
    logger.log("\nFINAL GENERATED TEXT:\n" + "="*80)
    
    # Get special token IDs for debugging
    pad_id = tokenizer.pad_token_id if hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None else None
    cls_id = tokenizer.cls_token_id if hasattr(tokenizer, 'cls_token_id') and tokenizer.cls_token_id is not None else None
    sep_id = tokenizer.sep_token_id if hasattr(tokenizer, 'sep_token_id') and tokenizer.sep_token_id is not None else None
    
    for i in range(token_ids.shape[0]):
        tokens = token_ids[i].cpu().tolist()  # Get token IDs as list of integers
        
        # Debug first sample
        if i == 0:
            logger.log(f"Sample 0 - First 20 token IDs: {tokens[:20]}")
            logger.log(f"Sample 0 - Unique tokens (first 50): {list(set(tokens[:50]))}")
            if pad_id is not None:
                pad_count = sum(1 for t in tokens if t == pad_id)
                logger.log(f"Sample 0 - [PAD] count: {pad_count}/{len(tokens)}")
            if cls_id is not None:
                cls_count = sum(1 for t in tokens if t == cls_id)
                logger.log(f"Sample 0 - [CLS] count: {cls_count}/{len(tokens)}")
            if sep_id is not None:
                sep_count = sum(1 for t in tokens if t == sep_id)
                logger.log(f"Sample 0 - [SEP] count: {sep_count}/{len(tokens)}")
        
        # Try decoding with skip_special_tokens=True
        text = tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        
        # If empty, try without skipping special tokens
        if not text.strip():
            text_with_special = tokenizer.decode(tokens, skip_special_tokens=False, clean_up_tokenization_spaces=True)
            logger.log(f"Sample {i} decoded to empty with skip_special_tokens=True")
            logger.log(f"  With special tokens: {text_with_special[:200]}")
            # Use the version with special tokens but clean it up
            text = text_with_special.strip()
            # Remove common special token strings
            text = text.replace('[PAD]', '').replace('[CLS]', '').replace('[SEP]', '').replace('[UNK]', '').strip()
            # Also try removing if they appear as actual tokens
            if pad_id is not None:
                tokens = [t for t in tokens if t != pad_id]
            if cls_id is not None:
                tokens = [t for t in tokens if t != cls_id]
            if sep_id is not None:
                tokens = [t for t in tokens if t != sep_id]
            # Re-decode if we filtered tokens
            if tokens:
                text = tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()
        
        if not text.strip():
            logger.log(f"WARNING: Sample {i} is still empty after all attempts!")
        
        print(f"{i:2d}: {text[:200] if text else '(EMPTY)'}")
        decoded_sentences.append(text)

    # Save to file using the original write_outputs function format
    model_dir = os.path.split(args.model_name_or_path)[0]
    model_base_name = os.path.split(args.model_name_or_path)[1]
    output_file_basepath = os.path.join(
        model_dir,
        f"{model_base_name}.samples_{len(decoded_sentences)}.steps-{args.diffusion_steps}.clamp-{args.clamp}",
    ) + ".txt"
    
    with open(output_file_basepath, "w") as text_fout:
        for generated_sentence in decoded_sentences:
            text_fout.write(generated_sentence + "\n")
    
    logger.log(f"written the decoded output to {output_file_basepath}")

def load_embeddings(checkpoint_path, tokenizer, emb_dim):
    embeddings = th.nn.Embedding(tokenizer.vocab_size, emb_dim)
    embeddings.load_state_dict(th.load(f'{checkpoint_path}/random_emb.torch'))
    return embeddings


def read_training_args(config_path):
    with open(config_path, "r") as f:
        return json.load(f)

if __name__ == "__main__":
    main()
