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
    
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(dist_util.load_state_dict(args.model_name_or_path, map_location="cpu"))
    model.eval()

    tokenizer = create_tokenizer(return_pretokenized=args.use_pretrained_embeddings, path=f"data/{args.dataset}/")
    
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
        all_samples.extend(generated.cpu().numpy())

        logger.log(f"created {len(all_samples)} samples")

    arr = np.concatenate(all_samples, axis=0)
    arr = arr[: args.num_samples * args.mbr_sample]

    x_t = th.tensor(arr).cuda()

    logits = model.get_logits(x_t)  # bsz, seqlen, vocab
    cands = th.topk(logits, k=1, dim=-1)

    decoded_sentences = []

    # Decode from token IDs (cands.indices) instead of continuous embeddings
    # cands.indices has shape [batch, seq_len, 1], so we squeeze the last dimension
    token_ids = cands.indices.squeeze(-1)  # [batch, seq_len]
    
    for seq in token_ids:
        tokens = seq.cpu().tolist()
        decoded = tokenizer.decode(tokens, skip_special_tokens=True).strip()
        print(decoded)
        decoded_sentences.append(decoded)

    logger.log("sampling complete")

    write_outputs(args=args, sentences=decoded_sentences)


def load_embeddings(checkpoint_path, tokenizer, emb_dim):
    embeddings = th.nn.Embedding(tokenizer.vocab_size, emb_dim)
    embeddings.load_state_dict(th.load(f'{checkpoint_path}/random_emb.torch'))
    return embeddings


def read_training_args(config_path):
    with open(config_path, "r") as f:
        return json.load(f)


def write_outputs(args: dict, sentences: List[str]) -> None:

    model_dir = os.path.split(args.model_name_or_path)[0]
    model_base_name = os.path.split(args.model_name_or_path)[1]
    
    num_samples = len(sentences)
    output_file_basepath = os.path.join(
        model_dir,
        f"{model_base_name}.samples_{num_samples}.steps-{args.diffusion_steps}.clamp-{args.clamp}",
    ) + ".txt"
    with open(output_file_basepath, "w") as text_fout:
        for generated_sentence in sentences:
            text_fout.write(generated_sentence + "\n")

        print(f"written the decoded output to {output_file_basepath}")


if __name__ == "__main__":
    main()
