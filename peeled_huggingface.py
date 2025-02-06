# YOU NEED TO BE LOGGED INTO HUGGINGFACE-HUB TO USE MOST USEFUL MODELS
# MAKE A HUGGINGFACE ACCOUNT, APPLY FOR ACCESS TO THE REPOSITORY, MAKE A TOKEN
# Login on CLI via: huggingface-cli login
#
# For your security, I recommend NOT storing your token in this repository's
# directories. If you must, use the '.token' extension so that gitignore will
# disallow indexing it and help you prevent accidental public token exposure.

# Dependent libraries
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Default libraries
import argparse
from typing import Union, List, Dict

class PromptLogitProcessor:
    """
        Based on https://github.com/NVIDIA/logits-processor-zoo/blob/main/logits_processor_zoo/transformers/cite_prompt.py
    """
    def __init__(self, tokenizer, prompt = None,
                 logit_push: float = 1.0,
                 push_eos: bool = True,
                 ):
        self.affect = logit_push
        self.push_eos = push_eos
        self.affected_token_ids = []
        if prompt is not None:
            self.update_prompt(prompt)

    def update_prompt(self, prompt, preserve_old_ids: bool = False):
        """
            Update the token IDs we affect based on prompt
            preserve_old_ids makes the set grow cumulatively with previous set(s)
        """
        new_ids = set()
        if self.push_eos:
            new_ids.add(self.tokenizer.eos_token_id)
        new_ids.add(set(self.tokenizer.encode(prompt)))
        if preserve_old_ids:
            new_ids = new_ids.union(set(self.affected_token_ids))
        self.affected_token_ids = list(new_ids)

    def __call__(self, input_ids, scores):
        """
            Additively adjust all affected token IDs in the logits by the
            affect amount
        """
        for i in range(scores.shape[0]):
            scores[i, self.affected_token_ids] += self.affect
        return scores

class HF_Interface():
    """
        Interface class to wrap my intended usage / interventions with
        HuggingFace transformers
    """
    def __init__(self,
                 model_name: str,
                 seed: int = 1024,
                 plot: bool = False,
                 text: bool = False,
                 ):
        self.model_name = model_name
        self.plot = plot
        self.text = text
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.set_seed(seed)

    def set_seed(self,
                 seed: int):
        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)

    def chat_format_prompt(self,
                           prompt: Union[str,Dict[str,str],List[Dict[str,str]]], # Prompt for LLM
                           role: str = 'user', # Role selection for prompts if given as string
                           add_generation_prompt: bool = True, # Add tokens to indicate LLM assistant should respond
                           ) -> str:
        # Can give your own chat_template (str) as a Jinja template to override
        # model/tokenizer's default, refer to the HF documentation for details:
        # https://huggingface.co/docs/transformers/chat_templating#advanced-how-do-chat-templates-work
        if type(prompt) is str:
            prompt = [{'role': role, 'content': prompt}]
        elif type(prompt) is dict:
            prompt = [prompt]
        return self.tokenizer.apply_chat_template(prompt,
                                                  tokenize=False, # Return as string
                                                  add_generation_prompt=True, # Add tokens to indicate LLM assistant should respond
                                                  )

    def generate_text_and_logits(self,
                                 prompt_: Union[str,Dict[str,str],List[Dict[str,str]]], # Prompt as human-given plaintext
                                 generation_config: GenerationConfig,
                                 logits_processor = None, # Given to alter logits during generation
                                 ):
        # Prepare the input
        prompt = self.chat_format_prompt(prompt_)
        # Tokenize the input
        inputs = self.tokenizer(prompt, return_tensors="pt")

        # Generate text
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                generation_config=generation_config,
                logits_processor=logits_processor, # If you want to affect logits *during* generation
                output_scores=True, # Enable score output to capture logits
                return_dict_in_generate=True, # Passed to .forward() or something
            )

        # Decode the output tokens to get the generated text
        sequence = output_ids.sequences[0]
        generated_text = self.tokenizer.decode(sequence, skip_special_tokens=True)

        # Post-process inspection of logits
        logits = output_ids.to_tuple()[1]
        # Vocab is usually string:id, invert it for id:string
        vk = dict((v,k) for (k,v) in self.tokenizer.get_vocab().items())
        # The token IDs we're operating on
        scores = output_ids.scores
        # This old version works when only one beam, fails for multi-beam
        #token_positions = [np.where(~token_logits[0].isinf())[0] for token_logits in output_ids.to_tuple()[1]]
        #logit_values = [[token_logits[0][idx].item() for idx in positions] \
        #                for (token_logits,positions) in zip(output_ids.to_tuple()[1],token_positions)]
        #logit_sort_order = [np.argsort(lv) for lv in logit_values]
        # Lookup has the vocab word for each non-infinite value in the logits
        #lookup = [[vk[k] for k in pos[lv]] for pos,lv in zip(token_positions,logit_sort_order)]
        # Argmax == greedy sampling which may not represent the generator's
        # sampling technique -- use `generation_config` to customize that behavior
        #highlight = [np.argmax(tid_list) for tid_list in logit_values]
        # Tuple (length #tokens) of tensors (#beams, |vocab|)
        out_tokens = []
        out_logits = []
        reconstruct = []
        highlight = []
        logit_sorts = []
        beam_selection = []
        for token_number, output_token in enumerate(scores):
            beam_tokens = []
            beam_logits = []
            beam_sorts = []
            beam_reconstruct = []
            beam_highlight = []
            for beam in output_token:
                generated = np.where(torch.isfinite(beam))[0]
                # I think this could be different for different beams?
                #if len(generated) == 0:
                #    continue
                beam_tokens.append(generated)
                beam_logits.append(beam[beam_tokens[-1]])
                beam_sorts.append(np.argsort(beam_logits[-1]))
                beam_reconstruct.append([vk[k] for k in beam_tokens[-1]])
                beam_highlight.append(beam_sorts[-1][0].item())
            # Nothing more to generate
            if sum(map(len,beam_tokens)) == 0:
                break
            out_tokens.append(beam_tokens)
            out_logits.append(beam_logits)
            logit_sorts.append(beam_sorts)
            reconstruct.append(beam_reconstruct)
            if len(output_token) == 1:
                beam_selection.append(0)
            else:
                beam_selection.append(output_ids.beam_indices[0,token_number])
            highlight.append(beam_highlight[beam_selection[-1]])

        # Reverse-engineer what highlight SHOULD look like based on the generated text?
        if self.plot:
            fig, ax = plt.subplots()
            # Per token generated
            for x_value, (beam_tokens, beam_heights) in enumerate(zip(reconstruct, out_logits)):
                n_beams = len(beam_tokens)
                # Per beam
                for beam_id, (beam_tok, beam_h) in enumerate(zip(beam_tokens, beam_heights)):
                    # Per candidate
                    beam_x = x_value+(beam_id/n_beams)
                    ax.scatter([beam_x]*len(beam_tok), beam_h)
                    for (text, height) in zip(beam_tok, beam_h):
                        ax.text(beam_x, height, text, ha='center',va='center',fontsize=8)
            plt.show()
        if self.text:
            print("\n".join([str(_) for _ in reconstruct]))
            print(highlight)

        return generated_text, reconstruct, out_logits

# Main function to tie everything together
def main(model_name: str,
         prompt: Union[str,Dict[str,str],List[Dict[str,str]]],
         seeds: list,
         generation_config: GenerationConfig,
         model: HF_Interface = None):
    if model is None:
        model = HF_Interface(model_name)

    for seed in seeds:
        print(f"Generating text with seed: {seed}")
        model.set_seed(seed)

        # Generate text and logits
        generated_text = model.generate_text_and_logits(prompt,
                                                        generation_config,
                                                        None, # No logit processing yet
                                                        )
        print(f"Generated Text with seed {seed}:\n{generated_text}")

def peeled_huggingface_build_ext(prs):
    dhelp = "(default: %(default)s)"
    prs.add_argument("--input", type=str, default=None,
                     help=f"Prompt text for the model")
    prs.add_argument("--input-from-file", action='store_true',
                     help=f"Indicates the --input argument is a file to be read {dhelp}")
    prs.add_argument("--system-prompt", type=str, default=None,
                     help=f"System prompt for the model")
    prs.add_argument("--system-prompt-from-file", action='store_true',
                     help=f"Indicates the --system-prompt argument is a file to be read {dhelp}")
    return prs

def peeled_huggingface_parse_ext(args):
    if args.input is None:
        args.input = "Once upon a time, in a distant land, there was a mysterious forest."
    elif args.input_from_file:
        with open(args.input,'r') as f:
            args.input = "".join(f.readlines())
    if args.system_prompt is not None:
        if args.system_prompt_from_file:
            with open(args.system_prompt,'r') as f:
                args.system_prompt = "".join(f.readlines())
        args.input = [{'role': 'system', 'content': args.system_prompt},
                      {'role': 'user', 'content': args.input}]
    return args

def build():
    dhelp = "(default: %(default)s)"
    req = "[REQUIRED]"
    prs = argparse.ArgumentParser()
    prs.add_argument("--seeds", type=int, default=None, action='append', nargs="+", required=True,
                     help=f"{req} RNG seeds for generation")
    prs.add_argument("--model-name", type=str, default="meta-llama/Llama-3.1-8B-Instruct",
                     help=f"HuggingFace model to load {dhelp}")
    gen_config = prs.add_argument_group("Generation Configuration")
    gen_config.add_argument("--temperature", type=float, default=0.7,
                            help=f"Softmax sharpening in [0,1] {dhelp}")
    gen_config.add_argument("--top-p", type=float, default=0.95,
                            help=f"Limit sampling to top-%% proportion by probability {dhelp}")
    gen_config.add_argument("--top-k", type=int, default=0,
                            help=f"Limit sampling to top-N tokens; typically 0 to disable and use other means {dhelp}")
    gen_config.add_argument("--num-beams", type=int, default=1,
                            help=f"Number of beams in beam search (1=no beam search) {dhelp}")
    gen_config.add_argument("--greedy-sample", action='store_true',
                            help=f"Reduce sampling variation by greedy token search {dhelp}")
    gen_config.add_argument("--max-new-tokens", type=int, default=500,
                            help=f"Number of new tokens permitted in response {dhelp}")
    gen_config.add_argument("--num-return-sequences", type=int, default=1,
                            help=f"When doing beam-search, number of decoding sequences to return {dhelp}")
    return prs

def parse(args=None, prs=None, build_extend_fn=None, prs_extend_fn=None):
    if prs is None:
        prs = build()
    if build_extend_fn is not None:
        prs = build_extend_fn(prs)
    if args is None:
        args = prs.parse_args()
    # Adjust generation config
    args.gen_config = GenerationConfig(
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            num_beams=args.num_beams,
            early_stopping=args.num_beams>1,
            do_sample=not args.greedy_sample,
            max_new_tokens=args.max_new_tokens,
            num_return_sequences=args.num_return_sequences,
            seed=None, # Overridden in HFI's generation wrapper
            )
    # Flatten args.seeds
    args.seeds = np.asarray(args.seeds).ravel()

    if prs_extend_fn is not None:
        args = prs_extend_fn(args)
    return args

if __name__ == "__main__":
    # CLI
    args = parse(build_extend_fn=peeled_huggingface_build_ext,
                 prs_extend_fn=peeled_huggingface_parse_ext)
    print(args.gen_config)

    # Run the script
    main(args.model_name, args.input, args.seeds, args.gen_config)

