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

# Default libraries
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
                 ):
        self.model_name = model_name
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
                                 input_text: str, # Prompt as human-given plaintext
                                 generation_config: GenerationConfig,
                                 logits_processor = None, # Given to alter logits during generation
                                 ):
        # Prepare the input
        prompt = self.chat_format_prompt(input_text)
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
        token_positions = [np.where(~token_logits[0].isinf())[0] for token_logits in output_ids.to_tuple()[1]]
        logit_values = [[token_logits[0][idx].item() for idx in positions] \
                        for (token_logits,positions) in zip(output_ids.to_tuple()[1],token_positions)]
        # Lookup has the vocab word for each non-infinite value in the logits
        lookup = [[vk[k] for k in pos] for pos in token_positions]
        # Argmax == greedy sampling which may not represent the generator's
        # sampling technique -- use `generation_config` to customize that behavior
        highlight = [np.argmax(tid_list) for tid_list in logit_values]
        # Reverse-engineer what highlight SHOULD look like based on the generated text?

        return generated_text

# Main function to tie everything together
def main(model_name: str, input_text: str, seeds: list, generation_config: GenerationConfig):
    hfi = HF_Interface(model_name)

    for seed in seeds:
        print(f"Generating text with seed: {seed}")
        hfi.set_seed(seed)

        # Generate text and logits
        generated_text = hfi.generate_text_and_logits(input_text,
                                                      generation_config,
                                                      None, # No logit processing yet
                                                      )
        print(f"Generated Text with seed {seed}:\n{generated_text}")

if __name__ == "__main__":
    # TODO: All of this stuff should really be in argparse
    # Parameters
    model_name = "meta-llama/Llama-3.1-8B-Instruct"  # Replace with your desired model
    input_text = "Once upon a time, in a distant land, there was a mysterious forest."
    seeds = [10648, 12345, 78910]  # Example seeds for varying outputs
    generation_config = GenerationConfig(
        temperature=0.7, # Softmax sharpening
        top_k=0, # Limit sampling to top-N candidate tokens; deactivated in most sampling regimes
        top_p=0.95, # Limit sampling to top-F proportion of probability
        num_beams=1,  # You can adjust this for different types of beam search
        # Leaving n_beams=1 is greedy, n_beams > 1 permits multiple hypotheses
        # to be evaluated and it picks the most likely FULL sequence at the end
        # If you set multiple beams, make sure to also set:
        #early_stopping=True,
        do_sample=True,  # Ensures randomness, set to False for greedy search
        max_new_tokens=50, # Adjust this as needed to permit/deny yapping
        num_return_sequences=1, # When doing beam-search etc, how many decoding sequences to return
        seed=None  # You can leave this out, as we handle the seed manually in HFI's generation wrapper
    )
    print(generation_config)

    # Run the script
    main(model_name, input_text, seeds, generation_config)

