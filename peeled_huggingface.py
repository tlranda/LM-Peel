# YOU NEED TO BE LOGGED INTO HUGGINGFACE-HUB TO USE MOST USEFUL MODELS
# MAKE A HUGGINGFACE ACCOUNT, APPLY FOR ACCESS TO THE REPOSITORY, MAKE A TOKEN
# Login on CLI via: huggingface-cli login
#
# For your security, I recommend NOT storing your token in this repository's
# directories. If you must, use the '.token' extension so that gitignore will
# disallow indexing it and help you prevent accidental public token exposure.
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import numpy as np

class HF_Interface():
    """
        Interface class to wrap my intended usage / interventions with
        HuggingFace transformers
    """
    def __init__(self, model_name: str, seed: int = 1024):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.set_seed(seed)

    def set_seed(self, seed: int):
        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)

    # Generate text and capture logits
    def generate_text_and_logits(self, input_text: str, generation_config: GenerationConfig):
        # Tokenize the input
        inputs = self.tokenizer(input_text, return_tensors="pt")

        # Generate text
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                generation_config=generation_config,
                output_scores=True,  # Enable score output to capture logits
                return_dict_in_generate=True
            )

        # Decode the output tokens to get the generated text
        sequence = output_ids.sequences[0]
        generated_text = self.tokenizer.decode(sequence, skip_special_tokens=True)

        logits = output_ids.to_tuple()[1]
        # Vocab is usually string:id, invert it for id:string
        vk = dict((v,k) for (k,v) in self.tokenizer.vocab.items())
        # The token IDs we're operating on
        tidxs = [np.where(~x[0].isinf())[0] for x in output_ids.to_tuple()[1]]
        tids = [[x[0][k].item() for k in idx] for (x,idx) in zip(output_ids.to_tuple()[1],tidxs)]
        import pdb
        pdb.set_trace()
        # Lookup has the vocab word for each non-infinite value in the logits
        #lookup = [[vk[k] for k in np.where(~x[0].isinf())[0]] for x in output_ids.to_tuple()[1]]
        lookup = [[vk[k] for k in x] for x in tidxs]
        # Argmax is not 100% correct here, the tokenizer makes some alternative decisions sometimes
        #highlight = [np.argmax(x[0][~x[0].isinf()]).item() for x in output_ids.to_tuple()[1]]
        highlight = [np.argmax(tid_list) for tid_list in tids]
        # Reverse-engineer what highlight SHOULD look like based on the generated text?

        # Convert logits list to a numpy array for easier analysis
        logits_matrix = np.array([logits.detach().cpu().numpy() for logits in self.logits_list])

        return generated_text, logits_matrix

    # Post-process generated text (customizable for different use cases)
    def post_process_output(self, generated_text: str):
        # Here you can add any custom processing you need (e.g., formatting, cleaning up special tokens)
        return generated_text.strip()

    # Example function to save logits to a file for later inspection
    def save_logits_to_file(self, logits_matrix: np.ndarray, filename: str):
        np.save(filename, logits_matrix)
        print(f"Logits saved to {filename}")

# Main function to tie everything together
def main(model_name: str, input_text: str, seeds: list, generation_config: GenerationConfig):
    hfi = HF_Interface(model_name)

    for seed in seeds:
        print(f"Generating text with seed: {seed}")
        hfi.set_seed(seed)

        # Generate text and logits
        generated_text, logits_matrix = hfi.generate_text_and_logits(
            input_text, generation_config)

        # Post-process the generated text (if needed)
        processed_text = hfi.post_process_output(generated_text)
        print(f"Generated Text with seed {seed}:\n{processed_text}")

        # Optionally save logits for inspection
        #hfi.save_logits_to_file(logits_matrix, f"logits_seed_{seed}.npy")

if __name__ == "__main__":
    # Parameters
    model_name = "meta-llama/Llama-3.1-8B-Instruct"  # Replace with your desired model
    input_text = "Once upon a time, in a distant land, there was a mysterious forest."
    seeds = [10648, 12345, 78910]  # Example seeds for varying outputs
    generation_config = GenerationConfig(
        temperature=0.7,
        top_p=0.95,
        num_beams=1,  # You can adjust this for different types of beam search
        do_sample=True,  # Ensures randomness, set to False for greedy search
        max_length=50,  # Adjust this as needed
        seed=None  # You can leave this out, as we handle the seed manually
    )
    print(generation_config)

    # Run the script
    main(model_name, input_text, seeds, generation_config)

