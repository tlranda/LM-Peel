# Pypi/environment manager of your choice for external package dependency
import pandas as pd
import numpy as np

# Local package dependency
from peeled_huggingface import HF_Interface, build, parse
from interactive_text_editor import chunker_with_cursor, text_trimmer

# Python3 Builtin
import pathlib

def datasets_load():
    """
        For now this is hard-coded, it should probably get controlled by argparse args at some point
    """
    train = pd.read_csv('training_data.csv')
    sm = pd.read_csv('all_SM_for_LLM.csv')
    xl = pd.read_csv('all_XL_for_LLM.csv')
    df = pd.concat([train,sm,xl]).reset_index(drop=True)
    return df

def llm_template(df, objective_columns=None, with_answer=False, with_query_answer=False):
    """
        Create the LLM-ready text representation of all data in df (DataFrame)
        * objective_columns = performance indicator if with_answer==True
        * with_answer = SHOW the answer
        * with_query_answer = SHOW prompt for answer, but withhold actual answer
    """
    if objective_columns is None:
        objective_columns = list()
    param_columns = [_ for _ in df.columns if _ not in objective_columns]
    n_params = len(param_columns)
    resp = []
    for (rowidx, row) in df.iterrows():
        config = "Hyperparameter configuration: "
        for idx, (col, val) in enumerate(row.to_dict().items()):
            if col in objective_columns:
                continue
            config += f"{col} is {val}"
            if idx < n_params-1:
                config += ","
        if with_answer:
            config += "\n"+f"Performance: ## {', '.join([str(_) for _ in row[objective_columns]])} ##"
        elif with_query_answer:
            config += "\nPerformance: "
        resp.append(config)
    return resp

def extend_build(prs):
    """
        Bonus args for driver here
    """
    dhelp = "(default: %(default)s)"
    req = "[REQUIRED]"
    # Evaluation settings
    eval_settings = prs.add_argument_group('Evaluation Settings')
    eval_settings.add_argument('--n-ICL', type=int, default=10,
                     help=f"Number of ICL examples to include {dhelp}")
    eval_settings.add_argument('--n-eval', type=int, default=1,
                     help=f"Number of evaluations to request {dhelp}")
    eval_settings.add_argument('--response-type', choices=['qualitative','quantitative'], required=True,
                     help=f"{req} Type of response the LLM should create {dhelp}")
    eval_settings.add_argument('--response-format', choices=['performance','configuration'], default='performance',
                     help=f"Pick how the LLM should respond {dhelp}")
    # Dataset handling
    dataset_settings = prs.add_argument_group('Dataset Handling')
    dataset_settings.add_argument('--class-column', default='size',
                     help=f"Which column is used to determine class {dhelp}")
    dataset_settings.add_argument('--ICL-classes', default=None, nargs="+", action='append', required=True,
                     help=f"{req} Classes to use for ICL examples")
    dataset_settings.add_argument('--eval-classes', default=None, nargs="+", action='append', required=True,
                     help=f"{req} Classes to use for evaluation")
    dataset_settings.add_argument('--objective-columns', required=True, nargs="+", action='append',
                     help=f"{req} Objective columns for LLM to predict {dhelp}")
    dataset_settings.add_argument('--objective-direction', choices=['minimize','maximize'], default='minimize',
                     help=f"Objectives are improved by moving this direction {dhelp}")
    # Tweak LLM instructions
    llm_instruct = prs.add_argument_group('LLM Instruction Tweaks')
    llm_instruct.add_argument('--problem-introduction', default=None,
                     help=f"Plaintext or path-to-plaintext-document with instructions about the user prompt {dhelp}")
    llm_instruct.add_argument('--qualitative-quantity', choices=['one','many','unstated'], default='one',
                     help=f"When response type is QUALITATIVE, request this many LLM responses {dhelp}")
    llm_instruct.add_argument('--explain', action='store_true',
                     help=f"System Prompt encourages LLM to explain prior to answering {dhelp}")
    llm_instruct.add_argument('--no-repeat', action='store_true',
                     help=f"System Prompt encourages LLM to not repeat prior ICL values for quantitative/performance evaluations {dhelp}")
    # Output settings
    out = prs.add_argument_group('Output Settings')
    out.add_argument('--in-text-editing', action='store_true',
                     help=f"Use in-Python editor instead of selecting an editor (or using editor indicated by environment's EDITOR) {dhelp}")
    out.add_argument('--plot', action='store_true',
                     help=f"Instead of text output, generate plot of possible outputs")
    return prs

def extend_prs(args):
    """
        Parse bonus args from extend_build() as needed
    """
    flatten = lambda x: np.asarray(x).ravel().tolist()
    for to_flat in ['objective_columns','ICL_classes','eval_classes']:
        setattr(args,to_flat,flatten(getattr(args,to_flat)))
    return args

def make_prompts(df, args):
    """
        Create system and user prompts for data based on argument values
        In particular, customizes for quantitative/qualitative, response format,
        special instructions like 'no repeat' and 'explain', all without mixing test/train data for ICL.

        Return the prompts AND whatever data is most relevant based on the prompt to evaluate the success of the LLM at its task
    """
    sys_prompt = \
"""The user may describe their optimization problem to give specific context.
Then they will demonstrate hyperparameter configurations for a regression problems in a feature-rich text-based CSV format.
"""
    if args.response_type == 'quantitative' and args.response_format == 'performance':
        sys_prompt += \
"""Following the examples, the user will provide a number of configurations without performance values; you will need to infer the objective based on their prior examples.
Do not alter the user's proposed configurations.
"""
    else: # qualitative or quantitative-configuration
        sys_prompt += \
"""Following the examples, the user will request that you propose a number of new configurations that may further improve the objective.
You only need to specify new configurations, the user will determine their performance.
"""
    if args.response_type == 'quantitative' and args.response_format == 'performance' and args.no_repeat:
        sys_prompt += \
"""Do not re-use performance values demonstrated by the user unless you expect the exact same performance as a configuration they show you.
Only provide a new performance value for configurations that the user gives you without an attached value.
"""
    if args.explain:
        sys_prompt += \
"""Explain your thought process, then conclude on separate lines with ONLY your answers following the format that the user demonstrated for you.
"""
    else:
        sys_prompt += \
"""Do NOT explain your thought process. ONLY respond with your answer following the format that the user demonstrated for you.
"""
    # Slap these together, then we'll disambiguate classes based on ICL/eval
    index_selector = df[args.class_column].astype(str)
    used_indices = []
    icl_bools = [(index_selector == icl_class) for icl_class in args.ICL_classes]
    while len(icl_bools) > 1:
        icl_bools[0] = np.logical_and(icl_bools[0],icl_bools[1])
    icl_conditions = icl_bools[0]
    eval_bools = [(index_selector == eval_class) for eval_class in args.eval_classes]
    while len(eval_bools) > 1:
        eval_bools[0] = np.logical_and(eval_bools[0], eval_bools[1])
    eval_conditions = eval_bools[0]
    icl_eligible = index_selector[icl_conditions].index.to_numpy()
    eval_eligible = index_selector[eval_conditions].index.to_numpy()
    # Determine the best results you could hope for
    oracle_order = np.argsort(df.loc[eval_eligible,args.objective_columns])
    if args.objective_direction == 'maximize':
        oracle_order = reversed(oracle_order)
    # Ravel or else it comes back as (1,X) shape
    best_eval_idx = eval_eligible[oracle_order].ravel()
    # Build prompts
    usr_prompt = []
    if args.problem_introduction is not None:
        if pathlib.Path(args.problem_introduction).exists():
            with open(args.problem_introduction,'r') as f:
                usr_prompt.append("".join(f.readlines()))
        else:
            usr_prompt.append(args.problem_introduction)
    if args.response_type == 'qualitative' and args.qualitative_quantity != 'unspecified':
        quantity_word = 'one' if args.qualitative_quantity == 'one' else 'several'
        usr_prompt.append(
f"""Please provide {quantity_word} candidate responses for each requested completion.
"""
        )

    usr_prompt.append(
"""Here are the examples:
"""
    )
    usr_prompt += llm_template(df.loc[icl_eligible[:args.n_ICL]],
                              objective_columns=args.objective_columns,
                              with_answer=True)
    # Drop any ICL eligible items that are INCLUDED in ICL prior to picking evaluations
    dropped_evals = set(eval_eligible).intersection(set(icl_eligible[:args.n_ICL]))
    if len(dropped_evals) > 0:
        best_eval_idx = [_ for _ in best_eval_idx if _ not in dropped_evals]
        eval_eligible = [_ for _ in eval_eligible if _ not in dropped_evals]
    usr_prompt.append(
"""Please complete the following:
"""
            )
    usr_prompt += llm_template(df.loc[eval_eligible[:args.n_eval]],
                               objective_columns=args.objective_columns,
                               with_query_answer=True)
    usr_prompt = "\n".join(usr_prompt)
    prompts = [{'role': 'system', 'content': sys_prompt},
               {'role': 'user', 'content': usr_prompt},
              ]
    # Returnable results
    if args.response_type == 'qualitative' or args.response_format == 'configuration':
        # Best results are based on distance to global best
        optimal_results = df.loc[best_eval_idx]
    else:
        # Best results are based on MAE/MSE vs ground truth
        optimal_results = df.loc[eval_eligible[:args.n_eval]]
    return prompts, optimal_results

def chunk_by_possibilities(text, possibilities):
    """
        Return list-of-lists(1-length) that can be concatenated to make 'text'
        based on the possibilities given to you (text<->possibility index match mapping)
    """
    space_indicator = 'Ġ'
    newline_indicator = 'Ċ'
    text_chunked = []
    text_index = 0
    assistant_text = text[text.index('assistant')+len('assistant')+2:]
    for chunk_list in possibilities:
        candidates = []
        #print(f"Process chunk: {chunk_list}")
        for beam_list in chunk_list:
            #print(f"Processe beam in chunk: {beam_list}")
            for option in beam_list:
                #print(f"Process option in beam: {option}")
                newopt = option.replace(space_indicator,' ').replace(newline_indicator,'\n')
                check = assistant_text[text_index:].startswith(newopt)
                if check:
                    # Cannot greedily full-match here, add to candidates and pick longest substring
                    #print(f"Candidate option: '{newopt}'")
                    candidates.append(newopt)
        # Pick candidate across beams by max-length match (most likely to be correct?)
        if len(candidates) > 0:
            newopt = candidates[np.argmax(list(map(len,candidates)))]
            #print(f"Accept option: '{newopt}'")
            text_chunked.append(newopt)
            text_index += len(newopt)
        # May warn about no matches, but this is likely an EOS token or similar
    return text_chunked

def possibilities_by_chunks(oldtext_map, new_text, possibilities):
    """
        Return list-of-lists(any-length) that can be concatenated by chunk_by_possibilities() to make new_text, based on oldtext_map's mappings (subset of input possibilities)
    """
    newtext_index = 0
    new_poss = []
    for (oldtext, oldposs) in zip(oldtext_map, possibilities):
        if new_text[newtext_index:].startswith(oldtext):
            new_poss.append(oldposs)
            newtext_index += len(oldtext)
    return new_poss

def user_trim_response(text, possibilities, args):
    """
        Use interactive editor to cull out portions of the final text that aren't relevant to evaluation here, but also associate that culling to the possibilities which is not covered by interactive editor directly
    """
    old_text_chunked_by_possibilities = chunk_by_possibilities(text, possibilities)
    if args.in_text_editing:
        ite = text_trimmer(text[text.index('assistant')+len('assistant')+2:])
        ite.cursor_marktext()
        new_text = ite.mask()
    else:
        tmp = pathlib.Path('tmp_llm_response.txt')
        version_number = 0
        while tmp.exists():
            tmp = tmp.with_stem(f'tmp_llm_response_{version_number}')
            version_number += 1
        with open(tmp,'w') as f:
            f.write("".join(text))
        import os
        import subprocess
        if 'EDITOR' in os.environ:
            editor_bin = os.environ['EDITOR']
        else:
            # Try to find a suitable editor
            maybe_editors = ['vi','vim','neovim','emacs','nano','gedit']
            found = {}
            for editor in maybe_editors:
                resp = subprocess.run(['which',editor], capture_output=True)
                if len(resp.stdout) > 0:
                    found[editor] = resp.stdout.decode().rstrip()
            ite = text_trimmer("\n".join(found.keys()))
            ite.cursor_marktext(instructions="Mark ONE editor you'd like to use (or use the EDITOR environment variable in the future to skip)")
            key = ite.mask(invert=True).rstrip()
            editor_bin = found[key]
        subprocess.call([editor_bin, tmp])
        with open(tmp,'r') as f:
            new_text = "".join(f.readlines())
    new_possibilities = possibilities_by_chunks(old_text_chunked_by_possibilities, new_text, possibilities)
    return new_text.rstrip(), new_possibilities

def get_number_fields(possibilities):
    """
        Use possibilities to generate all possible numeric outputs and return them
        as Depth-first-search
    """
    fielded = []
    n_beams = len(possibilities[0])
    # Each beam should be fielded separately
    import pdb
    pdb.set_trace()
    for beam in range(n_beams):
        n_opts = list(map(lambda x: len(x[beam]), possibilities))
        progress = [0] * len(n_opts)
    return fielded

def get_config_search(text, dataset):
    """
        Try to convert text into an entry in dataset, return None if it seems to
        be a nonexistent thing
    """
    return None

def main():
    args = parse(build_extend_fn=extend_build,
                 prs_extend_fn=extend_prs)
    # Make a prompt list
    dataset = datasets_load()
    prompts, optimal_results = make_prompts(dataset, args)
    print(f"System Prompt: {prompts[0]['content']}")
    print(f"User Prompt: {prompts[1]['content']}")
    model = HF_Interface(args.model_name, plot=args.plot)
    for seed in args.seeds:
        model.set_seed(seed)
        text, response_possibilities = model.generate_text_and_logits(prompts,
                                                                      args.gen_config,
                                                                      None, # Logits processor
                                                                      )
        print(f"Seed {seed} --> {text}")
        # TODO: Evaluate success
        print(f"Ground Truth: {optimal_results}")
        text, response_possibilities = user_trim_response(text, response_possibilities, args)
        if text is None:
            print(f"Sorry for bad LLM output :(")
            continue
        if args.response_type == 'quantitative' and args.response_format == 'performance':
            number_fields = get_number_fields(response_possibilities)
            print(number_fields)
        elif args.response_format == 'configuration':
            configs = get_config_search(text, dataset)
            print(configs)

if __name__ == '__main__':
    main()

