# Pypi/environment manager of your choice for external package dependency
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Local package dependency
from peeled_huggingface import HF_Interface, build, parse
from interactive_text_editor import chunker_with_cursor, text_trimmer, edit_via_editor
from pickle_cache import PickleCache

# Python3 Builtin
import itertools
import pathlib

def datasets_load(args):
    """
        For now this is hard-coded, it should probably get controlled by argparse args at some point
    """
    train = pd.read_csv('training_data.csv')
    sm = pd.read_csv('all_SM_for_LLM.csv')
    xl = pd.read_csv('all_XL_for_LLM.csv')
    # Sample so the LLM doesn't see sequential rows of data 100% of the time
    # Use very first seed to guarantee replicability of the results
    df = pd.concat([train,sm,xl]).sample(frac=1, random_state=args.seeds[0]).reset_index(drop=True)
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
    eval_settings.add_argument('--n-rounds', type=int, default=1,
                     help=f"Number of rounds of evaluations to request from the dataset (disjoint ICLs on same seeds) {dhelp}")
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
    out.add_argument('--highest-variation-only', action='store_true',
                     help=f"Only explore variations in highest variable tokens {dhelp}")
    out.add_argument('--title', default=None,
                     help=f"Title to use in generated plots {dhelp}")
    out.add_argument('--export', default=None, type=pathlib.Path,
                     help=f"Filename to save output files to (default: Interactive display)")
    out.add_argument('--override', action='store_true',
                     help=f"Override existing files on export {dhelp}")
    out.add_argument('--cache', default=None, type=pathlib.Path,
                     help=f"Cache file to store/recall LLM response/pruning (default: No cache)")
    out.add_argument('--llm-range-only', action='store_true',
                     help=f"Limit axes to LLM-generated values only {dhelp}")
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
        special instructions like 'no repeat' and 'explain', all without mixing
        test/train data for ICL.

        Return the prompts, prompt-objectives, AND whatever data is most
        relevant based on the prompt to evaluate the success of the LLM at its
        task
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
    rounds = []
    index_selector = df[args.class_column].astype(str)
    # Compress booleans down to full ICL-class match
    icl_bools = [(index_selector == icl_class) for icl_class in args.ICL_classes]
    while len(icl_bools) > 1:
        icl_bools[0] = np.logical_and(icl_bools[0],icl_bools[1])
    icl_conditions = icl_bools[0]
    # Compress booleans down to full eval-class match
    eval_bools = [(index_selector == eval_class) for eval_class in args.eval_classes]
    while len(eval_bools) > 1:
        eval_bools[0] = np.logical_and(eval_bools[0], eval_bools[1])
    eval_conditions = eval_bools[0]
    used_icl, used_eval = [], []
    for _round in range(args.n_rounds):
        # Select data independently between rounds
        icl_eligible = index_selector[icl_conditions].index.to_numpy()
        icl_eligible = np.array([_ for _ in icl_eligible if _ not in used_icl])
        eval_eligible = index_selector[eval_conditions].index.to_numpy()
        eval_eligible = np.array([_ for _ in eval_eligible if _ not in used_eval])
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
        # ICL selection
        used_icl.extend([_ for _ in icl_eligible[:args.n_ICL]])
        usr_prompt += llm_template(df.loc[icl_eligible[:args.n_ICL]],
                                  objective_columns=args.objective_columns,
                                  with_answer=True)
        prompt_objective = df.loc[icl_eligible[:args.n_ICL],args.objective_columns]
        # Drop any ICL eligible items that are INCLUDED in ICL prior to picking evaluations
        dropped_evals = set(eval_eligible).intersection(set(icl_eligible[:args.n_ICL]))
        if len(dropped_evals) > 0:
            best_eval_idx = [_ for _ in best_eval_idx if _ not in dropped_evals]
            eval_eligible = [_ for _ in eval_eligible if _ not in dropped_evals]
        usr_prompt.append(
"""Please complete the following:
"""
        )
        used_eval.extend([_ for _ in eval_eligible[:args.n_eval]])
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
        rounds.append([prompts, prompt_objective, optimal_results])
    return rounds

def make_hashable_prompts(prompts):
    """
        Dictionary cannot be directly hashed, make it hashable for use in caches
    """
    out = []
    for prompt in prompts:
        out.append(tuple([(k,v) for (k,v) in prompt.items()]))
    return tuple(out)

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

def possibilities_by_chunks(oldtext_map, new_text, possibilities, logits):
    """
        Return list-of-lists(any-length) that can be concatenated by chunk_by_possibilities() to make new_text, based on oldtext_map's mappings (subset of input possibilities)
    """
    newtext_index = 0
    new_poss = []
    new_logits = []
    for (oldtext, oldposs, oldlog) in zip(oldtext_map, possibilities, logits):
        if new_text[newtext_index:].startswith(oldtext):
            new_poss.append(oldposs)
            new_logits.append(oldlog)
            newtext_index += len(oldtext)
    return new_poss, new_logits

def user_trim_response(text, possibilities, logits, args):
    """
        Use interactive editor to cull out portions of the final text that aren't relevant to evaluation here, but also associate that culling to the possibilities which is not covered by interactive editor directly
    """
    old_text_chunked_by_possibilities = chunk_by_possibilities(text, possibilities)
    if args.in_text_editing:
        ite = text_trimmer(text[text.index('assistant')+len('assistant')+2:])
        new_text = ite.cursor_marktext(instructions="CTRL+e if you need a proper editor")
        if new_text is None:
            new_text = ite.mask()
    else:
        new_text = edit_via_editor(text, tmp='tmp_llm_response.txt')
    new_possibilities, new_logits = possibilities_by_chunks(old_text_chunked_by_possibilities,
                                                            new_text,
                                                            possibilities,
                                                            logits)
    return new_text.rstrip(), new_possibilities, new_logits

def get_number_fields(possibilities, logits, highest_variation_only):
    """
        Use possibilities to generate all possible numeric outputs and return them
        as Depth-first-search
    """
    all_numbers = []
    weights = []
    if highest_variation_only:
        # Find the one with the most variation and only run that
        n_possibilities = list(map(lambda x: max(map(len,x)),possibilities))
        still_variable = np.argmax(n_possibilities)
        other_possibilities = []
        other_logs = []
        # Sometimes the highest logit value for non-variable places would break numbers -- don't do that
        def intable(v):
            try:
                int(v)
            except:
                return False
            return True
        for (idx, (p,l)) in enumerate(zip(possibilities, logits)):
            if idx == still_variable:
                other_possibilities.append(p)
                other_logs.append(l)
                continue
            for (idx2, (pp, ll)) in enumerate(zip(p,l)):
                considerable = [idx3 for (idx3,v) in enumerate(pp) if v == '.' or intable(v)]
                best = np.asarray(considerable)[np.argmax(np.asarray(ll)[considerable])]
                other_possibilities.append([[pp[best]]])
                other_logs.append([ll[[best]]])
        per_beam = list(itertools.product(*other_possibilities))
        per_log = list(itertools.product(*other_logs))
    else:
        per_beam = list(itertools.product(*possibilities))
        per_log = list(itertools.product(*logits))
    for beam_id, (beam, logs) in enumerate(zip(per_beam, per_log)):
        within_beam = list(itertools.product(*beam))
        within_log  = list(itertools.product(*logs))
        string_nums = ["".join(particle) for particle in within_beam]
        # Somehow map the particles within a beam to their logit values
        particle_sequence = [sum(particle).item() for particle in within_log]
        beam_numbers = []
        beam_weight = []
        for string, weight in zip(string_nums, particle_sequence):
            try:
                num = float(string)
            except:
                continue
            beam_numbers.append(num)
            beam_weight.append(weight)
        all_numbers.append(beam_numbers)
        weights.append(beam_weight)
    return all_numbers, weights

def get_config_search(text, dataset):
    """
        Try to convert text into an entry in dataset, return None if it seems to
        be a nonexistent thing
    """
    return None

def main():
    args = parse(build_extend_fn=extend_build,
                 prs_extend_fn=extend_prs)
    # Make a prompt list using FIXED seed to load/shuffle data
    dataset = datasets_load(args)
    if args.cache is None:
        cache = None
    else:
        cache = PickleCache(args.cache)
    model = None
    rounds = make_prompts(dataset, args)
    for (roundidx, _round) in enumerate(rounds):
        prompts, icl_values, optimal_results = _round
        print(f"System Prompt: {prompts[0]['content']}")
        print(f"User Prompt: {prompts[1]['content']}")
        print(f"Ground Truth:")
        print(optimal_results)
        to_model = [prompts,
                    args.gen_config,
                    None, # Logits processor
                   ]
        fig, ax = None, None
        llm_min, llm_max = None, None
        if args.response_type == 'quantitative':
            llm_min, llm_max = [_.item() for _ in optimal_results.loc[optimal_results.index[0], args.objective_columns]]*2
        for seed in args.seeds:
            cached = False
            hashable_prompts = make_hashable_prompts(to_model[0])
            cache_key = tuple([args.model_name, seed, hashable_prompts]+to_model[1:])
            if cache is not None:
                if cache_key in cache:
                    (text, response_possibilities, logits,
                     og_text, og_response_possibilities, og_logits) = cache[cache_key]
                    cached = True
            if not cached:
                if model is None:
                    model = HF_Interface(args.model_name)
                model.set_seed(seed)
                og_text, og_response_possibilities, og_logits = model.generate_text_and_logits(*to_model)
                text, response_possibilities, logits = user_trim_response(og_text,
                                                                          og_response_possibilities,
                                                                          og_logits,
                                                                          args)
                if cache is not None:
                    cache[cache_key] = (text, response_possibilities, logits,
                                        og_text, og_response_possibilities, og_logits)
                    cache.to_pickle()
            if text is None:
                print(f"Sorry for bad LLM output :(")
                continue
            print(f"Seed {seed} --> {text}")
            if args.response_type == 'quantitative' and args.response_format == 'performance':
                generable_numbers, weight = get_number_fields(response_possibilities, logits, args.highest_variation_only)
                normalized_weight = np.asarray(weight).ravel()
                normalized_weight -= min(normalized_weight)
                normalized_weight /= max(normalized_weight)
                generable_numbers = np.asarray(generable_numbers).ravel()
                cand_min = min(generable_numbers)
                cand_max = max(generable_numbers)
                if llm_min is None:
                    llm_min = cand_min
                    llm_max = cand_max
                else:
                    if llm_min > cand_min:
                        llm_min = cand_min
                    if llm_max < cand_max:
                        llm_max = cand_max
                sort = np.argsort(generable_numbers)
                max_height = max(normalized_weight)
                if fig is None:
                    fig, ax = plt.subplots(figsize=(12,6))
                    ax.set_xlabel("Number generated")
                    ax.set_ylabel("Normalized likelihood of text generation")
                    # Plot ICL as vlines
                    ax.vlines(icl_values.to_numpy().ravel(), ymin=0.0, ymax=1.0,
                              alpha=0.5, color='k', zorder=-1,
                              label="ICL Values")
                    # Plot Ground Truth
                    ax.vlines(optimal_results.loc[optimal_results.index[0], args.objective_columns],
                              ymin=0.0, ymax=1.0, color='y',
                              zorder=0, label=f"Ground truth seed {seed}")
                    ax.scatter(optimal_results.loc[optimal_results.index[0], args.objective_columns],
                               0.0, marker='x', s=200, color='y', zorder=0)
                resps = ax.scatter(generable_numbers[sort], normalized_weight[sort],
                                   alpha=0.6, s=4,
                                   label=f'{args.model_name} Seed {seed}')
                try:
                    sampled_idx = np.argwhere(generable_numbers == float(text))[0,0]
                    sampled_idx = sort.tolist().index(sampled_idx)
                except:
                    print(f"Failed to find exact sampling match, using argmax")
                    sampled_idx = np.argmax(normalized_weight)
                ax.scatter(float(text), normalized_weight[sort[sampled_idx]],
                           label=f'Sampled response {seed}', marker='+', s=400,
                           color=resps.get_facecolor())
            elif args.response_format == 'configuration':
                configs = get_config_search(text, dataset)
                print(configs)
        if args.title is not None:
            ax.set_title(args.title)
        if args.llm_range_only:
            ax.set_xlim((0.95*llm_min, 1.05*llm_max))
        ax.legend(loc='best')
        fig.set_tight_layout(True)
        if args.export is None:
            plt.show()
        else:
            export = args.export.with_stem(f"{args.export.stem}_round_{roundidx}")
            if export.exists() and not args.override:
                idx = 0
                while export.with_stem(export.stem+f"_{idx}").exists():
                    idx += 1
                new_export = export.with_stem(args.export.stem+f"_{idx}")
                print(f"{export} already exists! Moving output to {new_export} instead!")
                export = new_export
            fig.savefig(export, dpi=300)
            print(f"Figure saved to {export}")

if __name__ == '__main__':
    main()

