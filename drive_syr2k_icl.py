# Pypi/environment manager of your choice for external package dependency
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Local package dependency
from peeled_huggingface import HF_Interface, build, parse
from interactive_text_editor import chunker_with_cursor, text_trimmer, edit_via_editor
from pickle_cache import PickleCache
from timerdict import TimerDict

# Python3 Builtin
import itertools
import multiprocessing
import pathlib
import time

def extend_build(prs):
    """
        Bonus args for driver here
    """
    dhelp = "(default: %(default)s)"
    req = "[REQUIRED]"

    # Dataset handling
    dataset_settings = prs.add_argument_group('Dataset Handling: Problem definition and data interfacing')
    dataset_settings.add_argument('--dataset-dir', default=pathlib.Path('Datasets'), type=pathlib.Path,
                     help=f"All CSVs in this directory are combined for the dataset to use {dhelp}")
    dataset_settings.add_argument('--dataset-shuffle-seed', default=None, type=int,
                     help=f"Seed used for dataset shuffling (performed once on load, default: first argument to --seed)")
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
    dataset_settings.add_argument('--n-ICL', type=int, default=10,
                     help=f"Number of ICL examples to include {dhelp}")
    dataset_settings.add_argument('--n-eval', type=int, default=1,
                     help=f"Number of evaluations to request {dhelp}")
    dataset_settings.add_argument('--n-rounds', type=int, default=1,
                     help=f"Number of rounds of datasetuations to request from the dataset (disjoint ICLs on same seeds) {dhelp}")
    dataset_settings.add_argument('--skip-rounds', type=int, default=None, nargs="*", action='append',
                     help=f"Round indices (0-base indexed) to skip {dhelp}")

    # Tweak LLM instructions
    llm_instruct = prs.add_argument_group('LLM Prompting: Instruction tweaks')
    llm_instruct.add_argument('--response-type', choices=['qualitative','quantitative'], required=True,
                     help=f"{req} Type of response the LLM should create {dhelp}")
    llm_instruct.add_argument('--response-format', choices=['performance','configuration'], default='performance',
                     help=f"Pick how the LLM should respond {dhelp}")
    llm_instruct.add_argument('--problem-introduction', default=None,
                     help=f"Plaintext or path-to-plaintext-document with instructions about the user prompt {dhelp}")
    llm_instruct.add_argument('--qualitative-quantity', choices=['one','many','unstated'], default='one',
                     help=f"When response type is QUALITATIVE, request this many LLM responses {dhelp}")
    llm_instruct.add_argument('--explain', action='store_true',
                     help=f"System Prompt encourages LLM to explain prior to answering {dhelp}")
    llm_instruct.add_argument('--no-repeat', action='store_true',
                     help=f"System Prompt encourages LLM to not repeat prior ICL values for quantitative/performance evaluations {dhelp}")
    llm_instruct.add_argument('--scientific-notation', action='store_true',
                     help=f"Express numeric values for performance in scientific notation {dhelp}")
    llm_instruct.add_argument("--show-prompts", action='store_true',
                     help=f"Show the generated prompts before serving them {dhelp}")

    # Handle LLM responses
    llm_response = prs.add_argument_group('LLM Response: Handling text from LLM')
    llm_response.add_argument('--llm-cache', default=None, type=pathlib.Path,
                     help=f"Cache file to store/recall LLM response/pruning (default: No cache)")
    llm_response.add_argument('--quantity-cache', default=None, type=pathlib.Path,
                     help=f"Cache file to store/recall number fields from LLM generation (default: No cache)")
    llm_response.add_argument('--in-text-editing', action='store_true',
                     help=f"Use in-Python editor instead of selecting an editor (or using editor indicated by environment's EDITOR) {dhelp}")

    # Plot settings
    plot_settings = prs.add_argument_group('Plot Settings: Visualizations')
    plot_settings.add_argument('--no-plot', action='store_true',
                     help=f"Do not produce plots {dhelp}")
    plot_settings.add_argument('--title', default=None,
                     help=f"Title to use in generated plots {dhelp}")
    plot_settings.add_argument('--export', default=None, type=pathlib.Path,
                     help=f"Filename to save plot_settingsput files to (default: Interactive display)")
    plot_settings.add_argument('--override', action='store_true',
                     help=f"Override existing files on export {dhelp}")
    plot_settings.add_argument('--llm-range-only', action='store_true',
                     help=f"Limit axes to LLM-generated values only {dhelp}")

    # Analysis settings
    analysis_settings = prs.add_argument_group('Analysis Settings: Textual analyses')
    analysis_settings.add_argument('--highest-variation-only', action='store_true',
                     help=f"Only explore variations in highest variable tokens {dhelp}")
    analysis_settings.add_argument('--haystack-error', default=None, action='append', type=float, nargs="*",
                     help=f"Error bound (as ratios) for a 'needle' generated value to be 'pickable' in the 'haystack' of generable numbers {dhelp}")
    return prs

def extend_prs(args):
    """
        Parse bonus args from extend_build() as needed
    """
    flatten = lambda x: np.asarray(x).ravel().tolist()
    for to_flat in ['objective_columns','ICL_classes','eval_classes','skip_rounds','haystack_error']:
        setattr(args,to_flat,flatten(getattr(args,to_flat)))
    return args

def datasets_load(args):
    """
        For now this is hard-coded, it should probably get controlled by argparse args at some point
    """
    df = pd.concat([pd.read_csv(_) for _ in pathlib.Path(args.dataset_dir).iterdir() if _.suffix == '.csv'])
    # Sample so the LLM doesn't see sequential rows of data 100% of the time
    # Use very first seed to guarantee replicability of the results, unless
    # the user gave a particular seed for shuffling (ie: re-investigating a
    # later seed without running the former seed(s))
    shuffle_seed = args.seeds[0]
    if args.dataset_shuffle_seed is not None:
        shuffle_seed = args.dataset_shuffle_seed
    return df.sample(frac=1, random_state=shuffle_seed).reset_index(drop=True)

def llm_template(df, objective_columns=None, with_answer=False, with_query_answer=False, scientific=False):
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
            if scientific:
                performance = ', '.join([f"{_:.6E}" for _ in row[objective_columns]])
            else:
                performance = ', '.join([f"{_}" for _ in row[objective_columns]])
            config += "\n"+f"Performance: ## {performance} ##"
        elif with_query_answer:
            config += "\nPerformance: "
        resp.append(config)
    return resp

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
                                  with_answer=True, scientific=args.scientific_notation)
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
                                   with_query_answer=True, scientific=args.scientific_notation)
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
            new_text = ite.mask(invert=True)
    else:
        new_text = edit_via_editor(text, tmp='tmp_llm_response.txt')
    new_possibilities, new_logits = possibilities_by_chunks(old_text_chunked_by_possibilities,
                                                            new_text,
                                                            possibilities,
                                                            logits)
    return new_text.rstrip(), new_possibilities, new_logits

def generate_beam_numbers(beam_and_log):
    """
        Parallelizable version -- especially important when there are millions of combinations
    """
    within_beam, within_log = beam_and_log
    string_num = "".join(within_beam)
    weight = sum([particle.item() for particle in within_log])
    try:
        num = float(string_num)
    except:
        return None
    return num, weight

def get_number_fields(possibilities, logits, highest_variation_only):
    """
        Use possibilities to generate all possible numeric outputs and return them
        as Depth-first-search
    """
    n_possibilities = list(map(lambda x: max(map(len,x)),possibilities))
    total_poss = np.prod(n_possibilities)
    print(f"N_possibilities per token {n_possibilities} (total={total_poss})")
    if not highest_variation_only and total_poss > 250_000_000:
        print(f"That's a bit too much; how about just the highest variation this time")
        highest_variation_only = True
    if highest_variation_only:
        # Find the one with the most variation and only run that
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
                # Scientific notation permits 'E+' and 'E-' in addition to decimal point
                considerable = [idx3 for (idx3,v) in enumerate(pp) if v in '.-+E' or intable(v)]
                best = np.asarray(considerable)[np.argmax(np.asarray(ll)[considerable])]
                other_possibilities.append([[pp[best]]])
                other_logs.append([ll[[best]]])
        per_beam = list(itertools.product(*other_possibilities))
        per_log = list(itertools.product(*other_logs))
    else:
        per_beam = list(itertools.product(*possibilities))
        per_log = list(itertools.product(*logits))

    all_numbers = []
    weights = []

    for beam_id, (beam, logs) in enumerate(zip(per_beam, per_log)):
        within_beam = list(itertools.product(*beam))
        within_log  = list(itertools.product(*logs))
        with multiprocessing.Pool() as pool:
            results = pool.map(generate_beam_numbers, zip(within_beam, within_log))
        for rval in results:
            if rval is None:
                continue
            number, weight = rval
            all_numbers.append(number)
            weights.append(weight)

    """
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
    """
    return all_numbers, weights

def maybe_cached(model, seed, roundidx, td, to_model, cache, cache_key, args, ignore_cache=False, reopen=False):
    """
        Create model if necessary (no cache or no entry in cache)
        Let the user trim a model response and update the cache
        Return updated model and cache-able response

        ignore_cache = pretend there isn't a cache entry even if there is
        reopen = re-time this interval to not die to TimerDict implementation
    """
    if cache is not None and not ignore_cache:
        if cache_key in cache:
            # Not caught: og_text, og_response_possibilities, og_logits
            (text, response_possibilities, logits, *_) = cache[cache_key]
            del _
            return model, text, response_possibilities, logits
    if model is None:
        model = HF_Interface(args.model_name)
    model.set_seed(seed)
    td_1 = ('generate_with_logits',f'Round: {roundidx}',f'Seed: {seed}')
    td_2 = ('user_trim',f'Round: {roundidx}',f'Seed: {seed}')
    if reopen:
        td.reopen(td_1)
        td.reopen(td_2)
    td[td_1]
    og_text, og_response_possibilities, og_logits = model.generate_text_and_logits(*to_model)
    td[td_1]
    td[td_2]
    text, response_possibilities, logits = user_trim_response(og_text,
                                                              og_response_possibilities,
                                                              og_logits,
                                                              args)
    td[td_2]
    if cache is not None:
        cache[cache_key] = (text, response_possibilities, logits,
                            og_text, og_response_possibilities, og_logits)
        cache.to_pickle()
    return model, text, response_possibilities, logits

def request_retry(instructions=None):
    ite = text_trimmer("Mark anywhere to retry")
    ite.cursor_marktext(instructions)
    key = ite.mask(invert=True).rstrip()
    return len(key) > 0

def get_config_search(text, dataset):
    """
        Try to convert text into an entry in dataset, return None if it seems to
        be a nonexistent thing
    """
    return None

def main():
    args = parse(build_extend_fn=extend_build,
                 prs_extend_fn=extend_prs)
    td = TimerDict()
    td['all_runtime']
    # Make a prompt list using FIXED seed to load/shuffle data
    td['dataset_prep']
    dataset = datasets_load(args)
    td['dataset_prep']
    td['cache_loads']
    if args.llm_cache is None:
        llm_cache = None
    else:
        llm_cache = PickleCache(args.llm_cache)
    if args.quantity_cache is None:
        quantity_cache = None
    else:
        quantity_cache = PickleCache(args.quantity_cache)
    td['cache_loads']
    model = None
    td['make_prompts']
    rounds = make_prompts(dataset, args)
    td['make_prompts']
    errors = []
    rel_errors = []
    copied = []
    possibly_copied = []
    n_haystack = []
    for (roundidx, _round) in enumerate(rounds):
        td_round_key = ('round',f'Round: {roundidx}')
        td[td_round_key]
        prompts, icl_values, optimal_results = _round
        if roundidx in args.skip_rounds:
            td[td_round_key]
            continue
        if args.show_prompts:
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
            hashable_prompts = make_hashable_prompts(to_model[0])
            cache_key = tuple([args.model_name, seed, hashable_prompts]+to_model[1:])
            # Collect answer with parsing
            cached = False
            validated = False
            td_loop_key = ('llm',f'Round: {roundidx}',f'Seed: {seed}')
            while True:
                if cached:
                    td.reopen(td_loop_key)
                td[td_loop_key]
                model, text, response_possibilities, logits = maybe_cached(model,
                                                                           seed,
                                                                           roundidx,
                                                                           td,
                                                                           to_model,
                                                                           llm_cache,
                                                                           cache_key,
                                                                           args,
                                                                           ignore_cache=cached,
                                                                           reopen=cached)
                cached = True
                # Validation -- break if not validated and not fixing, continue
                # if not validated and we'll try again. Do nothing to fall
                # through and validate/exit the infinite loop
                if text is None:
                    if args.in_text_editing:
                        td[td_loop_key]
                        if request_retry("LLM did not produce a response"):
                            continue
                        else:
                            break
                    else:
                        print("No response from LLM")
                        td[td_loop_key]
                        break
                if args.response_type == 'quantitative' and args.response_format == 'performance':
                    try:
                        gen_number = float(text)
                    except:
                        if args.in_text_editing:
                            td[td_loop_key]
                            if request_retry(f"LLM did not produce a number: '{text}'"):
                                continue
                            else:
                                break
                        else:
                            print("No number from LLM")
                            td[td_loop_key]
                            break
                td[td_loop_key]
                validated = True
                break
            if not validated:
                continue
            print(f"Seed {seed} --> {text}")
            if args.response_type == 'quantitative' and args.response_format == 'performance':
                td_quant_key = ('quantitative_perf_analysis',f'Round: {roundidx}',f'Seed: {seed}')
                td[td_quant_key]
                # Have to push bounds check here in case highest variability excludes the sampled value
                if llm_min > gen_number:
                    llm_min = gen_number
                if llm_max < gen_number:
                    llm_max = gen_number
                optimal_number = optimal_results.loc[optimal_results.index[0], args.objective_columns].item()
                errors.append(optimal_number-gen_number)
                rel_errors.append((optimal_number-gen_number)/optimal_number)
                copied.append(gen_number in icl_values.to_numpy())
                td_number_field_key = ('number_fields',f'Round: {roundidx}',f'Seed: {seed}')
                td[td_number_field_key]
                if quantity_cache is None:
                    generable_numbers, weight = get_number_fields(response_possibilities, logits, args.highest_variation_only)
                else:
                    if cache_key not in quantity_cache:
                        generable_numbers, weight = get_number_fields(response_possibilities, logits, args.highest_variation_only)
                        quantity_cache[cache_key] = (generable_numbers, weight)
                        quantity_cache.to_pickle()
                    else:
                        generable_numbers, weight = quantity_cache[cache_key]
                td[td_number_field_key]
                normalized_weight = np.asarray(weight).ravel()
                normalized_weight -= min(normalized_weight)
                normalized_weight /= max(normalized_weight)
                generable_numbers = np.asarray(generable_numbers).ravel()
                if args.haystack_error[0] is not None:
                    for error_bound in args.haystack_error:
                        n_haystack.append(len(np.where((np.abs(optimal_number-generable_numbers)/optimal_number) <= error_bound)[0]))
                        print(f"Haystack needles @{error_bound}: {n_haystack[-1]}")
                print(f"Range of generable values: {min(generable_numbers)}, {max(generable_numbers)}")
                print(f"Median generated value: {np.median(generable_numbers)}")
                possibly_copied.append(0)
                for icl_check in icl_values.to_numpy().ravel():
                    if icl_check == gen_number:
                        possibly_copied[-1] += 1
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
                if not args.no_plot:
                    td_plot_key = ('plot',f'Round: {roundidx}',f'Seed: {seed}')
                    td[td_plot_key]
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
                        sampled_logit = normalized_weight[sort[sampled_idx]]
                    except:
                        print(f"Failed to find exact sampling match, using 1.0 height default for sampled value")
                        #sampled_idx = np.argmax(normalized_weight)
                        sampled_logit = 1.0
                    ax.scatter(float(text), sampled_logit,
                               label=f'Sampled response {seed}', marker='+', s=400,
                               color=resps.get_facecolor())
                    td[td_plot_key]
                td[td_quant_key]
            elif args.response_format == 'configuration':
                configs = get_config_search(text, dataset)
                print(configs)
        if not args.no_plot:
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
        td[td_round_key]
    print(f"Across {args.n_rounds} rounds, accumulated errors:")
    print("\t"+f"MAE: {np.mean(np.abs(errors))}")
    print("\t"+f"MSE: {np.mean(np.asarray(errors)**2)}")
    print(f"As relative ratio errors:")
    print("\t"+f"MAE: {np.mean(np.abs(rel_errors))}")
    print("\t"+f"MSE: {np.mean(np.asarray(rel_errors)**2)}")
    print(f"# Copied answers: {np.sum(copied)}")
    print(f"# Possible Copied answers: {np.sum(possibly_copied)}")
    td['all_runtime']
    print(td.dump())
    if llm_cache is not None:
        import datetime
        now = datetime.datetime.now()
        llm_cache[f"Arguments for execution @ {now}"] = args
        llm_cache[f"Timings for execution @ {now}"] = td.dump()
        llm_cache.to_pickle()

if __name__ == '__main__':
    main()

