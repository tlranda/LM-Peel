import pandas as pd
import numpy as np

from peeled_huggingface import HF_Interface, build, parse

def datasets_load():
    train = pd.read_csv('training_data.csv')
    sm = pd.read_csv('all_SM_for_LLM.csv')
    xl = pd.read_csv('all_XL_for_LLM.csv')
    return train, sm, xl

def llm_template(df, objective_columns=None, with_answer=False, with_query_answer=False):
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
    default_help = "(default: %(default)s)"
    prs.add_argument('--n-ICL', type=int, default=10,
                     help=f"Number of ICL examples to include {default_help}")
    prs.add_argument('--one-class', action='store_true',
                     help=f"Don't use transfer from train->test, use all only test data in prompts {default_help}")
    prs.add_argument('--explain', action='store_true',
                     help=f"System Prompt encourages LLM to explain prior to answering {default_help}")
    prs.add_argument('--no-repeat', action='store_true',
                     help=f"System Prompt encourages LLM to not repeat prior values {default_help}")
    prs.add_argument('--objective-columns', required=True, nargs="+", action='append',
                     help=f"Objective columns for LLM to predict {default_help}")
    prs.add_argument('--plot', action='store_true',
                     help=f"Instead of text output, generate plot of possible outputs")
    return prs

def extend_prs(args):
    args.objective_columns = np.asarray(args.objective_columns).ravel().tolist()
    return args

def make_prompts(train, sm, xl, args):
    sys_prompt = \
"""The user may describe their optimization problem to give specific context.
Then they will demonstrate hyperparameter configurations for a regression problems in a feature-rich text-based CSV format.
Following each hyperparameter configuration, the user will either demonstrate the ground-truth value for the configuration or leave a blank for you to infer the objective.
Do not alter the user's proposed configuration.
"""
    if args.no_repeat:
        sys_prompt += \
"""Do not re-use performance values demonstrated by the user unless you expect the exact same performance as a configuration they show you.
Only provide a new performance value for configurations that the user gives you without an attached value.
"""
    if args.explain:
        sys_prompt += \
"""Explain your thought process, then conclude on a separate line with ONLY your answer following the format that the user demonstrated for you.
"""
    else:
        sys_prompt += \
"""Do NOT explain your thought process. ONLY respond with your answer following the format that the user demonstrated for you.
"""
    if args.one_class:
        df = sm
    else:
        df = train
    usr_prompt = llm_template(df.loc[:args.n_ICL],
                              objective_columns=args.objective_columns,
                              with_answer=True)
    if not args.one_class:
        df = sm
    usr_prompt += llm_template(df.loc[[args.n_ICL]],
                               objective_columns=args.objective_columns,
                               with_query_answer=True)
    usr_prompt = "\n".join(usr_prompt)
    return [{'role': 'system', 'content': sys_prompt},
            {'role': 'user', 'content': usr_prompt},
            ]

def main():
    args = parse(build_extend_fn=extend_build,
                 prs_extend_fn=extend_prs)
    train, sm, xl = datasets_load()
    # Make a prompt list
    prompts = make_prompts(train, sm, xl, args)
    print(prompts)
    model = HF_Interface(args.model_name, plot=args.plot)
    for seed in args.seeds:
        model.set_seed(seed)
        out = model.generate_text_and_logits(prompts,
                                             args.gen_config,
                                             None)
        print(f"Seed {seed} --> {out}")

if __name__ == '__main__':
    main()

