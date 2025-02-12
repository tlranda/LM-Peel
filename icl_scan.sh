#!/bin/bash

n_icl=( 1 5 10 50 100 );
icl_f=( "001" "005" "010" "050" "100" );
size="XL";

# Set parallel in environment
export TOKENIZERS_PARALLELISM=true;
for i in "${!icl_f[@]}"; do
    trythis=("python3 drive_syr2k_icl.py "
             # -- DISTRIBUTION EXPLORATION --
             # Seeds control randomness on repeated prompt/data combinations
             "--seed 1234 2024 9999 "

             # -- DATA HANDLING --
             "--dataset-dir Datasets/syr2k "
             # --disable-- "--dataset-shuffle-seed 1234 "
             "--class-column size "
             "--ICL-classes ${size} --eval-classes ${size} "
             "--objective-columns runtime "
             "--objective-direction minimize "
             "--n-ICL ${n_icl[$i]} "
             "--n-eval 1 "
             # Number of data sets to sample
             "--n-rounds 5 "
             # Skip 0-indexed rounds to fast-forward progress to a particular dataset
             # --disable-- "--skip-rounds "

             # -- LLM PROMPTING --
             # Only support these currently
             "--response-type quantitative "
             "--response-format performance "
             # Define the problem for LLM
             "--problem-introduction Prompts/syr2k_${size}_intro.txt "
             # --disable-- "--qualitative-quantity one " # --NOT IMPLEMENTED--
             # --disable-- "--explain "
             # --disable-- "--no-repeat "
             # --disable-- "--scientific-notation "

             # -- HANDLE LLM RESPONSES --
             # Recall previously answered / pruned values so we don't waste LLM usage
             "--llm-cache Caches/syr2k_${size}.pkl "
             "--quantity-cache Caches/number_fields_syr2k_${size}.pkl "
             # Use the ITE module
             "--in-text-editing "

             # -- PLOT SETTINGS --
             "--no-plot "
             "--title \"Polybench ${size} Syr2k Prediction with ${n_icl[$i]} ICL Examples\""
             "--export Figures/syr2k_${size}_${icl_f[$i]}.png "
             "--llm-range-only "
             "--override "

             # -- ANALYSIS SETTINGS --
             "--haystack-error 0.001 0.01 0.1 0.25 0.5 "
             # --disable-- "--highest-variation-only "
         );
    trythis="${trythis[*]}";
    echo "${trythis}";
    eval "${trythis}";
    if [ $? -ne 0 ]; then
        exit;
    fi;
done;

