#!/bin/bash

n_icl=( 1 5 10 50 100 );
icl_f=( "001" "005" "010" "050" "100" );
size="XL";
show_prompts=""; #--show-prompt";
if [[ "${size}" == "SM" ]]; then
    size_options="";
elif [[ "${size}" == "XL" ]]; then
    size_options=""; #--highest-variation-only";
fi

# Set parallel in environment
export TOKENIZERS_PARALLELISM=true;
for i in "${!icl_f[@]}"; do
    trythis="python3 drive_syr2k_icl.py --seed 1234 2024 9999 --response-type quantitative --ICL-classes ${size} --eval-classes ${size} --objective-columns runtime --in-text-editing --title \"Polybench ${size} Syr2k Prediction with ${n_icl[$i]} ICL Examples\" --n-ICL ${n_icl[$i]} --cache syr2k_${size}.pkl --export Figures/syr2k_${size}_${icl_f[$i]}.png --llm-range-only --override --n-rounds 5 ${size_options} --haystack-error 0.001 0.01 0.1 0.25 0.5 --scientific-notation --problem-introduction syr2k_${size}_intro.txt ${show_prompts}"
    echo "${trythis}";
    eval "${trythis}";
    if [ $? -ne 0 ]; then
        exit;
    fi;
done;

