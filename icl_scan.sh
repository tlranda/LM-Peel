#!/bin/bash

n_icl=( 1 5 10 50 100 );
icl_f=( "001" "005" "010" "050" "100" );

for i in "${!icl_f[@]}"; do
    trythis="python3 drive_syr2k_icl.py --seed 1234 2024 9999 --response-type quantitative --ICL-classes SM --eval-classes SM --objective-columns runtime --in-text-editing --title \"Polybench Syr2k Prediction with ${n_icl[$i]} ICL Examples\" --n-ICL ${n_icl[$i]} --cache syr2k_SM.pkl --export syr2k_${icl_f[$i]}.png --llm-range-only --override --highest-variation-only --n-rounds 5"
    echo "${trythis}";
    eval "${trythis}";
    if [ $? -ne 0 ]; then
        exit;
    fi;
done;

