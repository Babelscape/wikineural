#!/bin/sh


for idx in $(seq 0 9); do
  echo doing "conll-en-${idx}"
  PYTHONPATH=".:src" python scripts/predict.py --name "conll-en-${idx}" --dataset conll --language en --mode eval --split test >> "results/${model}-en-conll.tsv";
done;

#for model in ON+wikiner ON+wikineural ON+wikineural-DA wikineural-baseline conll+wikiann ON+wikiann; do
#  for dataset in conll wikigold ontonotes; do
#    for rand in $(seq 0 9); do
#      echo doing "${model}-${rand}"
#      PYTHONPATH=".:src" python scripts/predict.py --name "${model}-en-${rand}" --dataset "${dataset}" --language en --mode eval --split test >> "results/${model}-en-${dataset}.tsv";
#    done;
#  done;
#done

#for model in wikineural-DA; do
#  for size in 10000 20000 30000 None; do
#    for lang in de en nl; do
#      for rand in $(seq 0 3); do
#        PYTHONPATH=".:src" python scripts/predict.py --name $model-$lang-$size-rand$rand --dataset conll --language $lang --mode eval --split test >> results/$model-$lang-$size.tsv;
#      done;
#    done;
#  done;
#done
