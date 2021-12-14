[![Hugging Face Datasets](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-WikiNEuRal-blue)](https://huggingface.co/datasets/Babelscape/wikineural)

 ![logo](img/wikineural.png)
--------------------------------------------------------------------------------

Data and evaluation code for the paper [WikiNEuRal: Combined Neural and Knowledge-based Silver Data Creation for Multilingual NER](https://aclanthology.org/2021.findings-emnlp.215/).

```bibtex
@inproceedings{tedeschi-etal-2021-wikineural-combined,
    title = "{W}iki{NE}u{R}al: {C}ombined Neural and Knowledge-based Silver Data Creation for Multilingual {NER}",
    author = "Tedeschi, Simone  and
      Maiorca, Valentino  and
      Campolungo, Niccol{\`o}  and
      Cecconi, Francesco  and
      Navigli, Roberto",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2021",
    month = nov,
    year = "2021",
    address = "Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.findings-emnlp.215",
    pages = "2521--2533",
    abstract = "Multilingual Named Entity Recognition (NER) is a key intermediate task which is needed in many areas of NLP. In this paper, we address the well-known issue of data scarcity in NER, especially relevant when moving to a multilingual scenario, and go beyond current approaches to the creation of multilingual silver data for the task. We exploit the texts of Wikipedia and introduce a new methodology based on the effective combination of knowledge-based approaches and neural models, together with a novel domain adaptation technique, to produce high-quality training corpora for NER. We evaluate our datasets extensively on standard benchmarks for NER, yielding substantial improvements up to 6 span-based F1-score points over previous state-of-the-art systems for data creation.",
}
```

**Please consider citing our work if you use data and/or code from this repository.**

In a nutshell, WikiNEuRal consists in a novel technique which builds upon a multilingual lexical knowledge base (i.e., [BabelNet](https://babelnet.org/)) and transformer-based architectures (i.e., [BERT](https://arxiv.org/abs/1810.04805)) to produce high-quality annotations for multilingual NER. It shows consistent improvements of up to **6 span-based F1-score points against state-of-the-art alternative** data production methods on common benchmarks for NER. Moreover, in our paper we also present a new approach for creating **interpretable word embeddings** together with a **Domain Adaptation algorithm**, which enable WikiNEuRal to create **domain-specific training corpora**.



# Data

| Dataset Version | Sentences | Tokens | PER | ORG | LOC | MISC | OTHER |
| :------------- | -------------: | -------------: | -------------: | -------------: | -------------: | -------------: | -------------: |
| [WikiNEuRal EN](data/wikineural/en/train.conllu) | 116k | 2.73M | 51k | 31k | 67k | 45k | 2.40M |
| [WikiNEuRal ES](data/wikineural/es/train.conllu) | 95k | 2.33M | 43k | 17k | 68k | 25k | 2.04M |
| [WikiNEuRal NL](data/wikineural/nl/train.conllu) | 107k | 1.91M | 46k | 22k | 61k | 24k | 1.64M |
| [WikiNEuRal DE](data/wikineural/de/train.conllu) | 124k | 2.19M | 60k | 32k | 59k | 25k | 1.87M |
| [WikiNEuRal RU](data/wikineural/ru/train.conllu) | 123k | 2.39M | 40k | 26k | 89k | 25k | 2.13M |
| [WikiNEuRal IT](data/wikineural/it/train.conllu) | 111k | 2.99M | 67k | 22k | 97k | 26k | 2.62M |
| [WikiNEuRal FR](data/wikineural/fr/train.conllu) | 127k | 3.24M | 76k | 25k | 101k | 29k | 2.83M |
| [WikiNEuRal PL](data/wikineural/pl/train.conllu) | 141k | 2.29M | 59k | 34k | 118k | 22k | 1.91M |
| [WikiNEuRal PT](data/wikineural/pt/train.conllu) | 106k | 2.53M | 44k | 17k | 112k | 25k | 2.20M |
| [WikiNEuRal EN DA (CoNLL)](data/wikineural-DA-conll/en/train.conllu) | 29k | 759k | 12k | 23k | 6k | 3k | 0.54M |
| [WikiNEuRal NL DA (CoNLL)](data/wikineural-DA-conll/nl/train.conllu) | 34k | 598k | 17k | 8k | 18k | 6k | 0.51M |
| [WikiNEuRal DE DA (CoNLL)](data/wikineural-DA-conll/de/train.conllu) | 41k | 706k | 17k | 12k | 23k | 3k | 0.61M |
| [WikiNEuRal EN DA (OntoNotes)](data/wikineural-DA-ON/en/train.conllu) | 48k | 1.18M | 20k | 13k | 38k | 12k | 1.02M |

Further datasets, such as the combination of WikiNEuRal with gold-standard training data (i.e., **CoNLL**) or the gold-standard datasets themselves, can be obtained by simply concatenating the two `train.conllu` files together (e.g., `data/conll/en/train.conllu` and `data/wikineural/en/train.conllu` give `CoNLL+WikiNEuRal`).


# How to use
1. To train 10 models on CoNLL English, run:
    ```
    python run.py -m +train.seed_idx=0,1,2,3,4,5,6,7,8,9 data.datamodule.source=conll data.datamodule.language=en
    ```
    **note**: for the EN, ES, NL and DE versions of WikiNEuRal, you can use the CoNLL splits as validation and testing material (e.g., copy the `data/conll/en/val.conllu` into `data/wikineural/en/`). Similarly, for RU and PL you can use the BSNLP splits. For the other languages instead, you can use the [scripts/create_splits.py](scripts/create_splits.py) script to split a given `train.conllu` file into train, dev and test sets.

2. To produce results for the 10 trained models, run:
    ```
    bash test.sh
    ```
    
    `test.sh` also contains more complex bash for loops that can produce results on multiple datasets / models at once.

# License 
WikiNEuRal is licensed under the CC BY-SA-NC 4.0 license. The text of the license can be found [here](https://github.com/Babelscape/wikineural/blob/master/LICENSE).

We underline that the source from which the raw sentences have been extracted is Wikipedia ([wikipedia.org](https://www.wikipedia.org/)) and the NER annotations have been produced by [Babelscape](https://babelscape.com/).


# Acknowledgments
We gratefully acknowledge the support of the **ERC Consolidator Grant MOUSSE No. 726487** under the European Union’s Horizon2020 research and innovation programme ([http://mousse-project.org/](http://mousse-project.org/)).

This work was also supported by the **PerLIR project** (Personal Linguistic resources in Information Retrieval) funded by the MIUR Progetti di ricerca di Rilevante Interesse Nazionale programme (PRIN2017).

The code in this repository is built on top of [![](https://shields.io/badge/-nn--template-emerald?style=flat&logo=github&labelColor=gray)](https://github.com/lucmos/nn-template).
