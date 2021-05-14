# Optimal Robust Explantions
This repository constains the code to run the experiments of the paper "On Guaranteed Optimal Robust Explanations for NLP Models", which will appear on the proceeding of the 30th International Joint Conference on Artificial Intelligence. If you need to cite the paper, please use the following format:
```
@misc{malfa2021guaranteed,
    title={On Guaranteed Optimal Robust Explanations for NLP Models},
    author={Emanuele La Malfa and Agnieszka Zbrzezny and Rhiannon Michelmore and Nicola Paoletti and Marta Kwiatkowska},
    year={2021},
    eprint={2105.03640},
    archivePrefix={arXiv},
    primaryClass={cs.AI}
}
```

## Install Python Dependencies
In order to install the dependencies, you can run the command `pip install PACKAGENAME==X.Y.Z --user` where `PACKAGENAME` is the name of the missing dependency and `X.Y.Z` is the version, for example, `pip install numpy==1.18.5 --user`. If no specific version is specified in the previous list, you can run `pip install PACKAGENAME --user`.

## Install External Dependencies
You need to install a neural network verifier. Code in this folder is compliant with Marabou (https://github.com/NeuralNetworkVerification/Marabou/). We used the version of Marabou with commit hash `228234ba` and its Python API. Please note that we couldn't insert Marabou directly as it is an external tool with its own specific Licence and it has to be installed on the system: just clone into `OREs` folder (i.e., at the same level of folder `Explanations`) the files at the following link (https://github.com/NeuralNetworkVerification/Marabou/). Once installed (the Marabou's Github folder has instructions on how to install the package) you should have `Marabou` and `Explanations` as sub-directories of the `OREs` folder.

## Run Experiments
To launch an Optimal Robust Explanation (ORE) experiment, go to the `Explanations\abduction_algorithms\experiments\{DATASET}\` folder, where `{DATASET}` is either `SST` `Twitter` or `IMDB`, and run one of the python files (`.py` extension). We report a few usage examples.

### ORE for a Fully Connected Model on a Sample Review
Let's suppose that you want to extract an ORE from a sample review and you are using an Fully Connected model (FC), trained on SST dataset, with 25 input words and using the k-Nearest-Neighbours bounding box technique with `k=10`. 
You have to navigate to `Explanations\abduction_algorithms\experiments\SST\` folder and run the following:

```
python3 smallest_explanation_SST_fc_knn_linf.py -k 10 -w 5 -n 25 -i 'This is a very bad movie'
```

The algorithm identifies the words `is` and `bad` as an ORE. Additionally, results are logged and stored in a folder inside `results\HS-clear` (the complete path that is reported in the logs of the execution).

### Solving an hard instance with Adversarial Attacks
Let's suppose that you want to extract an ORE from a Twitter text and you are using a CNN model, trained on Twitter dataset, with 25 input words and using the k-Nearest-Neighbours bounding box technique with k=8. You can improve the convergence on hard instances with Adversarial Attacks by simply specifying the number of attacks that you want to launch (parameter -a/--adv). 
You have to navigate to the `Explanations\abduction_algorithms\experiments\Twitter\` folder and run the following:

```
python3 smallest_explanation_Twitter_cnn2d_knn_linf.py -k 8 -w 5 -n 25 -a 500 -i 'Spencer is not a good guy'
```

The algorithm identifies the words `Spencer` and `not` as an ORE (alongside a `PAD` token that highlights a problem with the model robustness). Again, results are stored in a folder inside `results\HS-clear` (the complete path that is reported in the logs of the execution).

### Detecting Decision Bias Using Cost Functions
Let's suppose you want to check whether an explanation will always contain the name of a character, actor or director: something which ideally should not be used for classifying whether a review is positive or negative. To do this, you can run the following command:

```
python3 smallest_HScost_explanation_SST_fc_knn_linf.py -k 27 -w 5 -a 0 -i '"Austin Powers in Goldmember has the right stuff for summer entertainment and has enough laughs to sustain interest to the end" -u False -x 'austin,powers,goldmember'
```

Please note that the `-x` command excludes specific words from the computation of the ORE (if possible). If the excluded word still appears in the ORE (which it does in this example, ``powers'' is present), you know that no explanation exists without it and therefore that there is a decision bias here.
