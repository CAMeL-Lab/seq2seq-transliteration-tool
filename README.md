# Seq2Seq Transliteration Tool

## Authors
- [Ali Shazal](https://github.com/alishazal)
- [Aiza Usman](https://github.com/aizausman)

## Prerequisites
It is important to match the versions of the prerequisites mentioned below in order to avoid errors. These prereqs assume you have a GPU. If you don't, ignore tensorflow-gpu (and instead install tensorflow), cudatoolkit 8.0, and cudnn 6.0.21.

- Python 3.6 and its following libraries:
    - [camel-tools](https://camel-tools.readthedocs.io/en/latest/getting_started.html)
    - tensorflow-gpu 1.4.0
    - cudatoolkit 8.0
    - cudnn 6.0.21
    - editdistance
    - numpy
    - pandas
    - scipy
- Anaconda 4.1.1
- CUDA 8.0
- GCC 4.9.3 (very important to match this version because of a grep command in seq2seq python scripts)

We ran our seq2seq systems with the GPU [NVIDIA Tesla V100 PCIe 32 GB](https://www.techpowerup.com/gpu-specs/tesla-v100-pcie-32-gb.c3184) on NYU Abu Dhabi's High Performance Computing cluster, known as Dalma. We set the memory flag to 30GB and set the max. time to 12 hours for each run. All other Dalma flags were kept as default. The .sh scripts that we ran can be seen in the file dalma_scripts.sh

## Repository Structure
```
ai/
    datasets/                   #module for preprocessing of any dataset
    models/                     #contains the model architectures of the fasttext-enabled seq2seq (named char_seq2seq) and simple seq2seq models
    tests/                      #contains the scripts that run the seq2seq systems, MLE system, and also contains the accuracy and bleu score scripts.
helpers/                        #contains helper files for the transliterate.py script that runs complete systems
output/
    evaluations/                #folder to store evaluation txt files
    models/                     #folder to store trained models
    predictions/                #folder to store predictions of trained systems
pretrained_word_embeddings/     #folder to store word embedding .bin files that are produced by Fasttext
splits_ldc/                     #contains the LDC data split into train, dev and test; there is also a source split that contains unannotated arabizi data which we use to train Fasttext.
temp/                           #folder to store machine learning input and output files that are produced during systems runs. These files are produced after preprocessing or ay-normalization.
```


# Transliteration Tool
There are 4 components of this tool. We will demostrate the use of each component using the [LDC BOLT Egyptian Arabic SMS/Chat and Transliteration](https://catalog.ldc.upenn.edu/LDC2017T07) data to transliterate Arabizi to Arabic.

## 1. Data Extraction from LDC XML Files
Download the [data](https://catalog.ldc.upenn.edu/LDC2017T07) and unzip the downloaded file. After unzipping you will get the folder 'bolt_sms_chat_ara_src_transliteration' which will contain 'data' folder, 'docs' folder and 'index.html' file. Place the folder 'bolt_sms_chat_ara_src_transliteration' in the root of this repository.

### Extracting Data Splits: Train, Dev & Test

We split the chat and SMS transliteration files in the following way:
- Train: CHT_ARZ_{20121228.0001-20150101.0002} and SMS_ARZ_{20120223.0001-20130902.0002}
- Dev: CHT_ARZ_{20120130.0000-20121226.0003} and SMS_ARZ_{20110705.0000-20120220.0000}
- Test: CHT_ARZ_{20150101.0008-20160201.0001} and SMS_ARZ_{20130904.0001-20130929.0000}.

We have already written the exact files for each split in this repo in split_ldc folder. The files are train.txt, dev.txt, and test.txt. These txt files can be used to move the xml files of a specific split into a separate folder using the script splits_ldc/makeNewLDCSplits.py in the following way:

1. To split the xml files into train, dev and test, run the following three commands (one command for each split):

```unix
# train
python3 splits_ldc/makeSplits.py bolt_sms_chat_ara_src_transliteration/data/transliteration/ splits_ldc/train.txt splits_ldc/train/xml_files

# dev
python3 splits_ldc/makeSplits.py bolt_sms_chat_ara_src_transliteration/data/transliteration/ splits_ldc/dev.txt splits_ldc/dev/xml_files

# test
python3 splits_ldc/makeSplits.py bolt_sms_chat_ara_src_transliteration/data/transliteration/ splits_ldc/test.txt splits_ldc/test/xml_files
```

Now the xml files for each split will reside in the specific folder of the split. Next, we will extract data from these XML files.

2. Extract the source and target. To do this run the following three commands (one command for each split)

```unix
# training data
python3 splits_ldc/getSourceAndTarget.py splits_ldc/train/xml_files/ splits_ldc/train/train-source.arabizi splits_ldc/train/train-word-aligned-target.gold splits_ldc/train/train-sentence-aligned-target.gold

# dev data
python3 splits_ldc/getSourceAndTarget.py splits_ldc/dev/xml_files/ splits_ldc/dev/dev-source.arabizi splits_ldc/dev/dev-word-aligned-target.gold splits_ldc/dev/dev-sentence-aligned-target.gold

# test data
python3 splits_ldc/getSourceAndTarget.py splits_ldc/test/xml_files/ splits_ldc/test/test-source.arabizi splits_ldc/test/test-word-aligned-target.gold splits_ldc/test/test-sentence-aligned-target.gold
```

The difference between word-aligned-target.gold files and sentences-aligned-target.gold files is the presence and absence of [+] and [-] tokens.

At this point in each of the split folders (train, dev and test) there will be there files: source.arabizi, word-aligned-target.gold, and sentence-aligned-target.gold.

### Extracting Unannotated Arabizi Data

We also extract data from the unannotated Arabizi files. This data is used to train Fasttext for pre-trained word embeddings. The files include train, dev and test Arabizi lines and many more (they have ~1M word). However, in order to make sure that dev and test lines are unseen, we exclude them when extracting all lines. To do all this, simply run the following command:

```unix
python3 splits_ldc/getSourceArabiziWithoutDevAndTest.py bolt_sms_chat_ara_src_transliteration/data/source splits_ldc/dev/xml_files splits_ldc/test/xml_files splits_ldc/source/source-without-dev-test.arabizi
```

### Training Fasttext with Unannotated Arabizi Data

Download fasttext at the root folder level by the following commands:

```unix
git clone https://github.com/facebookresearch/fastText.git

cd fastText

make
```

Now train word-embeddings on the unannotated arabizi data we extracted (without dev and test) using Fasttext. First, preprocess the data and then start training.

```unix
# Preprocess

cd ../ #move up one directory to come back to the root

python3 helpers/preprocess_fasttext_data.py --input_file=splits_ldc/source/source-without-dev-test.arabizi --output_file=splits_ldc/source/source-without-dev-test-preprocessed.arabizi

# Word-embeddings training
./fastText/fasttext skipgram -input splits_ldc/source/source-without-dev-test-preprocessed.arabizi -output pretrained_word_embeddings/arabizi_300_narrow -dim 300 -minn 2 -ws 2
```

This will save a .bin files at the output directory specified in the command. This bin file will be used in training to feed pre-trained word embeddings.

## 2. Training
To train models on the data we've extracted run any of the following scripts depending on which model you're training:
```unix
# Word2Word
python3 transliterate.py --predict=False --evaluate_accuracy=False --evaluate_bleu=False
```
```unix
# Line2Line
python3 transliterate.py --predict=False --evaluate_accuracy=False --evaluate_bleu=False --model_name=line2line --model_output_path=output/models/line2line_model --batch_size=1024
```
```unix
# MLE
python3 transliterate.py --predict=False --evaluate_accuracy=False --evaluate_bleu=False --model_name=mle --model_output_path=output/models/mle_model
```

**To train on any other data, please look at the flags in transliterate.py and run the scripts by setting the appropriate flags for your data.**

## 3. Prediction with Evaluation
To predict the dev (or test) files using the trained models (with their temp files for word embeddings) and evaluate them using the gold files, run the following scripts according to your model prediction input/output files. To disable preprocessing set the --preprocess flag as False.
```unix
# Word2word
python3 transliterate.py --train=False --predict_input_file=<prediction-input-file> --predict_output_file=<prediction-output-file> --predict_output_word_aligned_gold=<word-aligned-gold-file> --predict_output_sentence_aligned_gold=<sentence-aligned-gold-file> --evaluation_results_file=<evaluation-results-file>
```
```unix
# Line2Line
python3 transliterate.py --train=False --model_name=line2line --model_output_path=output/models/line2line_model --prediction_loaded_model_training_train_input=temp/line2line_training_train_input --prediction_loaded_model_training_train_output=temp/line2line_training_train_output --prediction_loaded_model_training_dev_input=temp/line2line_training_dev_input --prediction_loaded_model_training_dev_output=temp/line2line_training_dev_output --prediction_loaded_model_training_test_input=temp/line2line_training_test_input --predict_input_file=<prediction-input-file> --predict_output_file=<prediction-output-file> --predict_output_word_aligned_gold=<word-aligned-gold-file> --predict_output_sentence_aligned_gold=<sentence-aligned-gold-file> --evaluation_results_file=<evaluation-results-file> --batch_size=1024
```
```unix
# MLE
python3 transliterate.py --train=False --model_name=mle --model_output_path=output/models/mle_model --predict_input_file=<prediction-input-file> --predict_output_file=<prediction-output-file> --predict_output_word_aligned_gold=<word-aligned-gold-file> --predict_output_sentence_aligned_gold=<sentence-aligned-gold-file> --evaluation_results_file=<evaluation-results-file>
```

To run the hybrid system, which combines MLE (for OOV words) and Word2Word (for INV words) run the following script. It expects an MLE model and a Word2Word model through the --mle_model_file and --word2word_model_dir flags.
```unix
python3 transliterate.py --model_name=hybrid --train=False --mle_model_file=output/models/mle_model --word2word_model_dir output/models/word2word_model --predict_input_file=<prediction-input-file> --predict_output_file=<prediction-output-file> --predict_output_word_aligned_gold=<word-aligned-gold-file> --predict_output_sentence_aligned_gold=<sentence-aligned-gold-file> --evaluation_results_file=<evaluation-results-file>
```

## 4. Prediction without Evaluation
To generate predictions given a file using our best model (word2word) settings run the following script replacing \<input\_file\> with the path to your input file and \<output\_file\> with the path to the output file. Note: we're assuming that (a). the word2word model has already been trained and is stored in output/models/word2word_model (b). the temp folder has files that were automatically generated by the system for training (if you dont have these, we'd suggest running a complete cycle of the word2word system using the script given under the "development set results" below)

```unix
python3 transliterate.py --train=False --evaluate_accuracy=False --evaluate_bleu=False --predict_input_file=<input_file> --predict_output_file=<output_file>
```

To run the hybrid system to generate predictions, run the following:
```unix
python3 transliterate.py --model_name=hybrid --train=False --evaluate_accuracy=False --evaluate_bleu=False --mle_model_file=output/models/mle_model --word2word_model_dir output/models/word2word_model --predict_input_file=<input-file> --predict_output_file=<output-file>
```

## Running Complete Systems
Following are the scripts to replicate the results in our paper using the [LDC BOLT Egyptian Arabic SMS/Chat and Transliteration](https://catalog.ldc.upenn.edu/LDC2017T07) data with the word2word, line2line and MLE models (assuming all the data files are present and Fasttext has been trained - as explained in step 1). For no-proprocessing results, pass the flag --preprocess=False

#### Development Set Results
```unix
# Word2Word
python3 transliterate.py
```
```unix
# Line2Line
python3 transliterate.py --model_name=line2line --model_output_path=output/models/line2line_model --prediction_loaded_model_training_train_input=temp/line2line_training_train_input --prediction_loaded_model_training_train_output=temp/line2line_training_train_output --prediction_loaded_model_training_dev_input=temp/line2line_training_dev_input --prediction_loaded_model_training_dev_output=temp/line2line_training_dev_output --prediction_loaded_model_training_test_input=temp/line2line_training_test_input --predict_output_file=output/predictions/line2line-dev.out --evaluation_results_file=output/evaluations/line2line_dev_evaluation_results.txt --batch_size=1024
```
```unix
# MLE
python3 transliterate.py --model_name=mle --model_output_path=output/models/mle_model --predict_output_file=output/predictions/mle_dev.out --evaluation_results_file=output/evaluations/mle_dev_evaluation_results.txt
```

#### Test Set Results
```unix
# Word2Word
python3 transliterate.py --predict_input_file=splits_ldc/test/test-source.arabizi --predict_output_file=output/predictions/word2word-test.out --predict_output_word_aligned_gold=splits_ldc/test/test-word-aligned-target.gold --predict_output_sentence_aligned_gold=splits_ldc/test/test-sentence-aligned-target.gold --evaluation_results_file=output/evaluations/word2word_test_evaluation_results.txt
```
```unix
# MLE
python3 transliterate.py --model_name=mle --model_output_path=output/models/mle_model --predict_input_file=splits_ldc/test/test-source.arabizi --predict_output_file=output/predictions/mle-test.out --predict_output_word_aligned_gold=splits_ldc/test/test-word-aligned-target.gold --predict_output_sentence_aligned_gold=splits_ldc/test/test-sentence-aligned-target.gold --evaluation_results_file=output/evaluations/mle_test_evaluation_results.txt
```

## Troubleshooting
#### Tensor Shape Error on Word Embeddings: LHS not equal to RHS
This error comes up when there is a difference in the files that the seq2seq model was trained on and the files that the model is told are "training" files during predictions. The model needs the same files in training and prediction because it has to load the same word embeddings everytime. If the files are different, the shape of the tensors won't match. So make sure that the files for the flags --prediction_loaded_model_training_train_input, --prediction_loaded_model_training_train_output, --prediction_loaded_model_training_dev_input, --prediction_loaded_model_training_dev_output, --prediction_loaded_model_training_test_input are the same files that were produced in temp during training. 

## License
This tool is available under the MIT license. See the [LICENSE file](LICENSE) for more info.