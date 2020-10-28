# ---- Working on July 18, 2020 ----
# Script 1: word2word-train-predict-evaluate.sh
#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -p nvidia
#SBATCH --mail-type=ALL
#SBATCH --mail-user=as10505
#SBATCH --mem=30000
#SBATCH --time=24:00:00
module purge
module load all
module load anaconda/2-4.1.1
module load cuda/8.0
module load gcc/4.9.3
source activate capstone-gpu
# Run complete word2word system; it is the default so we dont need to pass any flags
python3 transliterate.py

# Script 2: line2line-train-predict-evaluate.sh
#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -p nvidia
#SBATCH --mail-type=ALL
#SBATCH --mail-user=as10505
#SBATCH --mem=30000
#SBATCH --time=24:00:00
module purge
module load all
module load anaconda/2-4.1.1
module load cuda/8.0
module load gcc/4.9.3
source activate capstone-gpu
# Run complete line2line system
python3 transliterate.py --model_name=line2line --model_output_path=output/models/line2line_model --prediction_loaded_model_training_train_input=temp/line2line_training_train_input --prediction_loaded_model_training_train_output=temp/line2line_training_train_output --prediction_loaded_model_training_dev_input=temp/line2line_training_dev_input --prediction_loaded_model_training_dev_output=temp/line2line_training_dev_output --prediction_loaded_model_training_test_input=temp/line2line_training_test_input --predict_output_file=output/predictions/line2line-dev.out --evaluation_results_file=output/evaluations/line2line_evaluation_results.txt --batch_size=1024


# ---- Working on July 19, 2020 ----
# Script 1: word2word-test-set-train-predict-evaluate.sh
#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -p nvidia
#SBATCH --mail-type=ALL
#SBATCH --mail-user=as10505
#SBATCH --mem=30000
#SBATCH --time=24:00:00
module purge
module load all
module load anaconda/2-4.1.1
module load cuda/8.0
module load gcc/4.9.3
source activate capstone-gpu
# Run complete word2word system on test set
python3 transliterate.py --predict_input_file=splits_ldc/test/test-source.arabizi --predict_output_file=output/predictions/word2word-test.out --predict_output_word_aligned_gold=splits_ldc/test/test-word-aligned-target.gold --predict_output_sentence_aligned_gold=splits_ldc/test/test-sentence-aligned-target.gold --evaluation_results_file=output/evaluations/word2word_test_evaluation_results.txt

# ---- Working on July 21, 2020 ----
# Script 1: word2word-no-preprocessing-train-predict-evaluate.sh
#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -p nvidia
#SBATCH --mail-type=ALL
#SBATCH --mail-user=as10505
#SBATCH --mem=30000
#SBATCH --time=24:00:00
module purge
module load all
module load anaconda/2-4.1.1
module load cuda/8.0
module load gcc/4.9.3
source activate capstone-gpu
# Run complete word2word system; it is the default so we dont need to pass any flags
python3 transliterate.py --preprocess=False

# Script 2: line2line-no-preprocessing-train-predict-evaluate.sh
#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -p nvidia
#SBATCH --mail-type=ALL
#SBATCH --mail-user=as10505
#SBATCH --mem=30000
#SBATCH --time=24:00:00
module purge
module load all
module load anaconda/2-4.1.1
module load cuda/8.0
module load gcc/4.9.3
source activate capstone-gpu
# Run complete line2line system
python3 transliterate.py --preprocess=False --model_name=line2line --model_output_path=output/models/line2line_model --prediction_loaded_model_training_train_input=temp/line2line_training_train_input --prediction_loaded_model_training_train_output=temp/line2line_training_train_output --prediction_loaded_model_training_dev_input=temp/line2line_training_dev_input --prediction_loaded_model_training_dev_output=temp/line2line_training_dev_output --prediction_loaded_model_training_test_input=temp/line2line_training_test_input --predict_output_file=output/predictions/line2line-dev.out --evaluation_results_file=output/evaluations/line2line_evaluation_results.txt --batch_size=1024

# Script 3: hybrid_predict_evaluate.sh
#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -p nvidia
#SBATCH --mail-type=ALL
#SBATCH --mail-user=as10505
#SBATCH --mem=30000
#SBATCH --time=24:00:00
module purge
module load all
module load anaconda/2-4.1.1
module load cuda/8.0
module load gcc/4.9.3
source activate capstone-gpu

# Create the mle model; we already trained the word2word model so dont need to train it again
python3 transliterate.py --model_name=mle --model_output_path=output/models/mle_model --predict=False --evaluate_accuracy=False --evaluate_bleu=False
# Run hybdrid predictions and evaluate
python3 transliterate.py --model_name=hybrid --train=False --mle_model_file=output/models/mle_model --word2word_model_dir output/models/word2word_model --predict_output_file=output/predictions/hybrid_dev.out --evaluation_results_file=output/evaluations/hybrid_dev_evaluation_results.txt
