#!/bin/bash
module load nvidia/cuda/10.2
nvidia-smi
#cd $SLUR_SUBMIT_DIR
#./train.sh
python main.py

# python main.py --model="CSAUNet" --mode=test --load_epoch=31 --batch_size=16 --data_path=../kaggle_3m_test --label_path=../kaggle_3m_test --checkpoint_dir=./checkpoints/201107_000426
# python main.py --model="CSAUNet" --mode=test --load_epoch=163 --batch_size=16 --data_path=../kaggle_3m_test --label_path=../kaggle_3m_test --checkpoint_dir=./checkpoints/201106_172144
# python main.py --model="CSAUNet" --mode=test --load_epoch=200 --batch_size=16 --data_path=../kaggle_3m_test --label_path=../kaggle_3m_test --checkpoint_dir=./checkpoints/201106_172144
# python main.py --model="AttUNet" --epoch=200 --batch_size=16 --loss="bce dice" --data_path=../kaggle_3m_train --label_path=../kaggle_3m_train --cross_validate --save_epoch_freq=20
