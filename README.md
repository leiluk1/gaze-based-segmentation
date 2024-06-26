# Gaze-based Segmentation

## Getting started

1. Install the required dependencies:
    ```
    pip install -r requirements.txt
    ```

2. Install MedSAM:
    ```
    git clone https://github.com/bowang-lab/MedSAM
    cd MedSAM
    pip install -e .
    ```

## Training

This training script demonstrates training of MedSAM with point prompts on the WORD dataset.

The training script `src/train_point_prompt.py` takes the following arguments:
* `--tr_npy_path`:` Path to the train data root directory;
* `--val_npy_path`: Path to the validation data root directory;
* `--medsam_checkpoint`: Path to the MedSAM checkpoint;
* `--max_epochs`: Maximum number of epochs;
* `--batch_size`: Batch size;
* `--num_workers`: Number of data loader workers;
* `--lr`: Learning rate (absolute lr);
* `--weight_decay`: Weight decay;
* `--accumulate_grad_batches`: Accumulate grad batches;
* `--seed`: Random seed for reproducibility;
* `--disable_aug`: Disable data augmentation;
* `--num_points`: Number of points in the prompt.


For instance, assume that the preprocessed data is stored in directory `data`, the MedSAM model is placed in `weigths/medsam` folder, and the model checkpoints should be saved in `train_point_prompt`. Then, to train the model, run the following commands:

1. WORD data preprocessing (with 10% saved on a disk):
    ```
    python src/preprocess_CT.py --nii_path "./data/WORD-V0.1.0/imagesTr" --gt_path "./data/WORD-V0.1.0/labelsTr" --img_name_suffix ".nii.gz" --npy_path "./data/WORD/train_" --proportion 0.1
    ```

    ```
    python src/preprocess_CT.py --nii_path "./data/WORD-V0.1.0/imagesVal" --gt_path "./data/WORD-V0.1.0/labelsVal" --img_name_suffix ".nii.gz" --npy_path "./data/WORD/val_" --proportion 0.1
    ```

2. Fine-tuning:

    One point prompt:

    ```
    python src/train_point_prompt.py --tr_npy_path "data/WORD/train_CT_Abd/" --val_npy_path "data/WORD/val_CT_Abd/" --medsam_checkpoint "weights/medsam/medsam_vit_b.pth" --max_epochs 200 --batch_size 24 --num_workers 0
    ```

    An example of the prompt with 20 points:

    ```
    python src/train_point_prompt.py --tr_npy_path "data/WORD/train_CT_Abd/" --val_npy_path "data/WORD/val_CT_Abd/" --medsam_checkpoint "weights/medsam/medsam_vit_b.pth" --max_epochs 200 --batch_size 24 --num_workers 0 --num_points 20
    ```
