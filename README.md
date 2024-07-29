# Gaze-based Segmentation

## Getting started

0. Clone this repo and MedSAM repo inside:
    ```
    git clone https://github.com/leiluk1/gaze-based-segmentation.git
    cd gaze-based-segmentation
    git clone https://github.com/bowang-lab/MedSAM
    ```

1. Build docker container:
    ```
    docker build -t medsam_ft:latest .
    ```

2. Run docker container as daemon:
    ```
    docker run \
    -v .:/repo/ \
    --gpus all \
    -it -d --name medsam_ft medsam_ft
    ```


3. Start bash inside the docker container:

    0. In order to run scripts in the background, install and launch screen:
    ```
    sudo apt install screen
    ```

    1. Start bash:

    ```
    docker exec -it medsam_ft bash
    ```

4. Download data and model checkpoints to `data` and `weights`, respectively:
    ```
    pip install gdown
    gdown 19OWCXZGrimafREhXm8O8w2HBHZTfxEgU -O ./data/  # download WORD dataset
    apt-get install p7zip-full
    cd data
    7z x WORD-V0.1.0.zip  # unzip WORD dataset
    wget https://github.com/HiLab-git/WORD/raw/main/WORD_V0.1.0_labelsTs.zip  # download WORD test annotations
    unzip WORD_V0.1.0_labelsTs.zip -d ./WORD-V0.1.0/
    ```

    ```
    wget https://zenodo.org/records/5903037/files/Subtask1.zip?download=1 -O ./data/Subtask1.zip # download AbdomenCT-1K 
    wget https://zenodo.org/records/5903037/files/Subtask2.zip?download=1 -O ./data/Subtask2.zip # download AbdomenCT-1K 
    cd data
    unzip Subtask1.zip
    uzip Subtask2.zip
    cd Subtask2/TrainImage
    ls | xargs -I {} mv {} 2_{}
    cd ../TrainMask
    ls | xargs -I {} mv {} 2_{}
    cd ..
    mv TrainImage/* ../Subtask1/TrainImage/
    mv TrainMask/* ../Subtask1/TrainMask/
    cd ..
    rm -r Subtask2
    ```

    ```
    wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -O weights/sam/sam_vit_b_01ec64.pth  # download SAM checkpoint
    ```

    ```
    gdown 1UAmWL88roYR7wKlnApw5Bcuzf2iQgk6_ -O ./weights/medsam/  # download MedSAM checkpoint
    ```

5. Inside the container, run the following commands to double check that dependencies are installed:
    ```
    pip install -r requirements.txt
    pip install -e MedSAM/
    ```

6. Initialize your clearml credentials via:
    ```
    clearml-init
    ```

## Training

This training script demonstrates training of MedSAM with point prompts on the WORD dataset.

The training script `src/train_point_prompt.py` takes the following arguments:
* `--tr_npy_path`:` Path to the train data root directory;
* `--val_npy_path`: Path to the validation data root directory;
* `--test_npy_path`: Path to the test data root directory;
* `--medsam_checkpoint`: Path to the MedSAM checkpoint;
* `--max_epochs`: Maximum number of epochs;
* `--batch_size`: Batch size;
* `--num_workers`: Number of data loader workers;
* `--lr`: Learning rate (absolute lr);
* `--weight_decay`: Weight decay;
* `--accumulate_grad_batches`: Accumulate grad batches;
* `--seed`: Random seed for reproducibility;
* `--disable_aug`: Disable data augmentation;
* `--freeze_prompt_encoder`: Freeze prompt emcoder;
* `--gt_in_ram`: Store gt in RAM during data processing;
* `--num_points`: Number of points in the prompt;
* `--mask_diff`: Approach based on the mask difference;
* `--base_medsam_checkpoint`: Path to the MedSAM base predictor checkpoint (used only with mask_diff approach; if not provided, base predictor is ours MedSAM model copy);
* `--eval_per_organ`: Add performance comparison of different organs (evaluation per each class).


For instance, assume that the preprocessed data is stored in directory `data`, the MedSAM model is placed in `weigths/medsam` folder, and the model checkpoints should be saved in `train_point_prompt`. Then, to train the model, run the following commands:

1. Data preprocessing (with 10% saved on a disk):
    1. WORD Dataset:
        ```
        python src/preprocess_CT.py \
        --nii_path "./data/WORD-V0.1.0/imagesTr" \
        --gt_path "./data/WORD-V0.1.0/labelsTr" \
        --img_name_suffix ".nii.gz" \
        --npy_path "./data/WORD/train_" \
        --proportion 0.1; \
        python src/preprocess_CT.py \
        --nii_path "./data/WORD-V0.1.0/imagesVal" \
        --gt_path "./data/WORD-V0.1.0/labelsVal" \
        --img_name_suffix ".nii.gz" \
        --npy_path "./data/WORD/val_" \
        --proportion 0.1; \
        python src/preprocess_CT.py \
        --nii_path "./data/WORD-V0.1.0/imagesTs" \
        --gt_path "./data/WORD-V0.1.0/labelsTs" \
        --img_name_suffix ".nii.gz" \
        --npy_path "./data/WORD/test_" \
        --proportion 0.1
        ```

    2. AbdomenCT-1K Dataset:
        ```
        python src/preprocess_CT.py \
        --nii_path "./data/Subtask1/TrainImage" \
        --gt_path "./data/Subtask1/TrainMask" \
        --npy_path "./data/AbdomenCT/train_" \
        --proportion 0.1
        ```

2. Fine-tuning:

    One point prompt:

    ```
    python src/train_point_prompt.py \
    --tr_npy_path "data/WORD/train_CT_Abd/" \
    --val_npy_path "data/WORD/val_CT_Abd/" \
    --test_npy_path "data/WORD/test_CT_Abd/" \
    --medsam_checkpoint "weights/medsam/medsam_vit_b.pth" \
    --max_epochs 200 \
    --batch_size 24 \
    --num_workers 0 \
    --no-gt_in_ram \
    --eval_per_organ
    ```

    An example of the prompt with 20 points:

    ```
    python src/train_point_prompt.py \
    --tr_npy_path "data/WORD/train_CT_Abd/" \
    --val_npy_path "data/WORD/val_CT_Abd/" \
    --test_npy_path "data/WORD/test_CT_Abd/" \
    --medsam_checkpoint "weights/medsam/medsam_vit_b.pth" \
    --max_epochs 200 \
    --batch_size 24 \
    --num_workers 0 \
    --num_points 20 \
    --no-gt_in_ram \
    --eval_per_organ
    ```

    An example of fine-tuning based on the mask difference with 20 points prompt:

    ```
    python src/train_point_prompt.py \
    --tr_npy_path "data/WORD/train_CT_Abd/" \
    --val_npy_path "data/WORD/val_CT_Abd/" \
    --test_npy_path "data/WORD/test_CT_Abd/" \
    --medsam_checkpoint "weights/medsam/medsam_vit_b.pth" \
    --max_epochs 200 \
    --batch_size 24 \
    --num_workers 0 \
    --num_points 20 \
    --no-gt_in_ram \
    --mask_diff \
    --eval_per_organ
    ```


## Testing

One point prompt:

    ```
    python src/test_model.py \
    --tr_npy_path "data/WORD/train_CT_Abd/" \
    --val_npy_path "data/WORD/val_CT_Abd/" \
    --test_npy_path "data/WORD/test_CT_Abd/" \
    --medsam_checkpoint "weights/medsam/medsam_vit_b.pth" \
    --checkpoint "exp_name=0-epoch=42-val_loss=0.00.ckpt" \
    --batch_size 24 \
    --num_workers 0 \
    --num_points 1 \
    --eval_per_organ
    ```