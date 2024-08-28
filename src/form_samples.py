import argparse
import os

import cv2
import numpy as np


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--prefix_name',
        type=str,
        default="CT_Abd_word_0021",
        help="Fixed prefix filename."
    )
    parser.add_argument(
        '--data_directory',
        type=str,
        default="human_exp/doctors_exp/gts/1",
        help="Path to the validation data root directory."
    )

    return parser


def compute_count_labels(directory, prefix_filename):
    filename_labels = {}
    count_labels = {}

    for filename in os.listdir(directory):
        if filename.startswith(prefix_filename):
            gt_pth = os.path.join(directory, filename)

            filename_labels[filename] = []
            gt = np.load(gt_pth, 'r', allow_pickle=True)  # multiple labels [0,1,4,5...], (256,256)
            label_ids = np.unique(gt)[1:]  # [1,4,5...]
            for label_id in label_ids:
                gt2D = np.uint8(gt == label_id) 
                gt2D = (gt2D * 255).astype(np.uint8)
                thresh = cv2.threshold(
                    gt2D, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )[1]
                cnts = cv2.findContours(
                    thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cnts = cnts[0] if len(cnts) == 2 else cnts[1]

                if label_id not in count_labels:
                    count_labels[label_id] = 0

                for i in range(len(cnts)):
                    mask = np.zeros_like(gt2D)
                    cv2.drawContours(
                        mask, cnts, i, (255, 255, 255), thickness=cv2.FILLED
                    )
                    filename_labels[filename].append(label_id)
                    count_labels[label_id] += 1

    return filename_labels, count_labels


def save_metadata(counts, data_directory, filename):
    file_path = os.path.join(data_directory, "metadata.txt")
    with open(file_path, 'w') as file:
        file.write(f"Filename prefix: {filename}\n")
        for label, count in counts.items():
            file.write(f"Label: {label}, Count: {count}\n")


def main(data_directory, prefix_filename):
    filename_labels, count_labels = compute_count_labels(
        data_directory,
        prefix_filename
    )

    sorted_filename_labels = dict(sorted(filename_labels.items()))

    sorted_count_labels = dict(sorted(count_labels.items()))

    print("Sorted filenames and labels:")
    for file, labels in sorted_filename_labels.items():
        print(f"Filename: {file}, labels: {labels}")

    print("\nSorted counts of labels:")
    for label, count in sorted_count_labels.items():
        print(f"Label: {label}, count: {count}")

    save_metadata(sorted_count_labels, data_directory, prefix_filename)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    main(args.data_directory, args.prefix_name)
