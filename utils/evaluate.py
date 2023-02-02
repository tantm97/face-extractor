# -*- coding: utf-8 -*-
# Standard library imports
import logging
import operator

import numpy as np
import torch
from tqdm import tqdm


@torch.no_grad()
def get_features_and_labels(model, dataset_dl, device, mode):
    """Get features and labels.

    Args:
        model: The backbone.
        dataset_dl: The dataloader.
        device: A string determines training devices.
        mode: A string indicating the mode is validation or testing.

    Returns:
        A tensor array of features and a list of labels.
    """
    pbar = tqdm(dataset_dl, ncols=80, desc=mode)

    features = []
    labels = []
    for x, y in pbar:
        x = x.to(device)
        x = model(x)
        x = x.data.cpu().numpy()
        features.append(x)
        labels.append(y.data.cpu().numpy())

    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    return torch.from_numpy(features).float(), list(labels)


def get_unknowns_list(labels):
    """Get unknowns' list in the dataset.

    Args:
        labels: A list of labels in the dataset.

    Returns:
        A list of unknowns.
    """
    labels_dict = {i: labels.count(i) for i in labels}
    unknowns_list = [k for k in labels_dict if labels_dict[k] <= 1]
    return unknowns_list


def predict(scores, labels, unknowns_list, threshold):
    """Calculate performances of the backbone.

    Args:
        scores: Cosine scores between features.
        labels: A list of labels in the dataset.
        unknowns_list: A list of unknowns.
        threshold: An integer indicates the threshold of the same class.

    Returns:
        A list indicates performances of the backbone.
    """
    misidentified_name = 0
    known_to_unknown = 0
    unknown_to_known = 0
    correct_unknown = 0
    identified_name = 0
    predicted = [0, 0]

    for index, score in enumerate(scores):
        scores_sorted, indices = torch.sort(score, descending=True)
        if scores_sorted[1] >= threshold:
            if labels[index] not in unknowns_list:
                if labels[indices[1]] == labels[index]:
                    predicted[0] += 1
                    identified_name += 1
                else:
                    predicted[1] += 1
                    misidentified_name += 1
            else:
                predicted[1] += 1
                unknown_to_known += 1
        else:
            if labels[index] not in unknowns_list:
                predicted[1] += 1
                known_to_unknown += 1
            else:
                predicted[0] += 1
                correct_unknown += 1
        del scores_sorted, indices
    return predicted, [misidentified_name, known_to_unknown, unknown_to_known, identified_name, correct_unknown]


def write_evaluate_results(values, save_visual_dir, num_of_sample):
    """AI is creating summary for write_evaluate_results

    Args:
        values: The evaluation results.
        save_visual_dir: A string indicates a directory that stores evaluate results.
        num_of_sample: Number of samples.
    """
    with open(save_visual_dir, "a", encoding="utf-8") as f:
        f.write("Num Of Samples," + str(num_of_sample) + "\n")
        f.write(
            "Threshold,Correct Samples,Wrong Samples,Accuracy,Identified Name,Correct Unknown,Misidentified Name,Known To Unknown,Unknown To Known"
            + "\n"
        )
        for v in values:
            f.write(
                v[0]
                + ","
                + v[1]
                + ","
                + v[2]
                + ","
                + v[3]
                + ","
                + v[4]
                + ","
                + v[5]
                + ","
                + v[6]
                + ","
                + v[7]
                + ","
                + v[8]
                + "\n"
            )


def evaluate(model, dataset_dl, device, save_visual_dir, mode="Val"):
    """Evaluate model.

    Args:
        model: The backbone.
        dataset_dl: The dataloader.
        device: A string determines training devices.
        save_visual_dir: A string indicates a directory that stores evaluate results.
        mode: A string indicating the mode is validation or testing. Defaults to 'Val'.

    Returns:
        Max accuracy in the range of thresholds.
    """
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # train_emb = torch.unsqueeze(features, 0).cuda()
    features, labels = get_features_and_labels(model, dataset_dl, device, mode)
    unknowns_list = get_unknowns_list(labels)

    train_emb = torch.unsqueeze(features, 0).to(device)
    test_emb = torch.unsqueeze(features, 1).to(device)
    total_len = test_emb.shape[0]
    criterion = torch.nn.CosineSimilarity(dim=2).to(device)

    sub_len = 32
    scores = []
    for i in range((total_len // sub_len) + 1):
        if sub_len * (i + 1) > total_len:
            x = test_emb[sub_len * i :, ...]
        else:
            x = test_emb[sub_len * i : sub_len * (i + 1), ...]
        scores.append(criterion(train_emb, x).cpu().detach())

    scores = torch.cat(scores).to(device)
    # scores = scores.cpu().detach()
    # print(scores.shape)

    k_acc = {}
    values = []

    pbar = tqdm(np.arange(0.4, 0.85, 0.05), ncols=80, desc="Threshold")

    for threshold in pbar:
        pbar.set_postfix_str(str(round(threshold, 2)))

        predicted, details = predict(scores, labels, unknowns_list, round(threshold, 2))

        acc = predicted[0] / (predicted[0] + predicted[1])

        # logging.info("\nthreshold: %.2f" % round(threshold,2))
        # logging.info("correct samples: %d, wrong samples: %d, accuracy: %.3f" %(predicted[0], predicted[1], acc))
        # logging.info("identified name: %d, correct unknown: %d" %(details[3], details[4]))
        # logging.info("misidentified name: %d, known to unknown: %d, unknown to known: %d" %(details[0], details[1], details[2]))
        # logging.info("-"*10)
        k_acc[round(threshold, 2)] = acc
        values.append(
            [
                str(round(threshold, 2)),
                str(predicted[0]),
                str(predicted[1]),
                str(acc),
                str(details[3]),
                str(details[4]),
                str(details[0]),
                str(details[1]),
                str(details[2]),
            ]
        )

    write_evaluate_results(values, save_visual_dir, len(features))

    del features, labels, train_emb, test_emb
    # torch.cuda.empty_cache()

    return max(k_acc.items(), key=operator.itemgetter(1))[1]
