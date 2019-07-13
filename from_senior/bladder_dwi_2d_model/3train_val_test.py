import random
import csv
import os


def create_cv(num_folds, samples):
    cv_splits = []
    num_each_fold = int(len(samples) / float(num_folds))
    num_left = len(samples) - num_each_fold * num_folds
    start = 0
    for i in range(num_folds):
        stop = start + num_each_fold
        if i < num_left:
            stop += 1
        cv_splits.append(samples[start: stop])
        start = stop
    assert start == len(samples)
    assert num_folds == len(cv_splits)
    return cv_splits


def get_slices(mapping, accessions):
    slices = []
    for accession in accessions:
        slices += mapping[accession]
    return slices


def save_slices_csv(slices, csv_file_name):
    with open(csv_file_name, 'wb') as f:
        writer = csv.writer(f)
        writer.writerows(slices)


def main():
    random.seed()
    source_file = '/DATA/data/yjgu/bladder/dwi_ax_detection_dataset/all.csv'
    target_dir = '/DATA/data/yjgu/bladder/dwi_ax_detection_dataset/'
    test_percent = 0.2
    num_folds = 4

    with open(source_file, 'rb') as fd:
        reader = csv.reader(fd)
        slices = list(reader)
    accession_slice_mapping = dict()
    for s in slices:
        accession = s[0].split('/', 1)[0]
        accession_slice_mapping.setdefault(accession, []).append(s)
    accessions = accession_slice_mapping.keys()
    random.shuffle(accessions)

    # split test set

    num_test_samples = int(round(len(accessions) * test_percent))
    test_accessions = accessions[:num_test_samples]
    test_slices = get_slices(accession_slice_mapping, test_accessions)
    save_slices_csv(test_slices, os.path.join(target_dir, 'test.csv'))
    train_val_accessions = accessions[num_test_samples:]

    # split cross validation sets
    cv = create_cv(num_folds, train_val_accessions)
    for i, val in enumerate(cv):
        val_slices = get_slices(accession_slice_mapping, val)
        train = []
        for t in cv:
            if t != val:
                train += t
        train_slices = get_slices(accession_slice_mapping, train)
        save_slices_csv(val_slices, os.path.join(target_dir, '{}_cv_val.csv'.format(i)))
        save_slices_csv(train_slices, os.path.join(target_dir, '{}_cv_train.csv'.format(i)))


if __name__ == '__main__':
    main()
