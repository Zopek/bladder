import cPickle
import random
import csv
import os


positive_slices = cPickle.load(open('/DATA3_DB7/data/public/renji_data/splits_accession_detection/positive_slices.txt', 'r'))
negative_slices = cPickle.load(open('/DATA3_DB7/data/public/renji_data/splits_accession_detection/negative_slices.txt', 'r'))

percent = 0.25

all = {'pos': positive_slices, 'neg': negative_slices}
accession_set = set()

for label, slices in all.items():
    for slice in slices:
        accession = slice[0].split('/', 1)[0]
        accession_set.add(accession)

accession_list = list(accession_set)
random.shuffle(accession_list)
split0 = int(round(len(accession_list) * percent))
split1 = int(round(len(accession_list) * percent * 2))
split2 = int(round(len(accession_list) * percent * 3))

set0 = set(accession_list[:split0])
set1 = set(accession_list[split0:split1])
set2 = set(accession_list[split1:split2])
set3 = set(accession_list[split2:])
print len(set0), len(set1), len(set2), len(set3)

def merge(list_list):
    merged = []
    for listi in list_list:
        merged = merged + list(listi)
    return merged

all_sets = [set0, set1, set2, set3]
for i in range(len(all_sets)):
    test_set = all_sets[i]
    train_set = merge(all_sets[:i] + all_sets[i+1:])
    print i, len(test_set), len(train_set)

    slice_dict = {'train': {'pos': [], 'neg': []},
              'test': {'pos': [], 'neg': []}}

    for label, slices in all.items():
        for slice in slices:
            accession = slice[0].split('/', 1)[0]
            if accession in train_set:
                slice_dict['train'][label].append(slice)
            else:
                assert accession in test_set
                slice_dict['test'][label].append(slice)

    target_dir = '/DATA3_DB7/data/public/renji_data/splits_accession_detection'
    for phase in slice_dict:
        for label in slice_dict[phase]:
            with open(os.path.join(target_dir, '{}_{}_{}.csv'.format(str(i),phase, label)), 'wb') as f:
                writer = csv.writer(f)
                writer.writerows(slice_dict[phase][label])
                print('phase:{}, label:{}, length:{}'.format(phase, label, len(slice_dict[phase][label])))
