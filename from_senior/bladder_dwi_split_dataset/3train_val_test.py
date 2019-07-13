import cPickle
import random
import csv
import os

random.seed()

positive_slices = cPickle.load(open('/DATA3_DB7/data/public/renji_data/splits_accession/positive_slices.txt', 'r'))
negative_slices = cPickle.load(open('/DATA3_DB7/data/public/renji_data/splits_accession/negative_slices.txt', 'r'))

train_percent = 0.6
val_percent = 0.2
all = {'pos': positive_slices, 'neg': negative_slices}
accession_set = set()
slice_dict = {'train': {'pos': [], 'neg': []},
              'val': {'pos': [], 'neg': []},
              'test': {'pos': [], 'neg': []}}

for label, slices in all.items():
    for slice in slices:
        accession = slice[0].split('/', 1)[0]
        accession_set.add(accession)

accession_list = list(accession_set)
random.shuffle(accession_list)
split0 = int(round(len(accession_list) * train_percent))
split1 = int(round(len(accession_list) * (train_percent + val_percent)))
train_set = set(accession_list[:split0])
val_set = set(accession_list[split0:split1])
test_set = set(accession_list[split1:])

for label, slices in all.items():
    for slice in slices:
        accession = slice[0].split('/', 1)[0]
        if accession in train_set:
            slice_dict['train'][label].append(slice)
        elif accession in val_set:
            slice_dict['val'][label].append(slice)
        else:
            assert accession in test_set
            slice_dict['test'][label].append(slice)


target_dir = '/DATA3_DB7/data/public/renji_data/splits_accession'
for phase in slice_dict:
    for label in slice_dict[phase]:
        with open(os.path.join(target_dir, '{}_{}.csv'.format(phase, label)), 'wb') as f:
            writer = csv.writer(f)
            writer.writerows(slice_dict[phase][label])
            print('phase:{}, label:{}, length:{}'.format(phase, label, len(slice_dict[phase][label])))
