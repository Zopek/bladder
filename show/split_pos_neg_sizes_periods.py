import os
import csv

def main():

    record_path = '/DB/rhome/qyzheng/Desktop/Link to renji_data/labels/bladder_tags_period.csv'
    record_path1 = '/DB/rhome/qyzheng/Desktop/qyzheng/source/renji_data/process/dwi_t2w_t2wfs/all_sizes.csv'
    save_path = '/DB/rhome/qyzheng/Desktop/qyzheng/source/renji_data/process/dwi_t2w_t2wfs/all_sizes_periods.csv'

    periods = {}
    with open(record_path, 'rb') as f:
        reader = csv.reader(f)
        for item in reader:

            assert item[0] not in periods.keys()
            periods[item[0]] = item[1]

    with open(record_path1, 'rb') as f1:
        reader1 = csv.reader(f1)
        with open(save_path, 'wb') as f:
            writer = csv.writer(f)
            for item in reader1:

                accession = item[0].split('/')[0]
                if accession in periods.keys() and item[2] == '1':
                    if periods[accession] == 'Ta':
                        period = 0
                    elif periods[accession] == 'Tis':
                        period = 1                    
                    elif periods[accession] == 'T1':
                        period = 2
                    elif periods[accession] == 'T2':
                        period = 3
                    elif periods[accession] == '>=T2':
                        period = 4
                    elif periods[accession] == 'T3':
                        period = 5
                    elif periods[accession] == 'T4':
                        period = 6

                    slices = (item[0], item[1], item[2], item[3], item[4], 
                        item[5], item[6], item[7], item[8], period)
                    writer.writerow(slices)

                '''
                if item[2] == '1':
                    accession = item[0].split('/')[0]
                    assert accession in periods.keys()
                    if periods[accession] == 'Ta':
                        period = 0
                    elif periods[accession] == 'Tis':
                        period = 1                    
                    elif periods[accession] == 'T1':
                        period = 2
                    elif periods[accession] == 'T2':
                        period = 3
                    elif periods[accession] == '>=T2':
                        period = 4
                    elif periods[accession] == 'T3':
                        period = 5
                    elif periods[accession] == 'T4':
                        period = 6

                    slices = (item[0], item[1], item[2], item[3], item[4], 
                        item[5], item[6], item[7], item[8], period)
                    writer.writerow(slices)

                else:
                    continue
                '''
                
if __name__ == "__main__":

    main()