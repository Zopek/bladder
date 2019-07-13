# import dicom
import os
import traceback


def main(root_path):
    for accession_dir in os.listdir(root_path):
        accession_path = os.path.join(root_path, accession_dir)
        if not os.path.isdir(accession_path):
            continue
        dwi_counter = 0
        for series_dir in os.listdir(accession_path):
            # print series_dir
            if 'dwi_ax' not in series_dir:
                continue
            series_path = os.path.join(accession_path, series_dir)
            if not os.path.isdir(series_path):
                continue
            dwi_counter += 1
        if dwi_counter != 1:
            print 'Warning: dwi_ax_counter = {}: {}'.format(dwi_counter, accession_path)
    pass


if __name__ == '__main__':
    root = '/raid/yjgu/bladder_cleaned_distinct_series'
    main(root)
