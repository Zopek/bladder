import plot_3D_image
import matplotlib.pyplot as plt
import read_dwi
import numpy as np
import preprocess_util


def main(series_dir_list):
    for series_dir in series_dir_list:

        dwi_ordered_dict, spacing, b_is_guessed = read_dwi.read_dwi(series_dir)
        adc = preprocess_util.dwi2adc(dwi_ordered_dict)
        # b0 = dwi_ordered_dict[0]
        # if 1000 in dwi_ordered_dict:
        #     b1000 = dwi_ordered_dict[1000]
        #     b1000_fake = preprocess_util.calculate_dwi(adc, b0, 0, 1000)
        #     fig = plt.figure()
        #     p = plot_3D_image.Multi3DArrayPlane(fig, 1, 2)
        #     p.add(plot_3D_image.transpose_and_flip(b1000), 'Truth', cmap='gray', fixed_window=False)
        #     p.add(plot_3D_image.transpose_and_flip(b1000_fake), 'Recover_from_{}_stacks'.format(len(dwi_ordered_dict)), cmap='gray', fixed_window=False)
        #     p.ready()
        #     fig.show()

        num_image = 1 + len(dwi_ordered_dict)
        rows = int(np.floor(np.sqrt(num_image)))
        columns = int(np.ceil(num_image / float(rows)))
        fig = plt.figure()
        plane = plot_3D_image.Multi3DArrayPlane(fig, rows, columns)

        fig_lndwi_curve = plt.figure()

        index_list = []
        ln_dwi_curve_list = []
        num_curves = 9
        for i in range(num_curves):
            index = np.array(adc.shape) / 2
            index[0:2] += np.random.randint(-30, 31, 2)
            index = tuple(index.astype(np.int))
            index_list.append(index)
            ln_dwi_curve_list.append(list())

        plane.add(plot_3D_image.transpose_and_flip(adc), title="ADC_all", fixed_window=True)
        # ln_dwi_b0 = np.log(dwi_ordered_dict[0])
        for b_value in dwi_ordered_dict:
            dwi = dwi_ordered_dict[b_value]
            ln_dwi = np.log(dwi)
            for i in range(num_curves):
                ln_dwi_curve_list[i].append(ln_dwi[index_list[i]])

            # if b_value != 0:
            #     x = np.array([0, b_value]).reshape((-1, 1, 1, 1))
            #     y = np.stack([ln_dwi_b0, ln_dwi])
            #     w, _ = dwi2adc.linear_regression_with_single_variable(x, y)
            #     adc = -w
            #     title = "ADC_0_{}".format(b_value)
            #     if b_is_guessed:
            #         title = title + "guess"
            #     plane.add(plot_3D_image.transpose_and_flip(adc), title, fixed_window=True)
            if b_is_guessed:
                plane.add(plot_3D_image.transpose_and_flip(np.log(dwi_ordered_dict[b_value])),
                          "ln(DWI) B_guess={}".format(b_value),
                          cmap='gray', fixed_window=False)
            else:
                plane.add(plot_3D_image.transpose_and_flip(np.log(dwi_ordered_dict[b_value])),
                          "ln(DWI) B={}".format(b_value),
                          cmap='gray', fixed_window=False)
        plane.ready()
        fig.show()
        ax = fig_lndwi_curve.add_subplot(111)
        for i in range(num_curves):
            ax.plot(dwi_ordered_dict.keys(), ln_dwi_curve_list[i], '-*')
        fig_lndwi_curve.show()
    plt.show()
    return


if __name__ == "__main__":
    series = [
        '/raid/yjgu/bladder/bladder_cleaned_distinct_series/D1624257/dwi_ax_0',  # can T3
        # '/raid/yjgu/bladder/bladder_cleaned_distinct_series/D2049831/dwi_ax_0', # inf 2b
        '/raid/yjgu/bladder/bladder_cleaned_distinct_series/D2291117/dwi_ax_0',  # can Ta
        '/raid/yjgu/bladder/bladder_cleaned_distinct_series/D0645094/dwi_ax_1',  # can T1
        # '/raid/yjgu/bladder/bladder_cleaned_distinct_series/W0312093/dwi_ax_0', #can 2b
        # '/raid/yjgu/bladder/bladder_cleaned_distinct_series/D2314620/dwi_ax_0', #nor 2b
        '/raid/yjgu/bladder/bladder_cleaned_distinct_series/D0667043/dwi_ax_0',  # can T2
        '/raid/yjgu/bladder/bladder_cleaned_distinct_series/D1598531/dwi_ax_0',  # can T1
        # '/raid/yjgu/bladder/bladder_cleaned_distinct_series/D1647256/dwi_ax_0', #can T4
        # '/raid/yjgu/bladder/bladder_cleaned_distinct_series/D0501566/dwi_ax_0', #inf
    ]
    main(series)
