import numpy as np
import matplotlib.pyplot as plt
import argparse


def get_wl(dist_weights, labels, np_train):
    feature_val = 0
    theta_val = 0
    num_ft = len(np_train[0])
    f_min = 999999
    for i in range(num_ft):
        ft_col = np_train[:, i]
        sorted_train = []

        for (sort_ft, sort_lb, sort_wts) in zip(ft_col, labels,
                dist_weights):
            sorted_train.append([sort_ft, sort_lb, sort_wts])

        sorted_train.sort()

        f_val = 0
        for j in range(len(labels)):
            if sorted_train[j][1] == 1:
                f_val += sorted_train[j][2]

        if f_val < f_min:
            f_min = f_val
            theta_val = sorted_train[j][0] - 1
            feature_val = i

        for j in range(len(labels)):
            f_val = f_val - sorted_train[j][1] * sorted_train[j][2]
            if f_val < f_min and j < len(labels) - 1 \
                and sorted_train[j][0] != sorted_train[j + 1][0]:
                f_min = f_val
                theta_val = 0.5 * (sorted_train[j][0] + sorted_train[j
                                   + 1][0])
                feature_val = i

    return (feature_val, theta_val)


def perform_AdaBoost_erm(s, lb, t):
    num_rows = len(lb)
    s_arr = np.array(s)

    # initialize d

    d = []
    for i in range(num_rows):
        d.append(1 *1.0/ num_rows*1.0)
    wl_err = []
    wl_say = []

    wl_h_val = []

    for i in range(t):
        (wl_ft, wl_thresh) = get_wl(d, lb, s_arr)

        # get feature column array

        ft_vals = s_arr[:, wl_ft]

        h_val = []

        # get hypothesis values for the WL feature

        for i in range(num_rows):
            if ft_vals[i] <= wl_thresh:
                h_val.append(-1)
            else:
                h_val.append(1)

        wl_h_val.append(h_val)
        err = get_wl_err(d, h_val, lb)

        if err == 0 or err == 1:
            say = 0.5 * np.log(num_rows - 1)

        else:

            say = 0.5 * np.log(1 / err - 1)


        wl_err.append(err)
        wl_say.append(say)

        for i in range(num_rows):
            d[i] = d[i] * np.exp(-say * lb[i] * h_val[i])
            # print (d[i])
        denominator = sum(d)

        for i in range(num_rows):
            d[i] = d[i]*1.0 / denominator*1.0
    sum_loss = 0

    for i in range(num_rows):
        wl_weighted_sum = 0
        for j in range(t):
            wl_weighted_sum += wl_say[j] * wl_h_val[j][i]

        predict_val = np.sign(wl_weighted_sum)

        if predict_val != lb[i]:
            sum_loss += 1

    erm_loss = sum_loss*1.0 / num_rows*1.0

    return erm_loss


def get_wl_err(wts, wl_predictions, actual_lb):
    ans = 0
    for i in range(len(actual_lb)):
        if wl_predictions[i] != actual_lb[i]:
            ans += wts[i]
    return ans


def performCV_AdaBoost_train(s, lb, t):

    num_rows = len(lb)
    s_arr = np.array(s)

    # initialize d

    d = []
    for i in range(num_rows):
        d.append(1 *1.0/ num_rows*1.0)
    # print (d)
    wl_say = []
    wl_thresholds = []
    wl_features = []

    for i in range(t):
        (wl_ft, wl_thresh) = get_wl(d, lb, s_arr)

        wl_features.append(wl_ft)
        wl_thresholds.append(wl_thresh)

        # get feature column array

        ft_vals = s_arr[:, wl_ft]

        h_val = []

        # get hypothesis values for the WL feature

        for i in range(num_rows):
            if ft_vals[i] <= wl_thresh:
                h_val.append(-1)
            else:
                h_val.append(1)

        # wl_h_val.append(h_val)

        err = get_wl_err(d, h_val, lb)
        # print(err)

        say = 0.5 * np.log(1 *1.0/(1.0*err) - 1)
            # print(say)
        # wl_err.append(err)

        wl_say.append(say)

        for i in range(num_rows):
            d[i] = d[i] * np.exp(-say * lb[i] * h_val[i])
        denominator = sum(d)

        for i in range(num_rows):
            d[i] = d[i]*1.0 / denominator*1.0

    return (wl_features, wl_thresholds, wl_say)


def performCV_AdaBoost_test(
    dtest,
    testlb,
    wl_features,
    wl_thresholds,
    wl_say,
    ):

    test_arr = np.array(dtest)
    sum_loss = 0

    for i in range(len(test_arr)):
        wl_weighted_sum = 0
        for j in range(len(wl_features)):
            if test_arr[i][wl_features[j]] <= wl_thresholds[j]:
                hyp = -1
            else:
                hyp = 1

            wl_weighted_sum += wl_say[j] * hyp

        predict_val = np.sign(wl_weighted_sum)

        if predict_val != testlb[i]:
            sum_loss += 1

    erm_loss = sum_loss*1.0 / len(test_arr)*1.0
    return erm_loss


def main():
    np.set_printoptions(suppress=True)
    my_argparse = \
        argparse.ArgumentParser(description='Perceptron commandline')

    # Add the arguments

    my_argparse.add_argument('--mode', type=str, help='ERM or ten fold')

    # Execute the parse_args() method

    args = my_argparse.parse_args()

    data = np.loadtxt('Breast_cancer_data.csv', delimiter=',',
                      skiprows=1)
    # data = np.delete(data, 5, 1)
    # initalize weights

    train_input = []
    data_labels = []

    # get train data without labels in train_inputs

    for i in data:
        train_input.append(i[0:-1])
        data_labels.append(i[len(i) - 1])

    for i in range(len(data_labels)):
        if data_labels[i] == 0:
            data_labels[i] = -1

    mode_val = args.mode
    epochs = 20
    nfold = 10
    cross_valid_rounds = 10

    # erm

    if mode_val == 'erm':
        erm_loss = perform_AdaBoost_erm(train_input, data_labels,
                epochs)
        print ('Final ERM error:', erm_loss)
        print ('Final ERM Accuracy:', 1 - erm_loss)


    elif mode_val == 'crossvalidation':

    # cross validation

        round_wise_cve = []
        round_wise_erm = []
        for c in range(cross_valid_rounds):

            # create folds

            fold = []
            fold_labels = []
            extras = len(train_input) % 10
            avg_fold_loss = 0
            avg_erm_loss = 0
            used_indices = []

            for i in range(10):
                if i < extras:
                    fold_size = int(len(train_input) / 10) + 1
                else:

                    fold_size = int(len(train_input) / 10)


                temp = []
                temp_labels = []
                for f in range(fold_size):
                    while True:
                        val = np.random.randint(0, len(train_input))
                        if val not in used_indices:
                            temp.append(train_input[val])
                            temp_labels.append(data_labels[val])
                            used_indices.append(val)
                            break
                fold.append(temp)
                fold_labels.append(temp_labels)

            for j in range(10):

                # test_data =[]

                train_data = []
                train_labels = []
                test_data = fold[j]
                test_labels = fold_labels[j]

                for k in range(10):
                    if k != j:
                        train_data.extend(fold[k])
                        train_labels.extend(fold_labels[k])

                # calculate error for test data

                (train_wl_features, train_wl_thresholds,
                 train_wl_say) = performCV_AdaBoost_train(train_data,
                        train_labels, epochs)
                erm_fold = performCV_AdaBoost_test(test_data,
                        test_labels, train_wl_features,
                        train_wl_thresholds, train_wl_say)

                avg_fold_loss += erm_fold

                # calculate erm for train data

                erm_train_loss = performCV_AdaBoost_test(train_data,
                        train_labels, train_wl_features,
                        train_wl_thresholds, train_wl_say)
                avg_erm_loss += erm_train_loss

            avg_fold_loss = avg_fold_loss/ 10
            round_wise_cve.append(avg_fold_loss)
            print ('Error for Round', c + 1, ':', avg_fold_loss)
            print ('Accuracy for Round', c + 1, ':', 1 - avg_fold_loss)

            avg_erm_loss = avg_erm_loss / 10
            round_wise_erm.append(avg_erm_loss)

        total = sum(round_wise_cve)
        final_avg_cve = total / cross_valid_rounds


        print ('Final Cross Validation Error:', final_avg_cve)
        print ('Final Cross Validation Accuracy:', 1 - final_avg_cve)

        # Plot for Cross Validation Error and Empirical Risk

        x = np.arange(1, 11)

        y1 = round_wise_cve
        y2 = round_wise_erm

        plt.plot(x, y1, marker='*', linestyle=':', label='Validation Error')
        plt.plot(x, y2, marker='*', linestyle=':', label='Empirical Risk')
        plt.title('Empirical Risk and Validation Errors corresponding to T=10 rounds')
        plt.xlabel('Round', fontsize=14)
        plt.ylabel('Error Values', fontsize=14)

        plt.legend()
        plt.show()


if __name__ == '__main__':
    main()
