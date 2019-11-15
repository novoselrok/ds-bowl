import os

import numpy as np
from bayes_opt import BayesianOptimization, JSONLogger, Events
from bayes_opt.util import load_logs
from sklearn.metrics import cohen_kappa_score
import matplotlib.pyplot as plt

from submission import get_train_test_features, stratified_group_k_fold, \
    label_encode_categorical_features, get_correct_attempts_clf, get_uncorrect_attempts_reg, fit_model


plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)

def attempts_to_group(correct_proba, uncorrect):
    uncorrect = uncorrect if uncorrect >= 0 else 0
    if correct_proba[0] > 0.4:
        return 0

    if correct_proba[0] < 0.06:
        return 3

    if uncorrect < 1:
        return 3
    if uncorrect < 2:
        return 2

    return 1


def lgb_multi_model_eval(**params):
    def lgb_score(
            x_train, _x_val,
            y_correct_train, _y_correct_val,
            y_uncorrect_train, _y_uncorrect_val,
            y_val_accuracy_group,
            val_installation_ids,
            clf_correct_attempts_params,
            reg_uncorrect_attempts_params,
    ):
        oof_pred = np.zeros((_x_val.shape[0], 2))
        scores = []
        for (val_split, test_split) in stratified_group_k_fold(None, y_val_accuracy_group, val_installation_ids, 2,
                                                               seed=2019):
            x_val = _x_val.iloc[val_split]
            y_correct_val = _y_correct_val[val_split]
            y_uncorrect_val = _y_uncorrect_val[val_split]

            x_test = _x_val.iloc[test_split]

            correct_attempts_clf = get_correct_attempts_clf(clf_correct_attempts_params)
            uncorrect_attempts_reg = get_uncorrect_attempts_reg(reg_uncorrect_attempts_params)

            fit_model(correct_attempts_clf, x_train, y_correct_train, x_val, y_correct_val)
            fit_model(uncorrect_attempts_reg, x_train, y_uncorrect_train, x_val, y_uncorrect_val)

            correct_attempts_pred = correct_attempts_clf.predict_proba(x_test)
            uncorrect_attempts_pred = uncorrect_attempts_reg.predict(x_test)

            oof_pred[test_split, 0] = correct_attempts_pred[:, 0]
            oof_pred[test_split, 1] = uncorrect_attempts_pred

            accuracy_group_pred = [
                attempts_to_group(c, u) for c, u in zip(correct_attempts_pred, uncorrect_attempts_pred)
            ]

            score = cohen_kappa_score(y_val_accuracy_group[test_split], accuracy_group_pred, weights='quadratic')
            scores.append(score)

        print(scores)
        return np.mean(scores), oof_pred

    (
        df_train_features,
        df_test_features,
        y_correct,
        y_uncorrect,
        y_accuracy_group,
        train_installation_ids,
        _test_installation_ids
    ) = get_train_test_features()

    integer_params = ['max_depth', 'max_bin', 'num_leaves', 'min_child_samples', 'n_splits', 'subsample_freq']
    df_train, _ = label_encode_categorical_features(df_train_features, df_test_features)
    return cv(lgb_score,
              df_train, y_correct, y_uncorrect, y_accuracy_group, train_installation_ids, integer_params, **params)


def cv(score_fn, df_train, y_correct, y_uncorrect, y_accuracy_group, train_installation_ids, integer_params, **params):
    n_splits = 5

    clf_correct_attempts_params = {}
    reg_uncorrect_attempts_params = {}

    for param, value in params.items():
        [model, param_name] = param.split(':')

        if param_name in integer_params:
            value = int(value)

        if model == 'clf_correct_attempts':
            clf_correct_attempts_params[param_name] = value
        else:
            reg_uncorrect_attempts_params[param_name] = value

    scores = []
    oof_preds = np.zeros((df_train.shape[0], 2))
    kf = stratified_group_k_fold(df_train, y_accuracy_group, train_installation_ids, n_splits, seed=2019)
    for fold, (train_split, test_split) in enumerate(kf):
        x_train, x_val = df_train.iloc[train_split, :], df_train.iloc[test_split, :]
        y_correct_train, y_correct_val = y_correct[train_split], y_correct[test_split]
        y_uncorrect_train, y_uncorrect_val = y_uncorrect[train_split], y_uncorrect[test_split]

        score, oof_pred = score_fn(
            x_train, x_val,
            y_correct_train, y_correct_val,
            y_uncorrect_train, y_uncorrect_val,
            y_accuracy_group[test_split],
            train_installation_ids[test_split],
            clf_correct_attempts_params,
            reg_uncorrect_attempts_params,
        )

        oof_preds[test_split, :] = oof_pred

        print("Done fold")
        print(score)
        scores.append(score)

    # color_map = {
    #     0: 'red',
    #     1: 'green',
    #     2: 'blue',
    #     3: 'black',
    # }
    #
    # for group, color in color_map.items():
    #     # plt.scatter(oof_preds[y_accuracy_group == group, 0], oof_preds[y_accuracy_group == group, 1], c=color)
    #     plt.title("group " + str(group) + " n samples: " + str((y_accuracy_group == group).sum()))
    #     plt.xlabel('probabilty of not solving correctly')
    #     plt.ylabel('uncorrect attempts')
    #     plt.xticks(np.arange(0, 1, step=0.05))
    #     plt.yticks(np.arange(0, 10, step=0.25))
    #     plt.hist2d(oof_preds[y_accuracy_group == group, 0], oof_preds[y_accuracy_group == group, 1], bins=[20, 40])
    #     plt.show()

    print(np.mean(scores))
    return np.mean(scores)


def bayes_opt(fn, params, probes=None):
    name = fn.__name__
    opt = BayesianOptimization(fn, params, verbose=2)
    if os.path.exists(f'./bayes_opt_logs/{name}.json'):
        print('Loading logs...')
        load_logs(opt, logs=[f'./bayes_opt_logs/{name}.json'])

    logger = JSONLogger(path=f'./bayes_opt_logs/{name}.json')
    opt.subscribe(Events.OPTMIZATION_STEP, logger)

    # Probe with a set of know "good" params
    if probes:
        for probe in probes:
            opt.probe(params=probe, lazy=True)

    opt.maximize(n_iter=200, init_points=5)
    print(opt.max)


if __name__ == '__main__':
    os.makedirs('bayes_opt_logs', exist_ok=True)

    bayes_opt(
        lgb_multi_model_eval,
        {
            'clf_correct_attempts:learning_rate': (0.01, 1.0),
            'clf_correct_attempts:max_depth': (3, 10),
            'clf_correct_attempts:max_bin': (2, 500),
            'clf_correct_attempts:num_leaves': (5, 500),
            'clf_correct_attempts:min_child_samples': (50, 1500),
            'clf_correct_attempts:min_child_weight': (0.1, 1000),
            'clf_correct_attempts:colsample_bytree': (0.05, 0.6),
            'clf_correct_attempts:reg_alpha': (0.1, 30),
            'clf_correct_attempts:reg_lambda': (0.1, 30),

            'reg_uncorrect_attempts:learning_rate': (0.01, 1.0),
            'reg_uncorrect_attempts:max_depth': (3, 10),
            'reg_uncorrect_attempts:max_bin': (2, 500),
            'reg_uncorrect_attempts:num_leaves': (5, 500),
            'reg_uncorrect_attempts:min_child_samples': (50, 1500),
            'reg_uncorrect_attempts:min_child_weight': (0.1, 1000),
            'reg_uncorrect_attempts:colsample_bytree': (0.05, 0.6),
            'reg_uncorrect_attempts:reg_alpha': (0.1, 30),
            'reg_uncorrect_attempts:reg_lambda': (0.1, 30),
        },
        probes=[
            {"clf_correct_attempts:colsample_bytree": 0.10334767151552683,
             "clf_correct_attempts:learning_rate": 0.029891107644305845,
             "clf_correct_attempts:max_bin": 403.63597687481536,
             "clf_correct_attempts:max_depth": 7.630901866600385,
             "clf_correct_attempts:min_child_samples": 867.6192673961293,
             "clf_correct_attempts:min_child_weight": 160.84184939114124,
             "clf_correct_attempts:num_leaves": 407.630333559809,
             "clf_correct_attempts:reg_alpha": 15.807080783827528,
             "clf_correct_attempts:reg_lambda": 10.109721556430491,
             "reg_uncorrect_attempts:colsample_bytree": 0.3885796025348852,
             "reg_uncorrect_attempts:learning_rate": 0.09146678093888543,
             "reg_uncorrect_attempts:max_bin": 298.03316385733507,
             "reg_uncorrect_attempts:max_depth": 7.439908933764865,
             "reg_uncorrect_attempts:min_child_samples": 274.88036337243125,
             "reg_uncorrect_attempts:min_child_weight": 361.1130703470717,
             "reg_uncorrect_attempts:num_leaves": 317.361765580482,
             "reg_uncorrect_attempts:reg_alpha": 5.2603473767855515,
             "reg_uncorrect_attempts:reg_lambda": 28.75196606940299},
            {"clf_correct_attempts:colsample_bytree": 0.7479871340435861,
             "clf_correct_attempts:learning_rate": 0.16420226500237103,
             "clf_correct_attempts:max_bin": 478.9454982979397,
             "clf_correct_attempts:max_depth": 7.074141701427287,
             "clf_correct_attempts:min_child_samples": 160.65491022582302,
             "clf_correct_attempts:min_child_weight": 22.407785965651843,
             "clf_correct_attempts:num_leaves": 265.60671862673746,
             "clf_correct_attempts:reg_alpha": 29.848926681241593,
             "clf_correct_attempts:reg_lambda": 27.132113615450297,
             "reg_uncorrect_attempts:colsample_bytree": 0.47037170385250415,
             "reg_uncorrect_attempts:learning_rate": 0.012162585925199094,
             "reg_uncorrect_attempts:max_bin": 25.980553497754865,
             "reg_uncorrect_attempts:max_depth": 5.786974861042519,
             "reg_uncorrect_attempts:min_child_samples": 1463.1282815050479,
             "reg_uncorrect_attempts:min_child_weight": 131.64426437899147,
             "reg_uncorrect_attempts:num_leaves": 471.97671065831497,
             "reg_uncorrect_attempts:reg_alpha": 12.371956179828944,
             "reg_uncorrect_attempts:reg_lambda": 19.2948688659447},
            {"clf_correct_attempts:colsample_bytree": 0.1248098023982449,
             "clf_correct_attempts:learning_rate": 0.10342583432650229,
             "clf_correct_attempts:max_bin": 187.14525960895386,
             "clf_correct_attempts:max_depth": 4.758365045540807,
             "clf_correct_attempts:min_child_samples": 235.51063646456035,
             "clf_correct_attempts:min_child_weight": 7.445893668578389,
             "clf_correct_attempts:num_leaves": 193.04674974320685,
             "clf_correct_attempts:reg_alpha": 12.398553150143126,
             "clf_correct_attempts:reg_lambda": 24.496136535125633,
             "reg_uncorrect_attempts:colsample_bytree": 0.3903008048793679,
             "reg_uncorrect_attempts:learning_rate": 0.15882978384614804,
             "reg_uncorrect_attempts:max_bin": 351.9710071636617,
             "reg_uncorrect_attempts:max_depth": 9.379019489701598,
             "reg_uncorrect_attempts:min_child_samples": 1304.882469005186,
             "reg_uncorrect_attempts:min_child_weight": 498.0164877147689,
             "reg_uncorrect_attempts:num_leaves": 16.041409064329585,
             "reg_uncorrect_attempts:reg_alpha": 2.5843001310082196,
             "reg_uncorrect_attempts:reg_lambda": 18.35238370653897},
        ]
    )
