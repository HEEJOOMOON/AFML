from feature_importance.importance import mean_decrease_accuracy
import shap
import matplotlib.pyplot as plt



def shapley(model, train_X, train_y, test_X):

    explainer = shap.TreeExplainer(model.fit(train_X, train_y))
    shap_values = explainer.shap_values(test_X)

    shap.plots.force(explainer.expected_value, shap_values)
    plt.savefig()
    plt.close()

    shap.force_plot(explainer.expected_value, shap_values, test_X)
    plt.show()
    shap.summary_plot(shap_values, test_X)
    plt.show()
    return shap_values


def MDA(model, X, y, cv_gen):
    return mean_decrease_accuracy(model=model, X=X, y=y, cv_gen=cv_gen)



if __name__ == '__main__':

    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from cross_validation import PurgedKFold
    from utils import standard_scaler

    feat_list = ['ATR_st', 'CCI_lt', 'DXY', 'AAAFF', 'WTIFutures']
    features = pd.read_csv("./Data/features_frac_sp.csv", index_col=0)
    final_features = features[feat_list]
    labels = pd.read_csv("./Labeling/sp500_returns_20.csv", index_col=0)
    final_features = standard_scaler(final_features)
    df = pd.concat([final_features, labels], axis=1)
    df.dropna(inplace=True)
    samples_info = pd.DataFrame()
    samples_info.index = df.index
    samples_info['end'] = df.index
    samples_info['end'] = samples_info.end.shift(-20)
    X = final_features.loc[df.index]
    y = labels.loc[df.index]

    train_test_split = PurgedKFold(n_splits=5, samples_info_sets=samples_info['end'], pct_embargo=0.01)
    model = RandomForestClassifier(max_depth=5, min_samples_split=500, min_samples_leaf=500)
    mda_featimp = MDA(model, X, y, train_test_split)
