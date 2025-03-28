from sklearn.tree import plot_tree
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestRegressor


def select_pred_feat() -> tuple[pd.DataFrame, pd.Series]:
    """Selects the target predictor and features"""
    iowa_data = pd.read_csv('train.csv')
    predictor = iowa_data['SalePrice']
    feature_names = ['LotArea',
                     'YearBuilt',
                     '1stFlrSF',
                     '2ndFlrSF',
                     'FullBath',
                     'BedroomAbvGr',
                     'TotRmsAbvGrd']

    features = iowa_data[feature_names]

    return features, predictor


def pred_feat_splitter(features: pd.DataFrame, predictor: pd.Series):
    """Selects the target predictor and features"""
    return train_test_split(features, predictor, random_state=1)


def model_creator(max_leaf_node: int, features: pd.DataFrame, predictor: pd.Series) -> DecisionTreeRegressor:
    """Creates a ML model"""
    iowa_model = DecisionTreeRegressor(
        max_leaf_nodes=max_leaf_node, random_state=0)
    iowa_model.fit(features, predictor)
    return iowa_model


def model_validator(iowa_model: DecisionTreeRegressor, val_X: pd.DataFrame, val_y: pd.Series):
    """Validates the created model"""
    val_predictions = iowa_model.predict(val_X)

    print(iowa_model.predict(val_X.head()))
    print(val_y.head().tolist())
    print(mean_absolute_error(val_predictions, val_y))


def get_mae(max_leaf_nodes: int, train_X: pd.DataFrame, val_X: pd.DataFrame, train_y: pd.Series, val_y: pd.Series):
    """Returns the max absolute error of a model"""
    model = model_creator(max_leaf_nodes, train_X, train_y)
    preds_val = model.predict(val_X)
    return mean_absolute_error(val_y, preds_val)


def find_best_tree_size(train_X: pd.DataFrame, val_X: pd.DataFrame, train_y: pd.Series, val_y: pd.Series):
    """Finds the optimal no of leaf nodes"""
    candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]
    mae_list = []
    for leaves in candidate_max_leaf_nodes:
        mae_list.append(get_mae(leaves, train_X, val_X, train_y, val_y))
    lowest_mae = min(mae_list)
    index_of_best_mae = mae_list.index(lowest_mae)
    return candidate_max_leaf_nodes[index_of_best_mae]


def tree_figure_creator(model: DecisionTreeRegressor, x: pd.DataFrame):
    """Creates a figure of the tree model"""
    plt.figure(figsize=(15, 10))
    plot_tree(model, feature_names=x.columns, filled=True)
    plt.show()


def random_forest_creator(train_X: pd.DataFrame, val_X: pd.DataFrame, train_y: pd.Series, val_y: pd.Series):
    """Creates a random forrest model"""

    rf_model = RandomForestRegressor(random_state=1)

    rf_model.fit(train_X, train_y)

    rf_val_mae = mean_absolute_error(val_y, rf_model.predict(val_X))

    print("Validation MAE for Random Forest Model: {}".format(rf_val_mae))


if __name__ == "__main__":
    x, y = select_pred_feat()
    train_X, val_X, train_y, val_y = pred_feat_splitter(x, y)
    model = model_creator(None, train_X, train_y)
    model_validator(model, val_X, val_y)
    optimal_tree_size = find_best_tree_size(train_X, val_X, train_y, val_y)
    final_model = model_creator(optimal_tree_size, x, y)
    # tree_figure_creator(final_model, x)
    random_forest_creator(train_X, val_X, train_y, val_y)
