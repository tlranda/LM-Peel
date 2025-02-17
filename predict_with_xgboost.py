# -*- coding: utf-8 -*-

import itertools
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from matplotlib.ticker import FuncFormatter
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split

# Configure plot aesthetics
sns.set_style("whitegrid")

plt.rcParams.update({"font.size": 12})
plt.rc("axes", titlesize=14)  # fontsize of the axes title
plt.rc("axes", labelsize=12)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=12)  # fontsize of the tick labels
plt.rc("ytick", labelsize=12)  # fontsize of the tick labels

exponent_to_char = {
    -6: "u",
    -3: "m",
    0: "",
}


# A detailed parameter grid
# models = {
#     'XGBoost': {
#         'model': xgb.XGBRegressor(random_state=42, n_jobs=-1),
#         'params': {
#             # Core parameters optimized for datasets with 1000-10000 samples
#             'n_estimators': [300, 500],
#             'learning_rate': [0.01, 0.03, 0.05, 0.1],
#             'max_depth': [3, 4, 5, 6, 7, 9],
#             'min_child_weight': [1, 3, 5],

#             # Simplified sampling parameters
#             'subsample': [0.7, 0.8, 0.9, 1.0],
#             'colsample_bytree': [ 0.9, 1.0],

#             # Regularization focused on medium-sized data (1000-10000 samples)
#             'gamma': [0],
#             'reg_alpha': [0, 0.01, 0.1, 1],
#             'reg_lambda': [0, 0.01, 0.1, 1],

#             # Advanced options
#             'booster': ['gbtree', 'dart'],
#             'grow_policy': ['depthwise'],
#             'max_bin': [256],
#             'tree_method': ['hist']
#         }
#     }
# }

# Best model for 0.8, XL
models = {
    "XGBoost": {
        "model": xgb.XGBRegressor(random_state=42, n_jobs=-1),
        "params": {
            "tree_method": ["hist"],
            "subsample": [0.8],
            "reg_lambda": [0.01],
            "reg_alpha": [0],
            "n_estimators": [500],
            "min_child_weight": [3],
            "max_depth": [9],
            "max_bin": [256],
            "learning_rate": [0.03],
            "grow_policy": ["depthwise"],
            "gamma": [0],
            "colsample_bytree": [0.9],
            "booster": ["dart"],
        },
    }
}

# Best model for 100, XL
# models = {
#     'XGBoost': {
#         'model': xgb.XGBRegressor(random_state=42, n_jobs=-1),
#         'params': {
#             'tree_method': ['hist'],
#             'subsample': [0.8],
#             'reg_lambda': [0.01],
#             'reg_alpha': [0.01],
#             'n_estimators': [200],
#             'min_child_weight': [5],
#             'max_depth': [9],
#             'max_bin': [256],
#             'learning_rate': [0.03],
#             'grow_policy': ['depthwise'],
#             'gamma': [0],
#             'colsample_bytree': [1.0],
#             'booster': ['gbtree']
#         }
#     }
# }

# Best model for 100, SM
# models = {
#     'XGBoost': {
#         'model': xgb.XGBRegressor(random_state=42, n_jobs=-1),
#         'params': {
#             'tree_method': ['hist'],
#             'subsample': [0.7],
#             'reg_lambda': [0.01],
#             'reg_alpha': [0],
#             'n_estimators': [500],
#             'min_child_weight': [3],
#             'max_depth': [7],
#             'max_bin': [256],
#             'learning_rate': [0.05],
#             'grow_policy': ['depthwise'],
#             'gamma': [0],
#             'colsample_bytree': [0.9],
#             'booster': ['gbtree']
#         }
#     }
# }


def mean_squared_relative_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(((y_pred - y_true) / y_true) ** 2)


def mean_absolute_relative_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_pred - y_true) / y_true))


def percentage_relative_error_less_than(y_true, y_pred, threshold):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_pred - y_true) / y_true) < threshold)


def round_to_significant_digits(df, significant_digits):
    return df.applymap(
        lambda x: (
            float(f"{x:.{significant_digits}g}") if isinstance(x, (int, float)) else x
        )
    )


def perform_tuning(csv_file_path, output_dir, training_size, significant_digits):
    """
    Perform hyperparameter tuning for the given configuration
    """
    # Load the CSV file
    df = pd.read_csv(csv_file_path)

    # Remove the size column and convert booleans
    df = df.drop("size", axis=1)
    bool_cols = [
        "first_array_packed",
        "second_array_packed",
        "interchange_first_two_loops",
    ]
    df[bool_cols] = df[bool_cols].astype(int)

    # Round all values
    df = round_to_significant_digits(df, significant_digits)

    # Split data
    X = df.drop("runtime", axis=1)
    y = df["runtime"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=training_size, random_state=42
    )

    # Perform hyperparameter tuning for all model families
    results = []
    best_models = {}

    for model_name, config in models.items():
        print(f"\n=== Tuning {model_name} ===")

        search = RandomizedSearchCV(
            estimator=config["model"],
            param_distributions=config["params"],
            n_iter=1,
            scoring="r2",
            cv=3,
            n_jobs=-1,
            random_state=42,
            verbose=1,
        )

        search.fit(X_train, y_train)

        # Save detailed hyperparameter search results to disk
        pd.DataFrame(search.cv_results_).to_csv(
            os.path.join(output_dir, f"hyperparam_search_full_{model_name}.csv"), index=False
        )

        best_model = search.best_estimator_
        best_models[model_name] = best_model

        y_pred = best_model.predict(X_test)
        rmse = root_mean_squared_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        msre = mean_squared_relative_error(y_test, y_pred)
        mare = mean_absolute_relative_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        results.append(
            {
                "Model": model_name,
                "Best Params": search.best_params_,
                "Test RMSE": rmse,
                "Test MSE": mse,
                "Test MAE": mae,
                "Test MSRE": msre,
                "Test MARE": mare,
                "Test R2": r2,
                "Test Relative Error < 0.5": percentage_relative_error_less_than(
                    y_test, y_pred, 0.5
                ),
                "Test Relative Error < 0.1": percentage_relative_error_less_than(
                    y_test, y_pred, 0.1
                ),
                "Test Relative Error < 0.01": percentage_relative_error_less_than(
                    y_test, y_pred, 0.01
                ),
                "Test Relative Error < 0.001": percentage_relative_error_less_than(
                    y_test, y_pred, 0.001
                ),
                "Best Score (CV)": -search.best_score_,
            }
        )

    # Display results summary
    results_df = pd.DataFrame(results)

    # Save best results to disk
    results_df.to_csv(os.path.join(output_dir, "best_hyperparam_summary.csv"), index=False)

    print("\n=== Final Results ===")
    print(results_df.sort_values("Test RMSE"))

    # Feature importance for best model
    bm_name = results_df.loc[results_df["Test RMSE"].idxmin(), "Model"]
    bm = best_models[bm_name]
    print(f"\nBest Model: {bm_name}")

    if hasattr(bm, "feature_importances_"):
        feature_importance = pd.DataFrame(
            {"Feature": X.columns, "Importance": bm.feature_importances_}
        ).sort_values("Importance", ascending=False)

        print("\nFeature Importance:")
        print(feature_importance)

    y_pred = bm.predict(X_test)

    return results_df, bm, bm_name, X.columns, y_pred, y_test


def make_plots(
    results_df, best_model, best_model_name, feature_columns, y_pred, y_test, output_dir
):
    """
    Generate plots for the best models
    """
    plot_best_model_comparison(results_df, output_dir)
    plot_actual_vs_predicted(y_test, y_pred, best_model_name, output_dir)
    plot_feature_importance(best_model, best_model_name, feature_columns, output_dir)
    plot_residuals(y_test, y_pred, best_model_name, output_dir)


def plot_best_model_comparison(results_df, output_dir):
    """
    Performance metrics comparing the best model found for each model type
    """
    plt.figure(figsize=(6, 4))
    metrics = ["Test RMSE", "Test MSRE", "Test MARE"]
    sorted_models = results_df.sort_values("Test RMSE")["Model"].tolist()
    n_models = len(sorted_models)

    # Generate appropriate number of colors based on models
    fill_colors = sns.color_palette("tab10", n_colors=n_models)

    # Define hatch patterns
    hatches = ["////", "\\\\", "||||", "----", "++++", "xxxx", "....", "****"][
        :n_models
    ]

    # Create custom legend handles with black edges
    legend_handles = [
        plt.Rectangle(
            (0, 0), 1, 1, fc=fill_colors[i], ec="k", hatch=hatches[i], linewidth=1.5
        )
        for i in range(n_models)
    ]

    for i, metric in enumerate(metrics, 1):
        ax = plt.subplot(1, 3, i)

        # Updated barplot with proper hue assignment
        bp = sns.barplot(
            x="Model",
            y=metric,
            data=results_df,
            order=sorted_models,
            hue="Model",
            palette=fill_colors,
            legend=False,
        )

        # Apply black hatches
        for idx, bar in enumerate(bp.patches):
            bar.set_hatch(hatches[idx % len(hatches)])
            bar.set_edgecolor("k")
            bar.set_linewidth(2)

        # Remove x-axis labels and ticks
        plt.xlabel("")
        plt.xticks([])

        # Scale and annotate with exponents
        max_val = results_df[metric].max()
        if max_val > 0:
            exponent = np.floor(np.log10(max_val))
            scaling_factor = 10 ** (-exponent)
        else:
            scaling_factor = 1
            exponent = 0

        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda x, _: f"{(x * scaling_factor):.2f}")
        )

        if exponent != 0:
            ax.annotate(
                f"×1e{int(exponent)}",
                xy=(1.02, 0.97),
                xycoords="axes fraction",
                fontsize=12,
                ha="left",
                va="top",
            )

    # Add unified legend below plots
    plt.figlegend(
        handles=legend_handles,
        labels=sorted_models,
        loc="lower center",
        ncol=n_models,
        bbox_to_anchor=(0.5, -0.15),
        fontsize=12,
        title="Models",
        title_fontsize=14,
    )

    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.savefig(os.path.join(output_dir, "model_comparison.png"), bbox_inches="tight")
    plt.savefig(os.path.join(output_dir, "model_comparison.pdf"), bbox_inches="tight")
    plt.close()


def plot_actual_vs_predicted(y_test, y_pred, best_model_name, output_dir):
    """
    Actual vs Predicted Values for the best model
    Also save the values to a CSV file
    """
    plt.figure(figsize=(4, 3))

    # Calculate scaling factor for axes
    max_val = max(y_test.max(), y_pred.max())
    exponent = np.floor(np.log10(max_val)) if max_val > 0 else 0

    if exponent <= -6:
        exponent = -6
    elif exponent <= -3:
        exponent = -3
    else:
        exponent = 0

    text_exponent = exponent_to_char[exponent]
    scaling_factor = 10 ** (-exponent) if exponent != 0 else 1

    # Create scaled plot
    point_color = sns.color_palette("tab10")[0]
    sns.scatterplot(
        x=y_test * scaling_factor,
        y=y_pred * scaling_factor,
        alpha=0.5,
        color=point_color,
        edgecolor="w",
        linewidth=0.3,
    )

    # Format reference line
    min_val = min(y_test.min(), y_pred.min()) * scaling_factor
    max_val = max_val * scaling_factor
    plt.plot([min_val, max_val], [min_val, max_val], "k--", lw=2)

    # Axis labels with scaling notation
    plt.xlabel(f"Actual Runtime ({text_exponent}s)")
    plt.ylabel(f"Predicted Runtime ({text_exponent}s)")

    # plt.title(f"R$^2$: {r2_score(y_test, y_pred):.2f}")

    plt.grid(False)
    plt.tight_layout()

    sanitized_name = best_model_name.replace(" ", "_")
    plt.savefig(
        os.path.join(
            output_dir, f"actual_vs_predicted_best_model_{sanitized_name}.png"
        ),
        bbox_inches="tight",
    )
    plt.savefig(
        os.path.join(
            output_dir, f"actual_vs_predicted_best_model_{sanitized_name}.pdf"
        ),
        bbox_inches="tight",
    )
    plt.close()

    test_data_comparison = pd.DataFrame(
        {"Actual Runtime": y_test, "Predicted Runtime": y_pred}
    )
    test_data_comparison.to_csv(
        os.path.join(output_dir, "test_data_comparison.csv"), index=False
    )


def plot_feature_importance(best_model, best_model_name, feature_columns, output_dir):
    """
    Feature Importance Visualization
    """
    plt.figure(figsize=(6, 4))
    if hasattr(best_model, "feature_importances_"):
        feature_importance = pd.DataFrame(
            {"Feature": feature_columns, "Importance": best_model.feature_importances_}
        ).sort_values("Importance", ascending=False)

        # Use tab10 color palette with first color
        bar_color = sns.color_palette("tab10")[0]
        sns.barplot(
            x="Importance",
            y="Feature",
            data=feature_importance,
            color=bar_color,
            edgecolor="k",
            linewidth=0.5,
        )

    else:
        plt.text(
            0.5,
            0.5,
            "Feature Importance not available",
            ha="center",
            va="center",
            fontsize=12,
        )

    plt.tight_layout()

    sanitized_name = best_model_name.replace(" ", "_")
    plt.savefig(
        os.path.join(output_dir, f"feature_importance_best_model_{sanitized_name}.png"),
        bbox_inches="tight",
    )
    plt.savefig(
        os.path.join(output_dir, f"feature_importance_best_model_{sanitized_name}.pdf"),
        bbox_inches="tight",
    )
    plt.close()


def plot_residuals(y_test, y_pred, best_model_name, output_dir):
    """Residual Plot with formatted axes"""
    plt.figure(figsize=(6, 4))
    residuals = y_test - y_pred

    # Calculate scaling factor for axes
    max_val = max(abs(residuals.max()), abs(residuals.min()), abs(y_pred.max()))
    exponent = np.floor(np.log10(max_val)) if max_val > 0 else 0
    scaling_factor = 10 ** (-exponent) if exponent != 0 else 1

    # Create plot with tab10 colors
    point_color = sns.color_palette("tab10")[0]
    sns.scatterplot(
        x=y_pred * scaling_factor,
        y=residuals * scaling_factor,
        alpha=0.5,
        color=point_color,
        edgecolor="w",
        linewidth=0.3,
    )

    plt.axhline(y=0, color=sns.color_palette("tab10")[3], linestyle="--")

    # Axis labels with scaling notation
    plt.xlabel(f"Predicted Values{f' (x1e{int(exponent)})' if exponent != 0 else ''}")
    plt.ylabel(f"Residuals{f' (x1e{int(exponent)})' if exponent != 0 else ''}")

    plt.tight_layout()

    sanitized_name = best_model_name.replace(" ", "_")
    plt.savefig(
        os.path.join(output_dir, f"residual_plot_best_model_{sanitized_name}.png"),
        bbox_inches="tight",
    )
    plt.savefig(
        os.path.join(output_dir, f"residual_plot_best_model_{sanitized_name}.pdf"),
        bbox_inches="tight",
    )
    plt.close()


if __name__ == "__main__":
    start = datetime.now()

    csv_files = ["datasets/syr2k/all_SM_for_LLM.csv", "datasets/syr2k/all_XL_for_LLM.csv"]
    # training_sizes = [100, 200, 300, 400, 500, 600, 0.8]
    # significant_digits_list = [x for x in range(1, 6)]

    # training_sizes = [100, 500, 1000, 5000, 0.8]
    training_sizes = [0.8]
    significant_digits_list = [6]

    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    counter = 1
    total = len(csv_files) * len(training_sizes) * len(significant_digits_list)

    for csv_file_path, training_size, significant_digits in itertools.product(
        csv_files, training_sizes, significant_digits_list
    ):
        output_dir = os.path.join(
            "output",
            run_timestamp,
            os.path.splitext(os.path.basename(csv_file_path))[0],
            f"training_size_{training_size if training_size != 0.8 else '0_8'}",
            f"significant_digits_{significant_digits}",
        )
        os.makedirs(output_dir, exist_ok=True)

        print("=" * 80)
        print(f"Performing for the following configuration ({counter} / {total}):")
        print(f"CSV File: {csv_file_path}")
        print(f"Output Directory: {output_dir}")
        print(f"Training Size: {training_size}")
        print(f"Significant Digits: {significant_digits}")

        results_df, best_model, best_model_name, feature_columns, y_pred, y_test = (
            perform_tuning(csv_file_path, output_dir, training_size, significant_digits)
        )

        make_plots(
            results_df,
            best_model,
            best_model_name,
            feature_columns,
            y_pred,
            y_test,
            output_dir,
        )

        print(f"\n\nElapsed Time: {datetime.now() - start}\n\n")
        counter += 1
