# Imports
import pandas as pd
import numpy as np

# ----------- Statistics -------------------- #
from scipy.stats import spearmanr

# Tests
from scipy.stats import chi2_contingency

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Re-export modules
__all__ = ["pd", "np", "spearmanr", "chi2_contingency", "plt", "sns"]


# Functions
# ----------------------- Function for hypothesis testing ---------------------------------- #
def hypothesis_testing(p_value, alpha=0.05):
    """
    Prints decision to reject/accept null hypothesis given a significance level and the p-value of the statistical test
    """
    if p_value < alpha:
        print(
            f"Given p value = {p_value} is smaller than alpha = {alpha}, the null hypothesis is rejected."
        )
    else:
        print(
            f"Given p value = {p_value} is greater than alpha = {alpha}, the null hypothesis fails to be rejected."
        )


def bootstrapping(data, num_samples):
    """
    Takes array, number of iterations (means to compute) and returns 95% CI of the mean
    Arguments:
        - Data: np.array
        - CI, number iter
    """
    bootstrap_means = np.zeros(num_samples)
    for idx_iter in range(num_samples):
        # Sampling from sample
        bootstrap_means[idx_iter] = np.random.choice(
            data, size=data.size, replace=True
        ).mean()

    # np.percentile: computes percentile of data without need to sort
    low_bound = np.percentile(bootstrap_means, 2.5)
    upper_bound = np.percentile(bootstrap_means, 97.5)
    return (low_bound, upper_bound)


def replace_category(df, target_category, new_category):
    # Find articles with the target category
    articles_to_modify = df[df["category1"] == target_category]["article"].unique()

    # Create a DataFrame to hold modified entries with the new category
    modified_entries = pd.DataFrame(
        {
            "article": articles_to_modify,
            "category1": new_category,
            "category2": new_category,
        }
    )

    # Remove all instances of these articles from the original DataFrame
    df_modified = df[~df["article"].isin(modified_entries["article"])]

    # Add the modified entries back to the DataFrame
    df_result = pd.concat([df_modified, modified_entries]).reset_index(drop=True)

    return df_result
