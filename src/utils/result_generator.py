from src.utils.utils import *

# Imports
import pandas as pd
import numpy as np
import networkx as nx
import random

# ----------- Statistics -------------------- #
from scipy.stats import spearmanr
from scipy.stats import linregress

# Tests
from scipy.stats import chi2_contingency

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
from adjustText import adjust_text


# ---------------------------------------- CATEGORIES ---------------------------------------------#


def plot_main_categories_prop_unfinished_games(df_unfinished, df_finished):

    # UNFINISHED PATHS
    # Select 'category1'
    target_categories = df_unfinished["category1"]
    # len(target_categories): 29185 target articles -> 27 NaNs -> 27 articles dont have corresponding category
    # Remove rows with NaN values in target_category
    target_categories_unfinished = target_categories.dropna()
    # Count number of known target categories for all articles in unfinished paths
    target_categories_unfinished = target_categories_unfinished.value_counts()

    # FINISHED PATHS
    target_categories = df_finished["category1"]
    # Remove rows with NaN values in target_category
    target_categories_finished = target_categories.dropna()
    # Count number of known target categories for all articles in finished paths
    target_categories_finished = target_categories_finished.value_counts()

    # Total number of target categories for all articles
    target_categories_total_count = (
        target_categories_finished + target_categories_unfinished
    )
    # Normalize unfinished path category count by total number
    target_categories_unf_norm = (
        target_categories_unfinished / target_categories_total_count
    )

    target_rank = target_categories_unf_norm.rank(ascending=False).sort_values()

    return target_categories_unf_norm


def plot_target_cateries_unf_norm(target_categories_unf_norm):
    # Bar plot for target categories
    target_categories_unf_norm.sort_values().plot(kind="bar", figsize=(12, 6))
    plt.title("Proportion of Unfinished Games by Target Category")
    plt.xlabel("Category")
    plt.ylabel("Proportion of unfinished games")
    plt.show()


def prop_nan(df_finished):
    nan_counts_all = df_finished["rating"].isna().sum()
    print(
        f" Proportion of NaN for 'rating' column:\n {nan_counts_all/len(df_finished)*100:.2f} %"
    )


def target_barplot_cat_difficulty(df_finished):
    # Drop entries (games) with no 'rating'
    df_finished_categories = df_finished.dropna(subset=["rating"]).reset_index(
        drop=True
    )

    # Computing average 'rating' for every category
    df_difficulty = df_finished[["rating", "category1"]]
    difficulty_target = df_difficulty.groupby("category1").value_counts().unstack()
    weighted_sum = (difficulty_target * difficulty_target.columns).sum(axis=1)
    total_count = difficulty_target.sum(axis=1)
    average_rating = weighted_sum / total_count

    return average_rating


def plot_target_rating_rank(average_rating):

    # Bar plot for target categories
    average_rating.sort_values().plot(kind="bar", figsize=(12, 6))
    plt.title("Diffiuclty of Target Category")
    plt.xlabel("Category")
    plt.ylabel("Average difficulty")
    plt.show()


def results_categories(target_categories_unf_norm, average_rating):
    # I. Calculate Spearman correlation
    correlation, p_value = spearmanr(target_categories_unf_norm, average_rating)

    print("Spearman correlation:", correlation)
    print("P-value:", p_value)

    # II. Perform linear regression (with log transformation)
    slope, intercept, r_value, p_value, std_err = linregress(
        np.log(target_categories_unf_norm), np.log(average_rating)
    )

    # Create the regression line
    x_vals = np.array(np.log(target_categories_unf_norm))
    y_vals_original = intercept + slope * x_vals

    # Use a colormap for each category
    categories = target_categories_unf_norm.index
    colormap = cm.get_cmap("tab20", len(categories))
    colors = {category: colormap(i) for i, category in enumerate(categories)}

    # Plot
    plt.figure(figsize=(8, 5))
    plt.scatter(
        np.log(target_categories_unf_norm),
        np.log(average_rating),
        c=[colors[category] for category in categories],
        label="Main category of target",
    )
    plt.plot(
        x_vals,
        y_vals_original,
        color="red",
        label=f"Regression line (R² = {r_value**2:.2f})",
    )

    # Add labels with adjustText to avoid overlap
    texts = [
        plt.text(x, y, category, fontsize=9)
        for category, x, y in zip(
            categories, np.log(target_categories_unf_norm), np.log(average_rating)
        )
    ]
    adjust_text(texts, arrowprops=dict(arrowstyle="->", color="gray", lw=0.5))

    plt.xlabel("log(Proportion of unfinished games)")
    plt.ylabel("log(Difficulty (average rating))")
    plt.title(
        "Log-Log Plot of Difficulty vs. Proportion of Unfinished Games (Target Category)"
    )
    plt.legend()
    plt.grid(True)
    plt.show()


def top_50_categories(df_finished, df_unfinished):
    # TOP SECONDARY CATEGORIES (with more counts)

    # Unfinished paths
    target_categories = df_unfinished["category2"]
    target_categories_unfinished = target_categories.dropna()
    target_categories_unfinished = target_categories_unfinished.value_counts()

    # Finished paths
    target_categories = df_finished["category2"]
    target_categories_finished = target_categories.dropna()
    target_categories_finished = target_categories_finished.value_counts()

    # Total number of target categories for all articles
    target_categories_total_count = (
        target_categories_finished + target_categories_unfinished
    )
    # Get the top 50 categories in `category2` based on their counts
    top_categories = (
        (target_categories_finished + target_categories_unfinished)
        .sort_values(ascending=False)[:50]
        .index
    )

    # Filter the DataFrame to include only rows with `category2` in the top categories
    df_target_categories_unfinished = df_unfinished[
        df_unfinished["category2"].isin(top_categories)
    ]
    df_target_categories_finished = df_finished[
        df_finished["category2"].isin(top_categories)
    ]
    return df_target_categories_finished, df_target_categories_unfinished


def subcategories_metrics(
    df_target_categories_unfinished, df_target_categories_finished
):

    # ---------------- Proportion fo unfinished games ----------------- #
    # Select 'category2'

    # Unfinished paths
    # Select 'category2'
    target_categories = df_target_categories_unfinished["category2"]
    # len(target_categories): 29185 target articles -> 27 NaNs -> 27 articles dont have corresponding category
    # Remove rows with NaN values in target_category
    target_categories_unfinished = target_categories.dropna()
    # Count number of known target categories for all articles in unfinished paths
    target_categories_unfinished = target_categories_unfinished.value_counts()

    # Finished paths
    target_categories = df_target_categories_finished["category2"]
    # Remove rows with NaN values in target_category
    target_categories_finished = target_categories.dropna()
    # Count number of known target categories for all articles in finished paths
    target_categories_finished = target_categories_finished.value_counts()

    # Total number of target categories for all articles
    target_categories_total_count = (
        target_categories_finished + target_categories_unfinished
    )
    # Normalize unfinished path category count by total number
    target_categories_unf_norm_2 = (
        target_categories_unfinished / target_categories_total_count
    )

    # ------------- Average rating ------------------------- #

    # Computing average 'rating' for every category
    df_difficulty = df_target_categories_finished[["rating", "category2"]]
    difficulty_target = df_difficulty.groupby("category2").value_counts().unstack()
    weighted_sum = (difficulty_target * difficulty_target.columns).sum(axis=1)
    total_count = difficulty_target.sum(axis=1)
    average_rating_2 = weighted_sum / total_count

    return target_categories_unf_norm_2, average_rating_2


def filtering_finished_ratio(df_finished, df_unfinished):
    target_finished = df_finished["target"].value_counts()
    target_unfinished = df_unfinished["target"].value_counts()
    ratio = target_unfinished / target_finished

    # If NaN: not present in finished -> 0
    ratio = ratio.fillna(0)
    ratio_articles = list(ratio[ratio < 0.5].index)
    # FIlter df accorsing to list
    df_finished_filtered = df_finished[df_finished["target"].isin(ratio_articles)]
    df_unfinished_filtered = df_unfinished[df_unfinished["target"].isin(ratio_articles)]
    return df_finished_filtered, df_unfinished_filtered


def results_subcategories(target_categories_unf_norm_2, average_rating_2):
    # I. Calculate Spearman correlation
    correlation, p_value = spearmanr(target_categories_unf_norm_2, average_rating_2)

    print("Spearman correlation:", correlation)
    print("P-value:", p_value)

    # II. Perform linear regression (with log transformation)
    categories = target_categories_unf_norm_2.index
    slope, intercept, r_value, p_value, std_err = linregress(
        np.log(target_categories_unf_norm_2), np.log(average_rating_2)
    )

    # Create the regression line
    x_vals = np.array(np.log(target_categories_unf_norm_2))
    y_vals_original = intercept + slope * x_vals

    # Plot
    plt.figure(figsize=(8, 5))
    plt.scatter(
        np.log(target_categories_unf_norm_2),
        np.log(average_rating_2),
        color="blue",
        label="Secondary category of target",
    )
    plt.plot(
        x_vals,
        y_vals_original,
        color="red",
        label=f"Regression line (R² = {r_value**2:.2f})",
    )

    # Add labels with adjustText to avoid overlap
    texts = [
        plt.text(x, y, category, fontsize=6)
        for category, x, y in zip(
            categories, np.log(target_categories_unf_norm_2), np.log(average_rating_2)
        )
    ]
    adjust_text(texts, arrowprops=dict(arrowstyle="->", color="gray", lw=0.5))

    plt.xlabel("log(Proportion of unfinished games)")
    plt.ylabel("log(Difficulty (average rating))")
    plt.title(
        "Log-Log Plot of Difficulty vs. Proportion of Unfinished Games (Target Category)"
    )
    plt.legend()
    plt.grid(True)
    plt.show()


##### ------------------------------ 2. GRAPH ------------------------------------------------------###


def graph_creation(df_categories, df_links):
    # Create dictionary with connections
    dict_connections = {}
    # Filter articles without category
    articles = df_categories["article"].unique()

    for article in articles:
        # Edges
        list_with_connections = df_links[df_links["linkSource"] == article][
            "linkTarget"
        ].unique()
        # Attributes
        main_category = df_categories[df_categories["article"] == article][
            "category1"
        ].values[0]
        secondary_category = df_categories[df_categories["article"] == article][
            "category2"
        ].values[0]
        # Dict creation
        dict_connections[article] = [
            list_with_connections,
            main_category,
            secondary_category,
        ]

    # Initialize a graph
    G = nx.DiGraph()

    # Add edges from the connections dictionary
    for article, characteristics_article in dict_connections.items():

        # Add the node with attributes
        G.add_node(
            article,
            main_category=characteristics_article[1],
            secondary_category=characteristics_article[2],
        )
        # Add edges
        linked_articles = characteristics_article[0]
        for linked_article in linked_articles:
            G.add_edge(article, linked_article)

    return G, dict_connections


def categories_node_metric(df_categories, closeness_centrality, pagerank, G):
    main_categories = list(df_categories["category1"].unique())
    category_metrics = {}
    for category in main_categories:
        # Analyse nodes by subsets determined by main category
        filtered_nodes = [
            node
            for node, attr in G.nodes(data=True)
            if attr.get("main_category") == category
        ]

        # Extract metrics for filtered nodes
        category_closeness_centrality = [
            closeness_centrality[node] for node in filtered_nodes
        ]
        category_pagerank = [pagerank[node] for node in filtered_nodes]

        # Compute average metrics
        avg_closeness = np.mean(category_closeness_centrality)
        avg_pagerank = np.mean(category_pagerank)

        # Add metrics to the dictionary
        category_metrics[category] = {
            "Average Closeness Centrality": avg_closeness,
            "Average PageRank": avg_pagerank,
        }

    # Convert the dictionary to a pandas DataFrame
    df_metrics = pd.DataFrame.from_dict(
        category_metrics
    ).T  # Transpose so categories are rows
    return df_metrics


def regression_node_metrics(df_metrics, target_categories_unf_norm):
    metrics = df_metrics.columns

    for metric in metrics:
        # Otherwise not correctly aligned
        aligned_metrics = df_metrics[metric].reindex(target_categories_unf_norm.index)

        slope, intercept, r_value, p_value, std_err = linregress(
            target_categories_unf_norm, aligned_metrics
        )

        # Create the regression line
        x_vals = np.array(target_categories_unf_norm)
        y_vals_original = intercept + slope * x_vals

        # Use a colormap for each category
        categories = target_categories_unf_norm.index
        colormap = cm.get_cmap("tab20", len(categories))
        colors = {category: colormap(i) for i, category in enumerate(categories)}

        # Plot
        plt.figure(figsize=(10, 6))
        plt.scatter(
            target_categories_unf_norm,
            aligned_metrics,
            c=[colors[category] for category in categories],
            label="Main category of target article",
        )
        plt.plot(
            x_vals,
            y_vals_original,
            color="red",
            label=f"Regression Line (R² = {r_value**2:.2f})",
        )

        # Add labels with adjustText to avoid overlap
        texts = [
            plt.text(x, y, category, fontsize=9)
            for category, x, y in zip(
                categories, target_categories_unf_norm, aligned_metrics
            )
        ]
        adjust_text(texts, arrowprops=dict(arrowstyle="->", color="gray", lw=0.5))

        plt.xlabel("Proportion of unfinished games")
        plt.ylabel(metric)
        plt.title(f"{metric} vs. Proportion of Unfinished Games")
        plt.legend()
        plt.grid(True)
        plt.show()


# FULL PATH: to see how network traverses


def get_shortest_path_category(row, graph, dict_connections):  # Implicit arg. row
    try:

        path = nx.shortest_path(graph, source=row["source"], target=row["target"])
        categories_path = [dict_connections[article][1] for article in path]
        return categories_path
    except nx.NetworkXNoPath:
        print("No path exists")
        return np.nan  # No path exists


def get_shortest_path(row, graph):
    try:
        # Compute the shortest path as article names
        path = nx.shortest_path(graph, source=row["source"], target=row["target"])
        return path
    except nx.NetworkXNoPath:
        return np.nan  # No path exists


def generator_shortest_paths(
    df_finished, df_unfinished, unique_categories, G, df_categories, dict_connections
):

    df_source_target = df_unfinished[
        ["source", "target"]
    ].drop_duplicates()  # keep each combination once
    # Add one col for category
    for category in unique_categories:
        df_source_target[category] = 0  # Initialize with 0 or any other value
    df_source_target["shortest_path_categories"] = df_source_target.apply(
        get_shortest_path_category, dict_connections=dict_connections, graph=G, axis=1
    )
    df_source_target["shortest_path_articles"] = df_source_target.apply(
        get_shortest_path, graph=G, axis=1
    )

    # If NaN: np path ; store number and elimnate entries
    nan_entries = df_source_target["shortest_path_articles"].isna()
    no_path_games = df_source_target[nan_entries]
    df_source_target = df_source_target[~nan_entries]
    df_source_target["len_shortest_path"] = df_source_target[
        "shortest_path_articles"
    ].apply(len)

    # Update counts for each category column
    for index, row in df_source_target.iterrows():
        # Count occurrences in shortest_path_categories
        counts = pd.Series(
            row["shortest_path_categories"]
        ).value_counts()  # gives category: count

        # Update the DataFrame for each category
        for category in counts.index:
            if category in df_source_target.columns:
                df_source_target.at[index, category] = counts.get(
                    category, 0
                )  # So default to 0 if category is missing

    article_proprortion_category = df_categories["category1"].value_counts()
    article_proprortion_category_normalized = (
        article_proprortion_category / article_proprortion_category.sum()
    )

    df_path_network_unfinished = df_source_target

    df_source_target = df_finished[
        ["source", "target"]
    ].drop_duplicates()  # keep each combination once
    # Add one col for category
    for category in unique_categories:
        df_source_target[category] = 0  # Initialize with 0 or any other value
    df_source_target["shortest_path_categories"] = df_source_target.apply(
        get_shortest_path_category, dict_connections=dict_connections, graph=G, axis=1
    )
    df_source_target["shortest_path_articles"] = df_source_target.apply(
        get_shortest_path, graph=G, axis=1
    )
    df_source_target["len_shortest_path"] = df_source_target[
        "shortest_path_articles"
    ].apply(len)

    # Update counts for each category column
    for index, row in df_source_target.iterrows():
        # Count occurrences in shortest_path_categories
        counts = pd.Series(
            row["shortest_path_categories"]
        ).value_counts()  # gives category: count

        # Update the DataFrame for each category
        for category in counts.index:
            if category in df_source_target.columns:
                df_source_target.at[index, category] = counts.get(
                    category, 0
                )  # So default to 0 if category is missing

    article_proprortion_category = df_categories["category1"].value_counts()
    article_proprortion_category_normalized = (
        article_proprortion_category / article_proprortion_category.sum()
    )

    df_path_network_finished = df_source_target

    return df_path_network_finished, df_path_network_unfinished


def boxplot_shortest_path_length(df_path_network_finished, df_path_network_unfinished):
    finished_lengths = df_path_network_finished["len_shortest_path"]
    unfinished_lengths = df_path_network_unfinished["len_shortest_path"]
    # Create the boxplot with specified colors
    plt.figure(figsize=(8, 6))
    boxprops = dict(color="blue", linewidth=1.5)
    medianprops = dict(color="blue", linewidth=2)
    plt.boxplot(
        finished_lengths,
        positions=[1],
        labels=["Finished Games"],
        boxprops=boxprops,
        medianprops=medianprops,
    )

    boxprops = dict(color="red", linewidth=1.5)
    medianprops = dict(color="red", linewidth=2)
    plt.boxplot(
        unfinished_lengths,
        positions=[2],
        labels=["Unfinished Games"],
        boxprops=boxprops,
        medianprops=medianprops,
    )

    # Add title and labels
    plt.title("Shortest Path Lengths: Finished vs Unfinished Games")
    plt.ylabel("Shortest Path Length")
    plt.xticks([1, 2], ["Finished Games", "Unfinished Games"])

    # Show plot
    plt.show()


def shortest_path_length_difficulty_ci(
    df_path_network_unfinished, df_unfinished, df_path_network_finished, df_finished
):
    path_lengths = [3, 4, 5, 6, 7]  # No 8 path length in finished
    mean_ratio_unfinished = []
    mean_ratings = []
    confidence_intervals = []  # To store confidence intervals for mean ratios

    # Number of bootstrap samples
    n_bootstrap = 1000
    ci_percentile = 95  # Confidence interval percentile
    threshold = 0.5  # Minimum ratio of finished to unfinished gamess

    mean_ratio_unfinished = []
    mean_ratings = []
    confidence_intervals = []  # To store confidence intervals for mean ratios
    total_counts = []
    # Iterate over the desired path lengths
    for path_length in path_lengths:

        # Unfinished paths
        filtered_unfinished_network = df_path_network_unfinished[
            df_path_network_unfinished["len_shortest_path"] == path_length
        ]
        source_target_combinations_unfinished = filtered_unfinished_network[
            ["source", "target"]
        ]  # combinations of (source, target)
        filtered_df_unfinished = df_unfinished.merge(
            source_target_combinations_unfinished, on=["source", "target"], how="inner"
        )

        # Finished paths
        filtered_finished_network = df_path_network_finished[
            df_path_network_finished["len_shortest_path"] == path_length
        ]
        source_target_combinations_finished = filtered_finished_network[
            ["source", "target"]
        ]  # combinations of (source, target)
        filtered_df_finished = df_finished.merge(
            source_target_combinations_finished, on=["source", "target"], how="inner"
        )

        # Count occurrences for unfinished and finished paths
        count_unfinished = (
            filtered_df_unfinished.groupby(["source", "target"])
            .size()
            .reset_index(name="count_unfinished")
        )
        count_finished = (
            filtered_df_finished.groupby(["source", "target"])
            .size()
            .reset_index(name="count_finished")
        )

        # Merge the counts for unfinished and finished paths
        merged_counts = count_unfinished.merge(
            count_finished, on=["source", "target"], how="outer"
        ).fillna(0)
        merged_counts["ratio_unfinished"] = merged_counts["count_unfinished"] / (
            merged_counts["count_unfinished"] + merged_counts["count_finished"]
        )

        # Apply the threshold: keep only rows where the ratio of finished to unfinished meets the threshold
        merged_counts = merged_counts[
            (
                merged_counts["count_finished"]
                / (merged_counts["count_unfinished"] + 1e-10)
            )
            >= threshold
        ]

        # Compute total counts (sum of finished + unfinished for this path length)
        total_count = (
            merged_counts["count_unfinished"].sum()
            + merged_counts["count_finished"].sum()
        )
        total_counts.append(total_count)  # Store the total count for annotation

        # Perform bootstrap sampling
        bootstrap_ratios = []
        for _ in range(n_bootstrap):
            sample = merged_counts.sample(
                frac=1, replace=True
            )  # Resample with replacement
            bootstrap_ratios.append(sample["ratio_unfinished"].mean())

        # Compute the confidence interval for the mean ratio
        lower_bound = np.percentile(bootstrap_ratios, (100 - ci_percentile) / 2)
        upper_bound = np.percentile(bootstrap_ratios, 100 - (100 - ci_percentile) / 2)
        confidence_intervals.append((lower_bound, upper_bound))

        # Compute the mean ratio and mean rating
        mean_ratio = merged_counts["ratio_unfinished"].mean()
        mean_rating = filtered_df_finished["rating"].mean()

        # Store results
        mean_ratio_unfinished.append(mean_ratio)
        mean_ratings.append(mean_rating)

    slope, intercept, r_value, p_value, std_err = linregress(
        mean_ratings, mean_ratio_unfinished
    )

    # R^2 value is the square of the correlation coefficient (r_value)
    r_squared = r_value**2
    print(f"R^2 value: {r_squared:.4f}")

    # Calculate yerr for error bars
    yerr = np.array(
        [
            [mean_ratio - lower, upper - mean_ratio]
            for (mean_ratio, (lower, upper)) in zip(
                mean_ratio_unfinished, confidence_intervals
            )
        ]
    ).T
    print(f"threshold:{threshold}")

    # Plot mean ratio of unfinished paths vs. path length with confidence intervals
    plt.figure(figsize=(8, 6))

    # Add annotations for total counts
    for i, txt in enumerate(total_counts):
        plt.annotate(
            f"n={txt}",
            (mean_ratings[i], mean_ratio_unfinished[i]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=10,
            color="black",
        )
    plt.errorbar(
        mean_ratings,
        mean_ratio_unfinished,
        yerr=yerr,
        fmt="o",
        linestyle="-",
        linewidth=2,
        markersize=8,
        capsize=5,
    )
    plt.title(
        "Mean Ratio of Unfinished Paths vs Path Length with Confidence Intervals",
        fontsize=14,
    )
    plt.xlabel("Average Difficulty", fontsize=12)
    plt.ylabel("Ratio of Unfinished Paths", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()


def analysis_shortest_path_metrics(
    df_path_network_unfinished,
    df_path_network_finished,
    pagerank_graph,
    closeness_graph,
):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    df_path_network_unfinished = df_path_network_unfinished[
        df_path_network_unfinished["len_shortest_path"] < 8
    ]
    df_network_unfinished = df_path_network_unfinished[
        df_path_network_unfinished["len_shortest_path"] > 2
    ]
    df_network_grouped_length = df_network_unfinished.groupby("len_shortest_path")
    for length_type, games in df_network_grouped_length:
        # For each path length
        num_games = len(games)
        path_length = length_type

        # Initialize an array to store betweenness centrality values
        pagerank_array = np.zeros((num_games, path_length))

        # Iterate over games
        for i, (_, game) in enumerate(games.iterrows()):
            path_articles = game["shortest_path_articles"]

            # Compute betweenness centrality for each node in the path
            pagerank = [pagerank_graph[node] for node in path_articles]

            # Store the betweenness centrality for this game's path
            pagerank_array[i, :] = np.array(pagerank)
            # plt.plot(range(1, path_length + 1), betweenness, marker='o', label=f'Path Length {path_length}')

        # Compute mean betweenness centrality for each position in the path
        pagerank_mean = np.median(pagerank_array, axis=0)

        # Plot mean betweenness centrality vs node position
        axs[0].plot(
            range(1, path_length + 1),
            pagerank_mean,
            marker="o",
            label=f"Path Length {path_length}",
        )

    df_path_network_finished = df_path_network_finished[
        df_path_network_finished["len_shortest_path"] < 8
    ]
    df_network_finished = df_path_network_finished[
        df_path_network_finished["len_shortest_path"] > 2
    ]
    df_network_grouped_length = df_network_finished.groupby("len_shortest_path")
    for length_type, games in df_network_grouped_length:
        # For each path length
        num_games = len(games)
        path_length = length_type

        # Initialize an array to store closeness centrality values
        closeness_array = np.zeros((num_games, path_length))

        # Iterate over games
        for i, (_, game) in enumerate(games.iterrows()):
            path_articles = game["shortest_path_articles"]

            # Compute closeness centrality for each node in the path
            closeness = [closeness_graph[node] for node in path_articles]

            # Store the closeness centrality for this game's path
            closeness_array[i, :] = np.array(closeness)
            # plt.plot(range(1, path_length + 1), betweenness, marker='o', label=f'Path Length {path_length}')

        # Compute mean closness centrality for each position in the path
        closeness_mean = np.median(closeness_array, axis=0)

        # Plot mean closenness centrality vs node position
        axs[1].plot(
            range(1, path_length + 1),
            closeness_mean,
            marker="o",
            label=f"Path Length {path_length}",
        )

    # Customize the plot
    axs[0].set_xlabel("Node Position in Path")
    axs[0].set_ylabel("PageRank")
    axs[1].set_xlabel("Node Position in Path")
    axs[1].set_ylabel("Closeness Centrality")

    # Set individual subplot titles
    axs[0].set_title("PageRank")
    axs[1].set_title("Closeness Centrality")

    # Set overall figure title
    fig.suptitle("Median Metrics vs Position in Path (Nodes)")

    axs[0].legend()
    axs[1].legend()

    # Display the plot
    plt.show()


def distance_to_target(actual_node, target, G):

    return len(nx.shortest_path(G, source=actual_node, target=target))


def distance_to_target_plot(
    df_path_network_finished, df_path_network_unfinished, df_finished, df_unfinished, G
):

    fig, axes = plt.subplots(4, 2, figsize=(14, 10), sharex="col", sharey="row")
    fig.subplots_adjust(hspace=0.4, wspace=0.3)  # Adjust spacing between subplots
    path_lengths = [3, 4, 5, 6]

    # Iterate over the desired path lengths
    for idx, path_length in enumerate(path_lengths):
        # Row and column indices for the subplot
        row = idx
        col_finished = 0  # Column for finished paths
        col_unfinished = 1  # Column for unfinished paths

        ### --- FINISHED PATHS --- ###
        # Filter finished network by path length
        filtered_finished_network = df_path_network_finished[
            df_path_network_finished["len_shortest_path"] == path_length
        ]
        source_target_combinations_finished = filtered_finished_network[
            ["source", "target"]
        ]
        filtered_df_finished = df_finished.merge(
            source_target_combinations_finished, on=["source", "target"], how="inner"
        )

        # Group by path length
        finished_group_lengths = filtered_df_finished.groupby("path_length")
        for i, (length, games) in enumerate(finished_group_lengths):
            # Store statistics

            # Initialize an list to store distance to target
            distance_target_finished = []
            for j, (_, game) in enumerate(games.iterrows()):
                # Iterate over games in finished paths
                path = game["path"].split(";")
                # Subs by previous to previous
                for i in range(len(path)):
                    if path[i] == "<":
                        path[i] = path[i - 2]

                # List to store distances for the current game
                distance_target = []
                skip_game = False  # Flag to indicate if the game should be skipped

                for i in range(len(path)):
                    try:
                        distance = distance_to_target(path[i], game["target"])
                        distance_target.append(distance)
                    except Exception as e:
                        # If an error occurs, skip this game
                        skip_game = True
                        break  # Exit the loop as we want to skip the entire game

                if not skip_game:
                    # Append the valid distances to the main list
                    distance_target_finished.append(distance_target)

            # Compute median closeness centrality for each position in the path
            distance_mean_finished = np.median(
                np.array(distance_target_finished), axis=0
            )
            if len(games) > 25:
                # Plot on the finished subplot
                axes[row, col_finished].plot(
                    range(1, length + 1),
                    distance_mean_finished,
                    marker="o",
                    label=f"Path Length {length}",
                    color="blue",
                )
        axes[row, col_finished].set_title(f"Finished: Path Length {path_length}")
        axes[row, col_finished].set_xlabel("Node Position")
        axes[row, col_finished].set_ylabel("Distance to Target Node")
        axes[row, col_finished].grid(True)
        axes[row, col_finished].legend()

        ### --- UNFINISHED PATHS --- ###
        # Filter unfinished network by path length
        filtered_unfinished_network = df_path_network_unfinished[
            df_path_network_unfinished["len_shortest_path"] == path_length
        ]
        source_target_combinations_unfinished = filtered_unfinished_network[
            ["source", "target"]
        ]
        filtered_df_unfinished = df_unfinished.merge(
            source_target_combinations_unfinished, on=["source", "target"], how="inner"
        )

        # Group by path length
        unfinished_group_lengths = filtered_df_unfinished.groupby("path_length")
        for i, (length, games) in enumerate(unfinished_group_lengths):
            # Initialize an list to store distance to target
            distance_target_unfinished = []
            for j, (_, game) in enumerate(games.iterrows()):
                # Iterate over games in finished paths
                path = game["path"].split(";")
                # Subs by previous to previous
                for i in range(len(path)):
                    if path[i] == "<":
                        path[i] = path[i - 2]

                # List to store distances for the current game
                distance_target = []
                skip_game = False  # Flag to indicate if the game should be skipped

                for i in range(len(path)):
                    try:
                        distance = distance_to_target(path[i], game["target"])
                        distance_target.append(distance)
                    except Exception as e:
                        # If an error occurs, skip this game
                        skip_game = True
                        break  # Exit the loop as we want to skip the entire game

                if not skip_game:
                    # Append the valid distances to the main list
                    distance_target_unfinished.append(distance_target)

            # Compute median closeness centrality for each position in the path
            distance_mean_unfinished = np.median(
                np.array(distance_target_unfinished), axis=0
            )
            if len(games) > 25:
                # Plot on the finished subplot
                axes[row, col_unfinished].plot(
                    range(1, length + 1),
                    distance_mean_unfinished,
                    marker="o",
                    label=f"Path Length {length}",
                    color="red",
                )
        axes[row, col_unfinished].set_title(f"Unfinished: Path Length {path_length}")
        axes[row, col_unfinished].set_xlabel("Node Position")
        axes[row, col_unfinished].set_ylabel("Distance to Target Node")
        axes[row, col_unfinished].grid(True)
        axes[row, col_unfinished].legend()

    # Finalize and show the plot
    plt.suptitle(
        "Distance to Target Node by Node Position (Finished vs Unfinished)", fontsize=16
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def pagerank_all_paths(
    df_path_network_finished,
    df_path_network_unfinished,
    df_finished,
    df_unfinished,
    pagerank_graph,
):
    # Storing statistics
    fig, axes = plt.subplots(4, 2, figsize=(14, 10), sharex="col", sharey="row")
    fig.subplots_adjust(hspace=0.4, wspace=0.3)  # Adjust spacing between subplots
    path_lengths = [3, 4, 5, 6]

    # Iterate over the desired path lengths
    for idx, path_length in enumerate(path_lengths):
        # Row and column indices for the subplot
        row = idx
        col_finished = 0  # Column for finished paths
        col_unfinished = 1  # Column for unfinished paths

        ### --- FINISHED PATHS --- ###
        # Filter finished network by path length
        filtered_finished_network = df_path_network_finished[
            df_path_network_finished["len_shortest_path"] == path_length
        ]
        source_target_combinations_finished = filtered_finished_network[
            ["source", "target"]
        ]
        filtered_df_finished = df_finished.merge(
            source_target_combinations_finished, on=["source", "target"], how="inner"
        )

        # Group by path length
        finished_group_lengths = filtered_df_finished.groupby("path_length")
        for i, (length, games) in enumerate(finished_group_lengths):

            # Initialize an array to store closeness centrality
            closeness_array_finished = np.zeros((len(games), length))
            for j, (_, game) in enumerate(games.iterrows()):
                # Iterate over games in finished paths
                path = game["path"].split(";")
                # Subs by previous to previous
                for i in range(len(path)):
                    if path[i] == "<":
                        path[i] = path[i - 2]

                closeness = [pagerank_graph[node] for node in path]
                closeness_array_finished[j, :] = np.array(closeness)

            # Compute median closeness centrality for each position in the path
            closeness_mean_finished = np.median(closeness_array_finished, axis=0)
            if len(games) > 25:
                # Plot on the finished subplot
                axes[row, col_finished].plot(
                    range(1, length + 1),
                    closeness_mean_finished,
                    marker="o",
                    label=f"Path Length {length}",
                    color="blue",
                )
        axes[row, col_finished].set_title(f"Finished: Path Length {path_length}")
        axes[row, col_finished].set_xlabel("Node Position")
        axes[row, col_finished].set_ylabel("Median PageRank")
        axes[row, col_finished].grid(True)
        axes[row, col_finished].legend()

        ### --- UNFINISHED PATHS --- ###
        # Filter unfinished network by path length
        filtered_unfinished_network = df_path_network_unfinished[
            df_path_network_unfinished["len_shortest_path"] == path_length
        ]
        source_target_combinations_unfinished = filtered_unfinished_network[
            ["source", "target"]
        ]
        filtered_df_unfinished = df_unfinished.merge(
            source_target_combinations_unfinished, on=["source", "target"], how="inner"
        )

        # Group by path length
        unfinished_group_lengths = filtered_df_unfinished.groupby("path_length")
        for i, (length, games) in enumerate(unfinished_group_lengths):

            # Initialize an array to store closeness centrality
            closeness_array_unfinished = np.zeros((len(games), length))
            for j, (_, game) in enumerate(games.iterrows()):
                # Iterate over games in unfinished paths
                path = game["path"].split(";")
                for i in range(len(path)):
                    if path[i] == "<":
                        path[i] = path[i - 2]

                closeness = [pagerank_graph[node] for node in path]
                closeness_array_unfinished[j, :] = np.array(closeness)

            # Compute median closeness centrality for each position in the path
            closeness_mean_unfinished = np.median(closeness_array_unfinished, axis=0)
            if len(games) > 25:
                # Plot on the unfinished subplot
                axes[row, col_unfinished].plot(
                    range(1, length + 1),
                    closeness_mean_unfinished,
                    marker="o",
                    label=f"Path Length {length}",
                    color="red",
                )
        axes[row, col_unfinished].set_title(f"Unfinished: Path Length {path_length}")
        axes[row, col_unfinished].set_xlabel("Node Position")
        axes[row, col_unfinished].set_ylabel("Median PageRank")
        axes[row, col_unfinished].grid(True)
        axes[row, col_unfinished].legend()

    # Finalize and show the plot
    plt.suptitle(
        "Median  PageRank by Node Position (Finished vs Unfinished)",
        fontsize=16,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def closeness_all_paths(
    df_path_network_finished,
    df_path_network_unfinished,
    df_finished,
    df_unfinished,
    closeness_graph,
):
    # Storing statistics
    fig, axes = plt.subplots(4, 2, figsize=(14, 10), sharex="col", sharey="row")
    fig.subplots_adjust(hspace=0.4, wspace=0.3)  # Adjust spacing between subplots
    path_lengths = [3, 4, 5, 6]

    # Iterate over the desired path lengths
    for idx, path_length in enumerate(path_lengths):
        # Row and column indices for the subplot
        row = idx
        col_finished = 0  # Column for finished paths
        col_unfinished = 1  # Column for unfinished paths

        ### --- FINISHED PATHS --- ###
        # Filter finished network by path length
        filtered_finished_network = df_path_network_finished[
            df_path_network_finished["len_shortest_path"] == path_length
        ]
        source_target_combinations_finished = filtered_finished_network[
            ["source", "target"]
        ]
        filtered_df_finished = df_finished.merge(
            source_target_combinations_finished, on=["source", "target"], how="inner"
        )

        # Group by path length
        finished_group_lengths = filtered_df_finished.groupby("path_length")
        for i, (length, games) in enumerate(finished_group_lengths):

            # Initialize an array to store closeness centrality
            closeness_array_finished = np.zeros((len(games), length))
            for j, (_, game) in enumerate(games.iterrows()):
                # Iterate over games in finished paths
                path = game["path"].split(";")
                # Subs by previous to previous
                for i in range(len(path)):
                    if path[i] == "<":
                        path[i] = path[i - 2]

                closeness = [closeness_graph[node] for node in path]
                closeness_array_finished[j, :] = np.array(closeness)

            # Compute median closeness centrality for each position in the path
            closeness_mean_finished = np.median(closeness_array_finished, axis=0)
            if len(games) > 25:
                # Plot on the finished subplot
                axes[row, col_finished].plot(
                    range(1, length + 1),
                    closeness_mean_finished,
                    marker="o",
                    label=f"Path Length {length}",
                    color="blue",
                )
        axes[row, col_finished].set_title(f"Finished: Path Length {path_length}")
        axes[row, col_finished].set_xlabel("Node Position")
        axes[row, col_finished].set_ylabel("Median Closeness Centrality")
        axes[row, col_finished].grid(True)
        axes[row, col_finished].legend()

        ### --- UNFINISHED PATHS --- ###
        # Filter unfinished network by path length
        filtered_unfinished_network = df_path_network_unfinished[
            df_path_network_unfinished["len_shortest_path"] == path_length
        ]
        source_target_combinations_unfinished = filtered_unfinished_network[
            ["source", "target"]
        ]
        filtered_df_unfinished = df_unfinished.merge(
            source_target_combinations_unfinished, on=["source", "target"], how="inner"
        )

        # Group by path length
        unfinished_group_lengths = filtered_df_unfinished.groupby("path_length")
        for i, (length, games) in enumerate(unfinished_group_lengths):

            # Initialize an array to store closeness centrality
            closeness_array_unfinished = np.zeros((len(games), length))
            for j, (_, game) in enumerate(games.iterrows()):
                # Iterate over games in unfinished paths
                path = game["path"].split(";")
                for i in range(len(path)):
                    if path[i] == "<":
                        path[i] = path[i - 2]

                closeness = [closeness_graph[node] for node in path]
                closeness_array_unfinished[j, :] = np.array(closeness)

            # Compute median closeness centrality for each position in the path
            closeness_mean_unfinished = np.median(closeness_array_unfinished, axis=0)
            if len(games) > 25:
                # Plot on the unfinished subplot
                axes[row, col_unfinished].plot(
                    range(1, length + 1),
                    closeness_mean_unfinished,
                    marker="o",
                    label=f"Path Length {length}",
                    color="red",
                )
        axes[row, col_unfinished].set_title(f"Unfinished: Path Length {path_length}")
        axes[row, col_unfinished].set_xlabel("Node Position")
        axes[row, col_unfinished].set_ylabel("Median Closeness Centrality")
        axes[row, col_unfinished].grid(True)
        axes[row, col_unfinished].legend()

    # Finalize and show the plot
    plt.suptitle(
        "Median Closeness Centrality by Node Position (Finished vs Unfinished)",
        fontsize=16,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def bias_oversampling(
    df_path_network_finished, df_path_network_unfinished, df_finished, df_unfinished, closeness_graph
):

    # Assuming df_path_network_finished, df_path_network_unfinished, df_finished, df_unfinished, and pagerank_graph are already defined.

    # Subsample size
    subsample_size = 3000

    # Storing statistics
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex="col", sharey="row")
    fig.subplots_adjust(hspace=0.4, wspace=0.3)  # Adjust spacing between subplots
    path_lengths = [4, 5]  # Subset for path lengths 4 and 5

    # Iterate over the desired path lengths
    for idx, path_length in enumerate(path_lengths):
        # Row and column indices for the subplot
        row = idx
        col_finished = 0  # Column for finished paths
        col_unfinished = 1  # Column for unfinished paths

        ### --- FINISHED PATHS --- ###
        # Filter finished network by path length
        filtered_finished_network = df_path_network_finished[
            df_path_network_finished["len_shortest_path"] == path_length
        ]
        source_target_combinations_finished = filtered_finished_network[
            ["source", "target"]
        ]
        filtered_df_finished = df_finished.merge(
            source_target_combinations_finished, on=["source", "target"], how="inner"
        )

        # Randomly subsample 3000 games
        filtered_df_finished = filtered_df_finished.sample(
            n=subsample_size, random_state=42
        )

        # Group by path length
        finished_group_lengths = filtered_df_finished.groupby("path_length")
        for length, games in finished_group_lengths:
            closeness_array_finished = np.zeros((len(games), length))
            for j, (_, game) in enumerate(games.iterrows()):
                path = game["path"].split(";")
                for i in range(len(path)):
                    if path[i] == "<":
                        path[i] = path[i - 2]
                closeness = [closeness_graph[node] for node in path]
                closeness_array_finished[j, :] = np.array(closeness)

            closeness_mean_finished = np.median(closeness_array_finished, axis=0)
            if len(games) > 100:
                axes[row].plot(
                    range(1, length + 1),
                    closeness_mean_finished,
                    marker="o",
                    label=f"Path Length {length}",
                    color="blue",
                )
        axes[row].set_title(f"Finished: Path Length {path_length}")
        axes[row].set_xlabel("Node Position")
        axes[row].set_ylabel("Median Closeness Centrality")
        axes[row].grid(True)
        axes[row].legend()

        ### --- UNFINISHED PATHS --- ###
        # Filter unfinished network by path length
        filtered_unfinished_network = df_path_network_unfinished[
            df_path_network_unfinished["len_shortest_path"] == path_length
        ]
        source_target_combinations_unfinished = filtered_unfinished_network[
            ["source", "target"]
        ]
        filtered_df_unfinished = df_unfinished.merge(
            source_target_combinations_unfinished, on=["source", "target"], how="inner"
        )
        row = row + 1

    # Finalize and show the plot
    plt.suptitle(
        "Median Closeness Centrality by Node Position (Finished) - Subsampled",
        fontsize=16,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
