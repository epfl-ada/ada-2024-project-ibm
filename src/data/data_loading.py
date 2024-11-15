import pandas as pd

from src.utils.utils import replace_category


def load_articles(file_path="./dataset/articles.tsv"):
    """Load links between articles"""
    column_names = ["article"]
    df = pd.read_csv(file_path, names=column_names, sep="\t", comment="#")
    return df


def load_links(file_path="./dataset/links.tsv"):
    """Load links between articles"""
    column_names = ["linkSource", "linkTarget"]
    df = pd.read_csv(file_path, names=column_names, sep="\t", comment="#")
    return df


def load_unfinished_paths(file_path="./dataset/paths_unfinished.tsv"):
    """Load and process the unfinished paths dataset."""

    df_categories = load_categories()
    column_names = [
        "hashedIpAddress",
        "timestamp",
        "durationInSec",
        "path",
        "target",
        "type",
    ]
    df = pd.read_csv(file_path, names=column_names, sep="\t", comment="#")

    # Add 'source' and 'quitting_article' columns
    df["source"] = df["path"].str.split(";").str[0]
    df["quitting_article"] = df["path"].str.split("[;,<]", regex=True).str[-1]

    df = df.merge(
        df_categories[["article", "category1", "category2"]],
        left_on="target",
        right_on="article",
        how="left",
    )

    return df


def load_finished_paths(file_path="./dataset/paths_finished.tsv"):
    """Load and process the finished paths dataset."""
    df_categories = load_categories()
    column_names = ["hashedIpAddress", "timestamp", "durationInSec", "path", "rating"]
    df = pd.read_csv(file_path, names=column_names, sep="\t", comment="#")

    # Add 'source' and 'target' columns
    df["source"] = df["path"].str.split(";").str[0]
    df["target"] = df["path"].str.split("[;,<]", regex=True).str[-1]

    df = df.merge(
        df_categories[["article", "category1", "category2"]],
        left_on="target",
        right_on="article",
        how="left",
    )

    return df


def load_categories(file_path="./dataset/categories.tsv"):
    """Load and process the categories dataset."""
    # Accessing categories
    categories_column_names = ["article", "category"]
    df_categories = pd.read_csv(
        file_path, sep="\t", names=categories_column_names, comment="#"
    )  # ignore metadata
    # Eliminate subject. from field 'category'
    df_categories["category"] = df_categories["category"].str.replace(
        "subject.", "", regex=True
    )

    # Create from 'category' two columns; ' category1' for main category ; 'category2' for secondary category
    df_categories["category1"] = df_categories["category"].str.split(".").str[0]
    # Apply lambda function to check whether category is hierarchical or just one identifier
    # No need to add .str: lambda function splits each string independently, so x in the function represents a single string, not the entire Series.
    df_categories["category2"] = df_categories["category"].apply(
        lambda x: x.split(".")[1] if len(x.split(".")) > 1 else x.split(".")[0]
    )
    # Drop 'category' column
    df_categories = df_categories.drop("category", axis=1)

    # ------------- Dealing with duplicated entries for article in 'Category' ----------------#

    # 1. Countries -> 2 categories ('Countries', 'Geography')
    # Discard 'Geography' category, retain 'Countries'
    df_categories = replace_category(
        df=df_categories, target_category="Countries", new_category="Countries"
    )

    # 2. Duplicated 'People' entries

    duplicated_people_category2 = {
        "Benjamin_of_Tudela": "Geographers_and_explorers",  # History
        "King_Arthur": "Historical_figures",  # History
        "Laika": "Astronomers_and_physicists",  # Science
        "Alfred_the_Great": "Monarchs_of_Great_Britain",  # History
        "Attila_the_Hun": "Historical_figures",  # History
        "Benjamin_Franklin": "Human_Scientists",  # History
        "Boyle_Roche": "Political_People",  # History
        "Edward_the_Confessor": "Monarchs_of_Great_Britain",  # History
        "Galileo_Galilei": "Astronomers_and_physicists",  # Science
        "James_Garfield": "USA_Presidents",  # History
        "James_Watt": "Engineers_and_inventors",  # Science
        "Leonardo_da_Vinci": "Artists",  # Art
        "Nikola_Tesla": "Engineers_and_inventors",  # Science
        "Oliver_Cromwell": "Political_People",  # History
        "Ulysses_S._Grant": "USA_Presidents",  # History
        "William_Ewart_Gladstone": "Political_People",  # History
    }

    # Select articles corresponding to above historical figures and with People entry
    df_relevant = df_categories[
        (df_categories["article"].isin(duplicated_people_category2.keys()))
        & (df_categories["category1"] == "People")
    ].drop_duplicates(
        subset="article"
    )  # Arbitrary elimination, since new classification will be done accordint to own criteria
    # Replace 'category2' by new classification
    df_relevant["category2"] = df_relevant["article"].map(duplicated_people_category2)
    # In original df, eliminate those article entries and add new ones
    df_categories = df_categories[
        ~df_categories["article"].isin(df_relevant["article"])
    ]
    # Concatenate with entires with new 'category2'
    df_categories = pd.concat([df_categories, df_relevant]).reset_index(drop=True)

    # Define a mapping dictionary for category2 to category1
    category2_to_category1 = {
        "Historical_figures": "History",
        "Monarchs_of_Great_Britain": "History",
        "Military_People": "History",  # Otherwise: Citizenship -> Connected to Politics
        "Mathematicians": "Mathematics",
        "Astronomers_and_physicists": "Science",
        "Chemists": "Science",
        "Engineers_and_inventors": "Science",
        "Computing_People": "Science",
        "Human_Scientists": "History",
        "Artists": "Art",
        "Writers_and_critics": "Language_and_literature",
        "Producers_directors_and_media_figures": "People",
        "Performers_and_composers": "Music",
        "Actors_models_and_celebrities": "People",
        "USA_Presidents": "History",
        "Political_People": "History",
        "Religious_figures_and_leaders": "History",
        "Geographers_and_explorers": "History",
        "Sports_and_games_people": "People",
        "Philosophers": "History",
    }

    # Filter the DataFrame in one step to retain only articles with relevant 'category2' values and 'category1' as "People"
    df_relevant = df_categories[
        (df_categories["category2"].isin(category2_to_category1.keys()))
        & (df_categories["category1"] == "People")
    ]

    modified_entries_df = pd.DataFrame(
        {
            "article": df_relevant["article"],
            "category1": df_relevant["category1"],
            "category2": df_relevant["category2"],
        }
    )

    # Map `category2` to `category1` for relevant entries
    modified_entries_df["category1"] = (
        modified_entries_df["category2"]
        .map(category2_to_category1)
        .combine_first(modified_entries_df["category1"])
    )
    # Remove all corresponding articles from dataset
    df_categories = df_categories[
        ~df_categories["article"].isin(modified_entries_df["article"])
    ]
    # Step 6: Concatenate back the cleaned and relevant data
    df_categories = pd.concat([df_categories, modified_entries_df]).reset_index(
        drop=True
    )

    # 3. Duplicates for not 'People' entries according to category hierarchy

    # Define the hierarchy for `category1`
    category_hierarchy = {
        "Everyday_life": 1,
        "Language_and_literature": 2,
        "Music": 3,
        "History": 4,
        "Art": 5,
        "Mathematics": 6,
        "Science": 7,
        "IT": 8,
        "Design_and_Technology": 9,
        "Business_Studies": 10,
        "Geography": 11,
        "Religion": 12,
        "Citizenship": 13,
        "Countries": 14,
    }

    # Add a `rank` column based on the hierarchy
    df_categories["rank"] = df_categories["category1"].map(category_hierarchy)

    # Sort by article and rank, then drop duplicates by keeping the highest priority category for each article
    # If some articles have duplicated 'category1' and different 'category2'
    # Ordered from lowest to highest rank-> First, paired alphabetically and then by rank ; and keep only first entry; then drop rank
    df_categories = (
        df_categories.sort_values(by=["article", "rank"])
        .drop_duplicates(subset="article", keep="first")
        .drop(columns=["rank"])
        .reset_index(drop=True)
    )

    # Add categories for missing articles
    df_categories.loc[len(df_categories)] = ["Pikachu", "Everyday_life", "Games"]
    df_categories.loc[len(df_categories)] = [
        "Friend_Directdebit",
        "Business_Studies",
        "Economics",
    ]
    df_categories.loc[len(df_categories)] = [
        "Directdebit",
        "Business_Studies",
        "Economics",
    ]
    df_categories.loc[len(df_categories)] = [
        "Sponsorship_Directdebit",
        "Business_Studies",
        "Economics",
    ]

    return df_categories
