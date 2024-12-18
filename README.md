# ADA 2024 Project P2


## *Unfinished Stories: What Wikispeedia Reveals About Us*

#### Quickstart

```bash
# clone project
git clone https://github.com/epfl-ada/ada-2024-project-ibm.git


# create conda environment that fulfills requirements
conda env create -f environment.yml


```

#### How to use the library

*results.ipynb* stores the main findings achieved until the moment; the remaining files were the different analysis have been carried out have been included in the *.gitignore* file.

To simulate findings of *results.ipynb* simply run code, having installed the necessary dependencies listed in the .yml file and having stored the dataset files, and data loading and utilities in their corresponding directories, as described below.

#### Project Structure


```
├── data                        <- Dataset files
│
├── src                         <- Source code
│   ├── data                            <- Data loading and preprocessing functions
│   ├── utils                           <- Funtions related to other functionalities
├── results.ipynb               <- notebook summarizing the results
│
├── .gitignore                  <- List of files ignored by git
├── environment.yml             <- File for installing python dependencies
└── README.md
```</pre>
```

## Abstract

This project investigates why users quit the Wikispeedia game—a Wikipedia-based game where users navigate from one article to another using only hyperlinks. By analyzing metrics such as centrality and PageRank in relation to the abandonment rate and topic difficulty, we aim to understand the behaviour of users and their 'quitting' behaviours.

## Research Questions

1. What factors contribute to the difficulty of navigation tasks in Wikispeedia, and how are these related to abandonment rates?
2. How do graph metrics (e.g., PageRank, closeness centrality) correlate with the likelihood of quitting a gama?
3. Which topics or categories (e.g., "Countries," "Everyday Life") exhibit higher abandonment rates, and what are the potential reasons?
4. How do different types of users and their familiarity with topics/game strategies affect the navigation path and completion rates?

## Proposed Additional Datasets

No additional datasets are proposed at this stage. However, if needed, we could explore Wikipedia data to obtain more information regarding topics and articles connectivity.

## Methods

* **Data preprocessing** : filtered, categorized, and restructured data to standardize article properties, ensuring accuracy in hierarchical categories.

  **Data dimension**: have realized limitations due to dimensions of dataset: certain tests and hypothesis cannot be evaluated due to relatively low count and diversity of specific articles in available games. In addition, the connectivity of articles is limited (does not really reflect all semantinc possibilities for associations with other articles, as Wikipedia dataset would).

  The most important modification has been duplicate handling for articles with various categories :

  1. Retained  "Countries" over "Geography" for articles in both, prioritizing country-specific detail.
  2. Reclassification of historical figures : moved historically significant figures from "People" to categories like "History" or "Science," leaving "People" focused on modern pop culture figures.
  3. Hierarchy ranking: established a category ranking (e.g., "Everyday Life" > "Language and Literature") to handle duplicates and ensure consistent categorization.
* **Target category correlations** : performed log-log regression to evaluate the relationship between topic difficulty and the proportion of unfinished games, analyzing differences between categories.

  * Future work: **Deeper analysis of isolated categories**
    * Examine assumptions around specificity (e.g., "Novels" or "Films") versus generality (e.g., "Countries") in difficulty and 'quitting', testing these statistical tests; analyse categories with low average difficulty , but high quitting rates.
* **Graph analysis**: calculated structural metrics such as degree centrality, PageRank, closeness centrality, and betweenness centrality, associating them with each article and its category.

  * Future work: **Investigate user behaviour with path analysis**
    * Analyze patterns of users in game with respect to graph: whether users usually follow shortest paths or explore suboptimal routes, study transitions between categories, analyze variations in paths users take for the same start etc...

## Proposed Timeline for Remaining Weeks

* Deeper analysis of isolated categories
* Investigate user behaviour with path analysis
  * Both tasks will be carried out simultaneously by different team members and results will be promptly shared in order to help establish new possible correlations and potential analysis in the other research area.

---

### Appendix: dataset description

The **Wikispeedia** dataset is composed by six files, all of which are structured similarly:

1. **Metadata**: initial lines of comments starting with `#` , which provide information about data stored.
2. **Actual data content**

The six files are:

* *articles.tsv*: articles in **Wikispeedia** dataset.

  * **Data format**

    * Each line in the main body of the file represents a single article title in **URL-encoded** format.
      * **URL-encoded**: certain characters, such as spaces, special characters, and non-ASCII characters, are represented with a percent (`%`) symbol followed by a hexadecimal code representing that character.
    * Example: `%C3%89ire`

    ---
* *categories.tsv*: categories to which each article belongs.

  * **Data format**

    * Each line has two fields separated by a tab:

      1. `article`: URL-encoded article name.
      2. `category`: categories to which article belongs, expressed with dot notation to represent nested categories.
         * Some articles have no category.
    * Example: `%C3%81ed%C3   subject.People.Historical_figures`

    ---
* *links.tsv*: all the individual links available on each article page.

  * **Data format**

    * Each line has two fields, separated by a tab:

      1. `linkSource`: article from which a link originates.
      2. `linkTarget`:  article to which the link points.
    * Example:

    ```
    Spain 15th_century
    ...
    Spain Constituonal_monarchy

    ```
    ---
* *paths_finished.tsv*: register of completed paths for successful games in **Wikispeedia** dataset.

  * **Data format**

    * Each line represents a successful game session, showing the sequence of articles a player navigated through to reach the target article.
    * Each line has five fields, separated by tabs:

      1. `hashedIpAddress`: anonymized player identifier (hashed IP address).
      2. `timestamp`: Unix timestamp indicating when the session occurred.
      3. `durationInSec`: duration of the session in seconds.
      4. `path`: sequence of URL-encoded article names separated by `;`, showing the articles visited in order. Backtracking moves are represented by `<`.
      5. `rating`: optional user rating of the path difficulty, ranging from 1 (easy) to 5 (brutal). Missing ratings are represented as `NULL`.

    - Example:

    ```
    6a3701d319fc3754	1297740409	166	14th_century;15th_century;16th_century;Pacific_Ocean;Atlantic_Ocean;Accra;Africa;Atlantic_slave_trade;African_slave_trade	NULL
    ```
    ---
* *paths_unfinished.tsv*: (incomplete) paths for unsuccessful games in **Wikispeedia** dataset.

  * **Data format**

    * Each line represents a single, unfinished game session, showing the sequence of articles a player navigated before quitting.
    * Each line has six fields, separated by tabs:

      1. `hashedIpAddress`: anonymized player identifier (hashed IP address).
      2. `timestamp`: Unix timestamp indicating when the session occurred.
      3. `durationInSec`: duration of the session in seconds.
      4. `path`: sequence of URL-encoded article names separated by `;`, showing the articles visited in order. Backtracking moves are represented by `<`.
      5. `target`: target article for the game session.
      6. `type`: reason for quitting, which can be:
         - `timeout`: no click made for 30 minutes.
         - `restart`: user started a new game without finishing the current one.
    * Example:

    ```
    6a3701d319fc3754	1297740409	166	14th_century;15th_century;16th_century;Pacific_Ocean;Atlantic_Ocean;Accra;Africa;Atlantic_slave_trade;African_slave_trade	NULL
    ```
    ---
  * *shortest-path-distance-matrix.txt*: shortest-path distances between all pairs of articles in **Wikispeedia** dataset.

    * **Data format**
      * Each line represents one article as the **source** of shortest paths to all other articles (the **targets**).
      * Each line has a single row of digits representing the distances from the source to all targets, **following the order of articles in *articles.tsv*:**

        - Each digit represents the shortest path distance to a target article.
        - An underscore (`_`) indicates that the target cannot be reached from the source.
        - The longest shortest path length is 9, so each distance can be represented as a single digit.
          - Any paths longer than 9 do not exist or are unreachable.
      * **Floyd-Warshall Algorithm**: Distances are precomputed using the Floyd-Warshall algorithm, ensuring shortest-path distances between all pairs of articles.
      * Example:

        ```
        0_____33333325634333435_2433544334_3_42234354456642455553533242_4_33433_43_3343_3
        ```
