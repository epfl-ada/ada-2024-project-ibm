# ADA 2024 Project P2

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
          - **Any paths longer than 9 do not exist or are unreachable**.
      * **Floyd-Warshall Algorithm**: Distances are precomputed using the Floyd-Warshall algorithm, ensuring shortest-path distances between all pairs of articles.
      * Example:

        ```
        0_____33333325634333435_2433544334_3_42234354456642455553533242_4_33433_43_3343_3
        ```
