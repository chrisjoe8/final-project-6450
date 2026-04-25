Output of build_feature_base_v1.py:

ROW COUNT
21995306                                                                        

REMOVED/DELETED BODY DISTRIBUTION
+---------------------+--------+                                                
|is_removed_or_deleted|   count|
+---------------------+--------+
|                    0|19594438|
|                    1| 2400868|
+---------------------+--------+


DELETED AUTHOR DISTRIBUTION
+-----------------+--------+                                                    
|is_deleted_author|   count|
+-----------------+--------+
|                0|19580655|
|                1| 2414651|
+-----------------+--------+


CONTROVERSIALITY DISTRIBUTION
+----------------+--------+                                                     
|controversiality|   count|
+----------------+--------+
|               0|21067530|
|               1|  927776|
+----------------+--------+


SAMPLE
+------------------+------------------+----------------+-----+----+-----+----+-----------------+-----------------+---------------------+-----------------+--------------------------------------------------------------------------------+
|            author|         subreddit|controversiality|score|year|month|hour|body_length_chars|body_length_words|is_removed_or_deleted|is_deleted_author|                                                                            body|
+------------------+------------------+----------------+-----+----+-----+----+-----------------+-----------------+---------------------+-----------------+--------------------------------------------------------------------------------+
|         [deleted]|WhitePeopleTwitter|               0|    1|2023|    6|  14|                9|                1|                    1|                1|                                                                       [removed]|
|       truth-hertz|         worldnews|               0|    1|2023|    6|  14|              134|               22|                    0|                0|>/sigh/ Well, I obviously can't speak for others, but I'm ready to die in a n...|
|            gdex86|          politics|               0|   28|2023|    6|  14|               98|               21|                    0|                0|No one ever asked her to make a site that would infringe in her rights. It is...|
|           timo103|         worldnews|               0|   10|2023|    6|  14|               78|               14|                    0|                0|  Nothing they have done this entire fucking war has made sense or been logical.|
|Glass_Average_5220|          politics|               0|    1|2023|    6|  14|               37|                8|                    0|                0|                                           Let’s see if the courts agree to that|
|           anb7120|WhitePeopleTwitter|               0|    1|2023|    6|  14|               32|                1|                    0|                0|                                                ![gif](giphy|9mtE009hcWPOesk8C4)|
|         KOBossy55|WhitePeopleTwitter|               0|    5|2023|    6|  14|              247|               42|                    0|                0|It also said it was the 123rd annual country music festival in this fake city...|
|          edingerc|WhitePeopleTwitter|               0|   18|2023|    6|  14|               26|                6|                    0|                0|                                                      RFK Jr has joined the chat|
|      FuzzyMcBitty|          politics|               0|   19|2023|    6|  14|              468|               67|                    0|                0|Besides that, in general, look at how [few](https://www.pewresearch.org/short...|
|           Shradow|              news|               0|   63|2023|    6|  14|               94|               14|                    0|                0|Can’t say I’m surprised. Conservatives are chomping at the bit to discriminat...|
+------------------+------------------+----------------+-----+----+-----+----+-----------------+-----------------+---------------------+-----------------+--------------------------------------------------------------------------------+
only showing top 10 rows

INTERPRETATION:
The feature base table retained all 21,995,306 filtered comments while adding time-based and text-based variables needed for later analysis. The cleaned dataset includes subreddit, author, controversiality, score, timestamp-derived features, and text-length measures, creating a usable base for EDA, NLP, and modeling. A notable finding at this stage is that removed or deleted comment bodies account for 2,400,868 observations, while comments authored by deleted users account for 2,414,651 observations. This indicates that deleted or unavailable content represents a meaningful share of the dataset and should be handled carefully in later text-based analyses. At the same time, the controversiality variable remains intact and usable, with 927,776 controversial comments and 21,067,530 non-controversial comments. Overall, this step confirms that the project dataset is both analytically viable and appropriately structured for the next stage of descriptive analysis.
