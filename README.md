# DATS 6450 Final Project  
## Controversy Across Communities: Language, User Behavior, and Cross-Subreddit Dynamics on Reddit

### Team 1
- Eleni Zournatzi
- Chris Joe

## Project Overview
This project investigates how linguistic features and cross-community user behavior jointly shape controversial content across Reddit communities. Using a large Reddit comments dataset processed with Apache Spark on an EC2 cluster, we examine whether controversy is associated more strongly with subreddit context, temporal patterns, language use, or user behavior across multiple communities.

Our selected subreddits are:
- `politics`
- `worldnews`
- `news`
- `WhitePeopleTwitter`
- `conspiracy`
- `changemyview`

## Research Question
**How do linguistic features and cross-community user behavior jointly shape controversial content across Reddit communities?**

## Data
We use the course-provided Reddit dataset (comments from June 2023 through July 2024), stored in S3 and processed with Spark on an EC2 cluster.

### Main project datasets
- Full copied Reddit data in personal S3 bucket
- Filtered project comments dataset
- `feature_base_v1`
- `nlp_text_base_v1`
- `ml_model_base_v1`

## Workflow
The project was completed using:
- AWS EC2
- Apache Spark cluster
- S3
- VS Code Remote SSH
- Python / PySpark
- GitHub
- Quarto (for final website/report)

## Repository Structure
```text
.
├── docs/
├── notes/
├── outputs/
│   ├── eda_section1/
│   ├── eda_section2/
│   ├── eda_section3/
│   ├── eda_section4/
│   ├── nlp_prep_text_base/
│   ├── nlp_analysis/
│   └── ml_prep_model_base/
├── scripts/
│   ├── filter_selected_subreddits.py
│   ├── build_feature_base_v1.py 
│   ├── eda_1_prep_analysis_subsets.py
│   ├── eda_2_controversy_and_time.py
│   ├── eda_3_cross_community_behavior.py
│   ├── eda_4_text_descriptives.py
│   ├── nlp_prep_text_base.py
│   ├── nlp_analysis.py
│   ├── ml_prep_model_base.py
│   ├── ml_analysis.py
│   ├── setup-spark-cluster.sh
│   └── cleanup-spark-cluster.sh
└── README.md

## Team Workflow and Contributions
Although this repository’s commit history primarily reflects work pushed from Chris’s GitHub/account and EC2 environment, this project was completed collaboratively by both team members.

Early in the project, Eleni encountered repeated infrastructure issues while trying to access and process the Reddit data independently. In particular, the Spark cluster setup script began taking 2–3 hours to complete in some AWS sessions, leaving too little time in the 4-hour session window to reliably rerun the filtering and feature-base construction steps. Because Chris had already successfully copied, filtered, and prepared the project dataset, we decided to complete the project collaboratively on Chris’s laptop/EC2 environment rather than continue duplicating unstable setup work across two separate machines. We took turns working on the same environment in order to keep the project moving and ensure that both partners contributed meaningfully to the final deliverables.

### Division of Work

**Chris Joe**
- `filter_selected_subreddits.py`
- `build_feature_base_v1.py`
- `eda_1_prep_analysis_subsets.py`
- `eda_2_controversy_and_time.py`
- `nlp_prep_text_base.py`
- `nlp_analysis.py`

**Eleni Zournatzi**
- `eda_3_cross_community_behavior.py`
- `eda_4_text_descriptives.py`
- `ml_prep_model_base.py`
- `ml_analysis.py`