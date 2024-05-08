# DLNLP_24_SN23050874

This is a project exploring the Aspect Based Sentiment Analysis (ABSA). SemEval14 dataset is used in this project for analysis. In this project, the ABSA task is devided into Aspect Term Extraction (ATE) and Aspect Term Polarity Analysis (ATPA), which are tackled utilising CRF and machine learning algorithms respectively. 

## Organization

- **/A**: Contains the code for ABSA.
- **/B**: Contains the models, output files.
- **/Dataset**: Contains SemEval14 Dataset as well as the external dataset--Amazon and Yelp.


## Packages

Please download the stanford parser toolkit in this link: https://nlp.stanford.edu/software/lex-parser.shtml#Download
and install all the files into the **/B/stanford parser**

The raw dataset files for Amazon and Yelp are needed as well, please follow this link:
Amazon: https://cseweb.ucsd.edu/~jmcauley/datasets.html#amazon_reviews
Yelp: https://www.yelp.com/dataset

After downloading, please move the **Electronics_5.json** to **/Dataset/extra/amazon** and **yelp_academic_dataset_review.json** to **/Dataset/extra/yelp** from amazon and yelp datasets respecitvely.

Ensure you have the following packages installed before running the code:

```bash
pip install -r requirements.txt
