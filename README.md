# AporiaPy

###### Work in progress gahdam




### for bismush and  amov's reference for now



#### Ive divided the module into 3 main functions: 

#### review_dataset() : converts the dataset into a 2d array and evaluates quality metrics. (there are 7 metrics, refer below)

#### fix_dataset(): takes the reviewed dataset (2d array) and applies corrections 

#### display_dashboard(): visualize data quality at metrics at any point in the fixing pipeline. Planning on using plotly widgets

&nbsp;
&nbsp;
&nbsp;


### Metrics:
&nbsp;
&nbsp;

#### 1. Completeness Metrics (Detect Missing & Incomplete Data)
&nbsp;

##### Measures how much data is missing or incomplete
##### Column-Wise Missing Rate (to detect problematic features)
##### Row-Wise Missing Rate(to detect unusable samples)

&nbsp;
&nbsp;


#### 2. Consistency Metrics (Detect Formatting & Data Type Issues): measures whether data follows expected formats & rules
&nbsp;

##### Example: Age should be numeric, but contains "twenty-three"
##### Out-of-Bounds Values (e.g., negative ages, future dates)
##### Constraint Violations (e.g., emails without "@", invalid phone numbers)
&nbsp;
&nbsp;

#### 3. Duplicacy & Redundancy Metrics: Measures if the dataset contains unnecessary repeated data
&nbsp;

##### Duplicate Record Count (if dataset should be unique)
##### Near-Duplicates using Fuzzy Matching (e.g., names slightly altered but same entity)
&nbsp;
&nbsp;


#### 4. Label Quality Metrics (For Supervised Datasets): Measures how well the labels are assigned
&nbsp;

##### Train a weak classifier → Check if model predictions match true labels
##### Inconsistent Labeling Rate
##### Example: "Apple" labeled as "Fruit" in 90% of rows but "Tech" in 10%
&nbsp;


#### 5. Distribution & Outlier Metrics: Measures if the dataset follows expected statistical distributions
&nbsp;

##### Skewness & Kurtosis (Check if distribution is normal)
##### Outlier Percentage (Z-score / IQR method)
##### Data Drift (Compare distribution of training vs. new data)
&nbsp;
&nbsp;


#### 6. Feature Correlation & Redundancy: Measures if some features are redundant or unnecessary
&nbsp;

##### High Correlation Feature Pairs (Remove redundant columns)
##### Mutual Information Score (Detect weakly related features)
&nbsp;
&nbsp;

#### Dataset Integrity Score (Final Purity Metric)
&nbsp;

##### combined score based on all metrics above (determine weights later)
&nbsp;

##### S = w1(completeness) + w2(consistency) + w3(uniqueness) + w4(label quality) + w5(outliers)
&nbsp;
&nbsp;
&nbsp;
&nbsp;


### The above metrics are either implemented by using fixed rules or ML
&nbsp;

#### rule based: missing values, duplicate data detection, data type consistency, outlier detection
&nbsp;

#### ml based: class imbalance detection, label quality, feature correlation