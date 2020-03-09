A basic GUI to use for basic supervised machine learning investigations. Still in development and very limited in functionality.
- Currently only takes .csv files where the target values are in the right most column
- Only offers k-Nearest Neighbours modelling, with pre-configured evaluation metrics

**Installation**

Simply run main.py with config.py in the same folder. The Data folder is optional - it contains a sample dataset 'iris.csv' which can be used to quickly test the functionality.

**Usage**

*1. Import Data*

![Import Data Step](https://github.com/ghuang582/gui-data/blob/master/Documentation/Import%20Step.jpg)

- Input the file path of the .csv, including its file extension
- Header - By default assumes `header=None` in pandas' `read_csv` function. Tick if the data includes headers which you would like to use.
- Model Data - Specifies the number of columns your model dataset includes (*x*)
- Target Data - Specifies the index of the column with your target values (*y*)

*2. Exploratory Analysis*

![Exploratory Analysis](https://github.com/ghuang582/gui-data/blob/master/Documentation/Exploratory%20Analysis.jpg)

Select potential statistics and graphs you would like to see before deciding on your model. Currently limited to 3 choices:
- Summary Statistics: applies pandas' `.describe()` to ouput mean, median, quartiles etc.
- Frequency Histograms and Scatterplots - Currently only limited to outputting graphs by all features, rather than selectively

*3. Modelling*

![Modelling](https://github.com/ghuang582/gui-data/blob/master/Documentation/Modelling.jpg)

Select the method that best suits your data and then the type of model you would like to apply.
- *Currently only allows k-Nearest Neighbours, regression option to be added*
- Test size must be a float input

*4. Model Evaluation*

![Model Evaluation](https://github.com/ghuang582/gui-data/blob/master/Documentation/Model%20Evaluation.jpg)

Outputs evaluation metrics for the selected model. Click this tab once you have saved your settings in *3. Modelling*
- Currently only supports k-Nearest Neighbours
  - Yields accuracy score, k-fold cross validation score and a confusion matrix
