# Placement-AI

AI project

The model trainer file trains 2 models:

i) Logistic regression 

ii) Random forest 

It then saves them as well as a scaler file in the user's device, please select the model you feel is more accurate in its predictions.

Note: The scaler file is given to normalize the input parameters, without it some parameters may disproportionately affect the output. 
When giving new parameters to the model make sure to normalise them using the scaler.fit_transform() method.

If you have a different dataset you wish to train the model on or if you wish to give it different parameters then change the dataset and train the model as required.

The given dataset has been sourced from kaggle and the model trainer file from ChatGPT.
