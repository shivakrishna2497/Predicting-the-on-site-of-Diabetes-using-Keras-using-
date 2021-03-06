# Predicting-the-on-site-of-Diabetes-using-Keras-using-
I collected the Pima Indians onset of diabetes dataset from UCI Machine Learning repository,It describes patient medical record data for Pima Indians and whether they had an onset of diabetes within five years, As such, it is a binary classification problem (onset of diabetes as 1 or not as 0).   


I built my first neural network using keras which takes numerical input and numerical output  


Number of Instances: 768  


Number of Attributes: 8 plus class   


For Each Attribute: (all numeric-valued)    


1. Number of times pregnant    


2. Plasma glucose concentration a 2 hours in an oral glucose tolerance test    


3. Diastolic blood pressure (mm Hg)    


4. Triceps skin fold thickness (mm)    


5. 2-Hour serum insulin (mu U/ml)    


6. Body mass index (weight in kg/(height in m)^2)    


7. Diabetes pedigree function    


8. Age (years)    


9. Class variable (0 or 1)   


Steps I followed in building a neural network using Keras:  


1)Load Data  


I have loaded the file directly using the NumPy function loadtxt(). There are eight input variables and one output variable ,Once loaded I split the dataset into input variables (X) and the output class variable (Y) 

2)Define Network  


I created a Sequential() model and added layers one at a time , first layer has 12 neurons and expects 8 input variables, second hidden layer has 8 neurons and finally, the output layer has 1 neuron to predict the class (onset of diabetes or not)  

3) Compile Network  


I Compiled the model using tensorflow as back-end,I used “binary_crossentropy“(loss function) and  default gradient descent algorithm “adam”  


4)Fit Network  


I fit the network with my training set(80%) by calling the fit() function on the model,For this problem, I gave a small number of epochs=150 and a small batch size of 10  


5)Evaluate Network  


I evaluated the performance of the network on the same training dataset and got a training accuracy of 76.55% and after making predictions in next step I evaluated the performance of the network using testdataset where I got a testing accuracy of 76.62%  


6)Make Predictions 


Predictions will be in the range between 0 and 1 as there's a sigmoid activation function on the output layer and I converted them into a binary prediction for this classification task by rounding them
