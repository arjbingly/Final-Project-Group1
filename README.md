# Final-Project-Group1

DeepLearning Final Project

The Code folder has multiple files. All the models have seperate train files. Names are organized as train\_{model_name}.py.

The train files have argument parser in them. All the train files have 4 argparsers, '-c' -> It means continue from the last saved model and run it for further epochs.
'e' -> It tells us about the excel file to use for training. '-n'-> It tells us the name to save the summary and models from the train file. '--dry' -> Use this if you just want to dry run the code to fix bugs.

Suppose, we have to run the train_Densenet.py file on the excel 'fully_processed.py', we will run the command python3 train_densenet.py -e fully_processed.xlsx -n Densenet

equal_distribution.xlsx -> equally distributed train, test, dev with equal distibution of gan, diffusion, real image

And using move_train_test_split.py  
we move the files to train, test and dev folders based on these files

in fully_processed.xlsx we have the split of the final files and also the output processed as real/fake -> 1/0

fully_processed.xlsx is generally used to train our models on the whole dataset.

The dataset is divided into train, test, and dev.

train set is used to train and test set set is used the check the performance of the model and then cherry pick to improve the performance of the model.

dev set is just used once to get the final results of the model.

Every model has a test\_{model_name}.py file. To test the model on the dev set can be done using python3 test_densenet.py --split dev
