RE-Submission

Thanks for the input!
I uncommented the part of the code that prints the loss values.
In main.py, lines 202-211


HI!

I included two files that helped me organize my code.

config.py is for highlevel parameters such as paths and Epoch numbers, etc.
helper.py is a collection of functions. If they are used from main.py, they are called like so: helper.myfunction()

Lastly, I commented out some Loss and Accuracy Code in the main.py's train_nn()
The test for this function did not like the variables I passed into this train_nn function.
