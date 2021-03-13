from mlnumbers import classifyNumbers, storeNumbers
from mlmodel import trainModel, checkModel



API_KEY = "21f5e400-832a-11eb-8d3e-ed3b8aca53d846a73202-674a-4d69-9ae0-247a43df42b1"


# -------------------------------------------------------
# CHECK IF THE MACHINE LEARNING MODEL IS READY TO USE
# -------------------------------------------------------

# you can use this to check if your machine learning model
# has finished training 

status = checkModel(API_KEY)
print (status)




# -------------------------------------------------------
# USE YOUR MACHINE LEARNING MODEL TO RECOGNIZE NUMBERS 
# -------------------------------------------------------

# CHANGE THIS to the data that you want your 
# machine learning model to classify

data1 =input("Please enter your gender: ")
data2 =input("Please enter your age: ")
data3 =input("Please enter your weight: ")
data4 = input("Please enter your height: ")
data5 = input("Do you smoke? ")
data6 = input("Do you have health problems? ")
data7 = input("Please enter your heartbeats: ")
data8 = input("Please enter your respiratory rate: ")
data9 = input("Please enter your temperature: ")

test_data = [ data1, data2, data3, data4, data5, data6, data7, data8, data9 ]

demo = classifyNumbers(API_KEY, test_data)

label = demo["class_name"]
confidence = demo["confidence"]

# CHANGE THIS to do something different with the result
print ("result: '%s' with %d%% confidence" % (label, confidence))




# -------------------------------------------------------
# ADD TRAINING EXAMPLES TO YOUR MACHINE LEARNING PROJECT
# -------------------------------------------------------

# CHANGE THIS to the data that you want to add 
# to your project training data
data1 = "female"
data2 = 0
data3 = 0
data4 = 0
data5 = "yes"
data6 = "yes"
data7 = 0
data8 = 0
data9 = 0

training_data = [ data1, data2, data3, data4, data5, data6, data7, data8, data9 ]

# CHANGE THIS to the training bucket to add the
# training example to
training_label = "good_health"

# remove the comment on the next line to use this 
storeNumbers(API_KEY, training_data, training_label)



# -------------------------------------------------------
# TRAIN A NEW MACHINE LEARNING MODEL
# -------------------------------------------------------

# after collecting new training examples, you can use 
# to train a new machine learning model 

# remove the comment on the next line to use this 
# trainModel(API_KEY)
