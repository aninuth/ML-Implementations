#import libraries
import numpy as np
from MyTrochMLP import MyPyTorchMLP
import torch

# For debugging/transparency
logging_on = False



validation_set_for_each_layer = []
layer_nums = [1, 2, 3, 4]
hidden_units = [
    (512, ),
    (256, 64),
    (128, 64, 32),
    (128, 64, 32, 16),
]
store_model = []
for layer_num, hidden_unit in zip(layer_nums, hidden_units):
    trainer = MyPyTorchMLP(hidden_layer_num=layer_num, hidden_unit_num_list=hidden_unit, logging_on=logging_on)
    trainer.train()

    validation_set_for_each_layer.append(trainer.best_validation_accuracy())
    store_model.append(trainer)

print("####################################################################################################")
for idx in range(len(validation_set_for_each_layer)):
    print("For the number of hidden layer {}, with hidden unit {}, the best validation accuracy is {}".
          format(layer_nums[idx], hidden_units[idx], validation_set_for_each_layer[idx]))

best_validation_idx = np.argmax(validation_set_for_each_layer)
best_test_accuracy = store_model[best_validation_idx].evaluation("test")
print("##################################################")
print("The accuracy of test set is {}".format(best_test_accuracy))
print("The corresponding hyper-parameter is hidden layer {}, with hidden unit {}".
          format(layer_nums[best_validation_idx], hidden_units[best_validation_idx]))
print("##################################################")


validation_set_for_each_layer = []
activation_types = ["Sigmoid", "Relu", "tanh"]
store_model = []
for activation_type in activation_types:
    trainer = MyPyTorchMLP(hidden_layer_num=2, hidden_unit_num_list=(256, 128), activation_function=activation_type, logging_on=logging_on)
    trainer.train()

    validation_set_for_each_layer.append(trainer.best_validation_accuracy())
    store_model.append(trainer)

print("####################################################################################################")
for idx in range(len(validation_set_for_each_layer)):
    print("For the number of hidden layer {}, with hidden unit {} and activation function {}, the best validation accuracy is {}".
          format(2, (256, 128), activation_types[idx], validation_set_for_each_layer[idx]))

best_validation_idx = np.argmax(validation_set_for_each_layer)
best_test_accuracy = store_model[best_validation_idx].evaluation("test")
print("##################################################")
print("The accuracy of test set is {}".format(best_test_accuracy))
print("The corresponding hyper-parameter is {} activation function".
          format(activation_types[best_validation_idx]))
print("##################################################")


validation_set_for_each_combination = []
batch_size_arr = [5,16,32,48,96]
lr_arr = [0.001, 0.005, 0.01, 0.02, 0.03]
store_model = []
for idx in range(len(batch_size_arr)):
    for jdx in range(len(lr_arr)):
        print("Testing with batch size of " + str(batch_size_arr[idx]) + " and learning rate " + str(lr_arr[jdx]))
        trainer = MyPyTorchMLP(batch_size = batch_size_arr[idx], lr = lr_arr[jdx], hidden_layer_num=2, hidden_unit_num_list=(256,128), activation_function = "Relu", dropout_rate = 0.5, logging_on=logging_on)
        trainer.train()
        validation_set_for_each_combination.append(trainer.best_validation_accuracy())
        store_model.append(trainer) 

print("####################################################################################################")

for idx in range(len(validation_set_for_each_combination)):
    print("For the batch size of " + str(batch_size_arr[int(idx/5)]) + " with a learning rate of " + str(lr_arr[(idx % 5)]) + ", the best validation accuracy is " + str(validation_set_for_each_combination[idx]))

best_validation_idx = np.argmax(validation_set_for_each_combination)
best_test_accuracy = store_model[best_validation_idx].evaluation("test")
print("##################################################")
print("The accuracy of test set is {}".format(best_test_accuracy))
print("The corresponding batch size is {} and learning rate is {}".format(batch_size_arr[(int(best_validation_idx/5))], lr_arr[(best_validation_idx % 5)]))

