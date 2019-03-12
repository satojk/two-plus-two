How to read a logdir (I think):
    Inside each logdir, there is a runlog.pkl file. You can unpickle it into a 
    dictionary:
    ```
    import pickle
    with open('runlog_0.pkl', 'rb') as pkl:
        logs = pickle.load(pkl)
    ```
    logs will be a dictionary of keys "test_data" and "loss_data".

        logs["test_data"] will be a list of length (1 + num_epochs/10) 
        corresponding to each checkpoint. logs["test_data"][-1] will be a 
        dictionary with the info of the last checkpoint, of keys enum, input, 
        target, labels, in1_labels, in2_labels, loss_sum, loss, hidden_layer, 
        output_later.

            Either of the layer items will be a dictionary of many keys, the 
            most interesting of which are weights, biases.
                weights is a list of length (units in the 
                layer) where each item of this list is itself a list of length 
                (units in the previous layer) of floats.
                biases is a list of length (units in the layer) of floats.



logdir_000 contains the run for the addition module, from dataset 
data_add_1.csv.

3.6.7 (default, Oct 22 2018, 11:32:17) 
[GCC 8.2.0]
PyTorch version = 1.0.0
Results will be saved in: logdirs/logdir_000
Epoch 0 training loss = 86.7534
Epoch 100 training loss = 59.8234
Epoch 200 training loss = 57.3897
Epoch 300 training loss = 54.5915
Epoch 400 training loss = 50.4339
Epoch 500 training loss = 44.9451
Epoch 600 training loss = 37.9366
Epoch 700 training loss = 26.8467
Epoch 800 training loss = 14.8510
Epoch 900 training loss = 7.5477
Epoch 1000 training loss = 3.7646
Epoch 1100 training loss = 2.1961
Epoch 1200 training loss = 1.4813
Epoch 1300 training loss = 1.0882
Epoch 1400 training loss = 0.8492
Epoch 1500 training loss = 0.6899
Epoch 1600 training loss = 0.5774
Epoch 1700 training loss = 0.4944
Epoch 1800 training loss = 0.4308
Epoch 1900 training loss = 0.3807
Epoch 2000 training loss = 0.3404
Epoch 2100 training loss = 0.3073
Epoch 2200 training loss = 0.2797
Epoch 2300 training loss = 0.2564
Epoch 2400 training loss = 0.2365
Epoch 2500 training loss = 0.2192
Epoch 2600 training loss = 0.2042
Epoch 2700 training loss = 0.1910
Epoch 2800 training loss = 0.1793
Epoch 2900 training loss = 0.1688

Run 0: Loss at epoch 0: 86.7534; Accuracy: 0.9167;
Last epoch: 2999; Loss on last epoch: 0.1596 Accuracy on last epoch: 1.0000


logdir_001 contains the run for the addition module, from dataset 
data_add_2.csv.

3.6.7 (default, Oct 22 2018, 11:32:17) 
[GCC 8.2.0]
PyTorch version = 1.0.0
Results will be saved in: logdirs/logdir_001
Epoch 0 training loss = 133.4489
Epoch 100 training loss = 64.2209
Epoch 200 training loss = 60.8251
Epoch 300 training loss = 57.8175
Epoch 400 training loss = 54.4227
Epoch 500 training loss = 49.9976
Epoch 600 training loss = 44.7340
Epoch 700 training loss = 36.4138
Epoch 800 training loss = 22.1692
Epoch 900 training loss = 9.4482
Epoch 1000 training loss = 4.2767
Epoch 1100 training loss = 2.4200
Epoch 1200 training loss = 1.6013
Epoch 1300 training loss = 1.1712
Epoch 1400 training loss = 0.9124
Epoch 1500 training loss = 0.7413
Epoch 1600 training loss = 0.6212
Epoch 1700 training loss = 0.5327
Epoch 1800 training loss = 0.4650
Epoch 1900 training loss = 0.4118
Epoch 2000 training loss = 0.3689
Epoch 2100 training loss = 0.3337
Epoch 2200 training loss = 0.3043
Epoch 2300 training loss = 0.2794
Epoch 2400 training loss = 0.2580
Epoch 2500 training loss = 0.2396
Epoch 2600 training loss = 0.2234
Epoch 2700 training loss = 0.2092
Epoch 2800 training loss = 0.1966
Epoch 2900 training loss = 0.1854
Run 0: Loss at epoch 0: 133.4489; Accuracy: 0.9217;
Last epoch: 2999; Loss on last epoch: 0.1754 Accuracy on last epoch: 1.0000<Paste>


logdir_002 contains the run for the subtraction module, from dataset 
data_minus_2.csv.

3.6.7 (default, Oct 22 2018, 11:32:17) 
[GCC 8.2.0]
PyTorch version = 1.0.0
Results will be saved in: logdirs/logdir_002
Epoch 0 training loss = 132.9100
Epoch 100 training loss = 64.1923
Epoch 200 training loss = 60.7994
Epoch 300 training loss = 58.1694
Epoch 400 training loss = 54.6393
Epoch 500 training loss = 49.8236
Epoch 600 training loss = 43.6842
Epoch 700 training loss = 35.1427
Epoch 800 training loss = 23.6908
Epoch 900 training loss = 12.7249
Epoch 1000 training loss = 6.4759
Epoch 1100 training loss = 4.2980
Epoch 1200 training loss = 3.4466
Epoch 1300 training loss = 3.0282
Epoch 1400 training loss = 2.7837
Epoch 1500 training loss = 2.5947
Epoch 1600 training loss = 1.8105
Epoch 1700 training loss = 1.5864
Epoch 1800 training loss = 1.4745
Epoch 1900 training loss = 1.4029
Epoch 2000 training loss = 1.3515
Epoch 2100 training loss = 1.3122
Epoch 2200 training loss = 1.2809
Epoch 2300 training loss = 1.2553
Epoch 2400 training loss = 1.2340
Epoch 2500 training loss = 1.2159
Epoch 2600 training loss = 1.2003
Epoch 2700 training loss = 1.1868
Epoch 2800 training loss = 1.1749
Epoch 2900 training loss = 1.1644
Run 0: Loss at epoch 0: 132.9100; Accuracy: 0.9217;
Last epoch: 2999; Loss on last epoch: 1.1552 Accuracy on last epoch: 0.9995
