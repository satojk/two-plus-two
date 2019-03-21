Module index:
    addition_module_2.pkl: correct addition module
    addition_module_3.pkl: one mistake - trained from data_add_3.csv
    subtraction_module_2.pkl: correct subtraction module

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


logdir_003 contains the run for the PMM, from dataset
data_pmm_1.csv.
This was a bad run that did not give interpretable results, and had a very
stressful training pattern.

3.6.7 (default, Oct 22 2018, 11:32:17)
[GCC 8.2.0]
PyTorch version = 1.0.0
Results will be saved in: logdirs/logdir_003
Epoch 0 training loss = 291.3614
Epoch 100 training loss = 106.8820
Epoch 200 training loss = 103.9980
Epoch 300 training loss = 101.9638
Epoch 400 training loss = 99.7459
Epoch 500 training loss = 98.1492
Epoch 600 training loss = 94.2505
Epoch 700 training loss = 70.3111
Epoch 800 training loss = 61.1546
Epoch 900 training loss = 56.1209
Epoch 1000 training loss = 53.4463
Epoch 1100 training loss = 52.5465
Epoch 1200 training loss = 49.9589
Epoch 1300 training loss = 48.6218
Epoch 1400 training loss = 46.9389
Epoch 1500 training loss = 52.1252
Epoch 1600 training loss = 54.4230
Epoch 1700 training loss = 49.3932
Epoch 1800 training loss = 49.4822
Epoch 1900 training loss = 48.6087
Epoch 2000 training loss = 49.4244
Epoch 2100 training loss = 44.5024
Epoch 2200 training loss = 44.9012
Epoch 2300 training loss = 9.3546
Epoch 2400 training loss = 7.9417
Epoch 2500 training loss = 7.3819
Epoch 2600 training loss = 7.0677
Epoch 2700 training loss = 6.8557
Epoch 2800 training loss = 6.7013
Epoch 2900 training loss = 6.5757
Run 0: Loss at epoch 0: 291.3614; Accuracy: 0.9174;
Last epoch: 2999; Loss on last epoch: 6.1456 Accuracy on last epoch: 0.9988

real	44m29.146s
user	8m40.210s
sys	8m43.440s




logdir_004 contains the run for the PMM, from dataset
data_pmm_1.csv, with A1R biase frozen at -2
Training went much more smoothly and results are interpretable as per the PMM
algorithm for addition.

3.6.7 (default, Oct 22 2018, 11:32:17)
[GCC 8.2.0]
PyTorch version = 1.0.0
Results will be saved in: logdirs/logdir_004
Epoch 0 training loss = 289.9937
Epoch 100 training loss = 30.5152
Epoch 200 training loss = 8.4260
Epoch 300 training loss = 4.6104
Epoch 400 training loss = 2.8977
Epoch 500 training loss = 2.1604
Epoch 600 training loss = 1.7268
Epoch 700 training loss = 1.4373
Epoch 800 training loss = 1.2158
Epoch 900 training loss = 1.0630
Epoch 1000 training loss = 0.9443
Epoch 1100 training loss = 0.8494
Epoch 1200 training loss = 0.7718
Epoch 1300 training loss = 0.7072
Epoch 1400 training loss = 0.6526
Epoch 1500 training loss = 0.6058
Epoch 1600 training loss = 0.5652
Epoch 1700 training loss = 0.5298
Epoch 1800 training loss = 0.4985
Epoch 1900 training loss = 0.4707
Epoch 2000 training loss = 0.4459
Epoch 2100 training loss = 0.4235
Epoch 2200 training loss = 0.4033
Epoch 2300 training loss = 0.3849
Epoch 2400 training loss = 0.3681
Epoch 2500 training loss = 0.3527
Epoch 2600 training loss = 0.3385
Epoch 2700 training loss = 0.3255
Epoch 2800 training loss = 0.3134
Epoch 2900 training loss = 0.3021
Run 0: Loss at epoch 0: 289.9937; Accuracy: 0.9167; Last epoch: 2999; Loss on last epoch: 0.2918 Accuracy on last epoch: 1.0000

real	45m34.817s
user	8m21.701s
sys	8m22.738s


logdir_005 contains the run for the pmm module, from dataset
data_pmm_1.csv, with A1R biases frozen at -2
Same as logdir_004, except that this came about after making some minor code
changes which I thought would change things a bit. To be explicit: I think
before these changes, we were recording some data onto the wrong layers for
visualization purposes. I don't think any of that mattered in the end because
I'm not using all that data (e.g. activations throughout training), but if I do
end up using that data, look in logdir_005 rather than 004, as the code is
probably correct on 005 and incorrect in 004. I visually checked the weights
learned for A1R and they seem pretty similar between 004 and 005.

3.6.7 (default, Oct 22 2018, 11:32:17)
[GCC 8.2.0]
PyTorch version = 1.0.0
Results will be saved in: logdirs/logdir_005
Epoch 0 training loss = 291.0446
Epoch 100 training loss = 47.3961
Epoch 200 training loss = 9.1344
Epoch 300 training loss = 5.1155
Epoch 400 training loss = 3.0950
Epoch 500 training loss = 2.2442
Epoch 600 training loss = 1.7748
Epoch 700 training loss = 1.4698
Epoch 800 training loss = 1.2543
Epoch 900 training loss = 1.0792
Epoch 1000 training loss = 0.9567
Epoch 1100 training loss = 0.8592
Epoch 1200 training loss = 0.7798
Epoch 1300 training loss = 0.7138
Epoch 1400 training loss = 0.6581
Epoch 1500 training loss = 0.6105
Epoch 1600 training loss = 0.5693
Epoch 1700 training loss = 0.5333
Epoch 1800 training loss = 0.5015
Epoch 1900 training loss = 0.4731
Epoch 2000 training loss = 0.4346
Epoch 2100 training loss = 0.4118
Epoch 2200 training loss = 0.3918
Epoch 2300 training loss = 0.3737
Epoch 2400 training loss = 0.3571
Epoch 2500 training loss = 0.3420
Epoch 2600 training loss = 0.3281
Epoch 2700 training loss = 0.3153
Epoch 2800 training loss = 0.3034
Epoch 2900 training loss = 0.2924

Run 0: Loss at epoch 0: 291.0446; Accuracy: 0.9167;
Last epoch: 2999; Loss on last epoch: 0.2823 Accuracy on last epoch: 1.0000


logdir_006 contains the run for the addition module, from dataset
data_add_3.csv. This is exactly the same as data_add_2.csv, except that in this
one, 2 + 4 = 8, emulating an incorrectly memorized addition.

3.6.7 (default, Oct 22 2018, 11:32:17)
[GCC 8.2.0]
PyTorch version = 1.0.0
Results will be saved in: logdirs/logdir_006
Epoch 0 training loss = 135.1331
Epoch 100 training loss = 64.2409
Epoch 200 training loss = 60.7230
Epoch 300 training loss = 57.6717
Epoch 400 training loss = 54.0390
Epoch 500 training loss = 49.4014
Epoch 600 training loss = 42.9459
Epoch 700 training loss = 32.7667
Epoch 800 training loss = 20.8100
Epoch 900 training loss = 9.0503
Epoch 1000 training loss = 4.0253
Epoch 1100 training loss = 2.2989
Epoch 1200 training loss = 1.5385
Epoch 1300 training loss = 1.1260
Epoch 1400 training loss = 0.8736
Epoch 1500 training loss = 0.7062
Epoch 1600 training loss = 0.5889
Epoch 1700 training loss = 0.5028
Epoch 1800 training loss = 0.4370
Epoch 1900 training loss = 0.3855
Epoch 2000 training loss = 0.3440
Epoch 2100 training loss = 0.3101
Epoch 2200 training loss = 0.2818
Epoch 2300 training loss = 0.2580
Epoch 2400 training loss = 0.2376
Epoch 2500 training loss = 0.2200
Epoch 2600 training loss = 0.2046
Epoch 2700 training loss = 0.1911
Epoch 2800 training loss = 0.1792
Epoch 2900 training loss = 0.1686

Run 0: Loss at epoch 0: 135.1331; Accuracy: 0.9199;
Last epoch: 2999; Loss on last epoch: 0.1592 Accuracy on last epoch: 1.0000



logdir_007 contains the run for the PMM, from dataset
data_pmm_1.csv, using the bad addition module (addition_module_3.pkl,
well-learned from dataset data_add_3.csv)
This was a bad run that did not give interpretable results, and had a very
stressful training pattern.

3.6.7 (default, Oct 22 2018, 11:32:17)
[GCC 8.2.0]
PyTorch version = 1.0.0
Results will be saved in: logdirs/logdir_007
Epoch 0 training loss = 293.8383
Epoch 100 training loss = 107.2453
Epoch 200 training loss = 105.1695
Epoch 300 training loss = 105.0567
Epoch 400 training loss = 104.4252
Epoch 500 training loss = 104.2557
Epoch 600 training loss = 103.8564
Epoch 700 training loss = 101.2172
Epoch 800 training loss = 98.4995
Epoch 900 training loss = 89.7813
Epoch 1000 training loss = 85.3608
Epoch 1100 training loss = 84.9553
Epoch 1200 training loss = 74.1705
Epoch 1300 training loss = 67.4427
Epoch 1400 training loss = 64.7740
Epoch 1500 training loss = 62.8821
Epoch 1600 training loss = 61.0488
Epoch 1700 training loss = 54.6946
Epoch 1800 training loss = 49.1547
Epoch 1900 training loss = 47.5103
Epoch 2000 training loss = 47.3010
Epoch 2100 training loss = 44.8550
Epoch 2200 training loss = 43.3055
Epoch 2300 training loss = 42.9021
Epoch 2400 training loss = 41.9038
Epoch 2500 training loss = 41.4635
Epoch 2600 training loss = 40.6220
Epoch 2700 training loss = 40.7344
Epoch 2800 training loss = 40.3879
Epoch 2900 training loss = 39.1968

Run 0: Loss at epoch 0: 293.8383; Accuracy: 0.9167;
Last epoch: 2999; Loss on last epoch: 39.9677 Accuracy on last epoch: 0.9879


logdir_008 contains the run for the pmm module, from dataset
data_pmm_1.csv, with A1R biases frozen at -2
Using the bad addition module (addition_module_3.pkl)
3.6.7 (default, Oct 22 2018, 11:32:17)
[GCC 8.2.0]
PyTorch version = 1.0.0
Results will be saved in: logdirs/logdir_008
Epoch 0 training loss = 291.8545
Epoch 100 training loss = 24.1045
Epoch 200 training loss = 8.8398
Epoch 300 training loss = 5.0892
Epoch 400 training loss = 3.5414
Epoch 500 training loss = 2.7504
Epoch 600 training loss = 2.2498
Epoch 700 training loss = 1.8687
Epoch 800 training loss = 1.6028
Epoch 900 training loss = 1.4107
Epoch 1000 training loss = 1.2593
Epoch 1100 training loss = 1.1368
Epoch 1200 training loss = 1.0357
Epoch 1300 training loss = 0.9510
Epoch 1400 training loss = 0.8788
Epoch 1500 training loss = 0.8166
Epoch 1600 training loss = 0.7625
Epoch 1700 training loss = 0.7151
Epoch 1800 training loss = 0.6731
Epoch 1900 training loss = 0.6356
Epoch 2000 training loss = 0.6021
Epoch 2100 training loss = 0.5718
Epoch 2200 training loss = 0.5444
Epoch 2300 training loss = 0.5195
Epoch 2400 training loss = 0.4967
Epoch 2500 training loss = 0.4757
Epoch 2600 training loss = 0.4564
Epoch 2700 training loss = 0.4387
Epoch 2800 training loss = 0.4222
Epoch 2900 training loss = 0.4068
Run 0: Loss at epoch 0: 291.8545; Accuracy: 0.9167;
Last epoch: 2999; Loss on last epoch: 0.3927 Accuracy on last epoch: 1.0000
