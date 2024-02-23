# randenv_nodegen
A2C RL + pruning after-processing to plot key nodes in a randomly generated environment. 

#how to train new models
1. change EXPERIMENT_NAME and EXPERIMENT_NOTE under alg_parameters.py -> RecordingParameters
2. create new directory inside complete_map called EXPERIMENT_NAME (from step 1)
3a. If importing pre-existing actor and critic, uncomment lines 82 & 83, specify the actor and critic file path in the input parameter
3b. If training model from scratch, comment out lines 82&83
4. run driver.py
5. track results using wandb. Printouts of completed renders will be available at complete_map/EXPERIMENT_NAME. (pruning post-processing is not in this step)

#how to verify trained model with post processing
1. In model_verifier.py, specify trained actor file path in the input parameter for actor_file_name (line 19)
2. run model_verifier,py. Observe pygame window to see final node position after pruning
3. press enter to regenerate random map and see new node positions

NOTE:
1. reward strcture is within finder_gym.py
2. ray casting function is within map_tester.py
