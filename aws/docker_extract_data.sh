# extract data from docker container
containerId=containerId
docker cp ${containerId}:/home/ubuntu/Cat-Classifier/data/keras_model.h5 ~/keras_model.h5
docker cp ${containerId}:/home/ubuntu/Cat-Classifier/data/model_fit.pickle ~/model_fit.pickle
docker cp ${containerId}:/home/ubuntu/Cat-Classifier/data/submissions.csv ~/submissions.csv

# scp data to local windows
scp ~/keras_model.h5 E:\GitHub\Cat-Classifier\data\keras_model.h5
scp ~/model_fit.pickle E:\GitHub\Cat-Classifier\data\model_fit.pickle
scp ~/submissions.csv E:\GitHub\Cat-Classifier\data\submissions.csv
