# set container id
containerId=`docker ps | grep oislen/cat-classifier:latest | awk '{print $1;}'`
# extract data from docker container
docker cp ${containerId}:/home/ubuntu/Cat-Classifier/data/keras_model.h5 ~/keras_model.h5
docker cp ${containerId}:/home/ubuntu/Cat-Classifier/data/model_fit.pickle ~/model_fit.pickle
docker cp ${containerId}:/home/ubuntu/Cat-Classifier/data/submission.csv ~/submission.csv
docker cp ${containerId}:/home/ubuntu/Cat-Classifier/data/checkpoints ~/checkpoints
# extract report from docker container
docker cp ${containerId}:/home/ubuntu/Cat-Classifier/report/generator_plot.jpg ~/generator_plot.jpg
docker cp ${containerId}:/home/ubuntu/Cat-Classifier/report/model_accuracy.png ~/model_accuracy.png
docker cp ${containerId}:/home/ubuntu/Cat-Classifier/report/model_loss.png ~/model_loss.png
docker cp ${containerId}:/home/ubuntu/Cat-Classifier/report/pred_images.jpg ~/pred_images.jpg
docker cp ${containerId}:/home/ubuntu/Cat-Classifier/report/random_image.jpg ~/random_image.jpg