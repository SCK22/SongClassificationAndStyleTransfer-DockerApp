# SongClassificationAndStyleTransfer

You can download the docker using the command ```docker pull jarvis42/song:latest``` or use the link
https://cloud.docker.com/u/jarvis42/repository/docker/jarvis42/song

Note : Songs might not play on some systems, the prediction can still be obtained using the app.

```
# to pull docker from dockerhub, login to your docker hub account from terminal and run

docker pull jarvis42/song:latest
```

```
# to run the docker use the following command

docker run -p 8000:8000 jarvis42/song python manage.py runserver 0.0.0.0:8000
```

> djangorestui folder
- Has the code for loading the model and giving the prediction
- Has the code for the django app-
- Has the model definitions and the pickle files necessary for the model
