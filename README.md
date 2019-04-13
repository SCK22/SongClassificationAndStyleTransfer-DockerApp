# SongClassificationAndStyleTransfer

You can download the docker using the command docker pull jarvis42/song:latest 
Docker : https://cloud.docker.com/u/jarvis42/repository/docker/jarvis42/song

Note : Songs might not play on some systems, the prediction can still be obtained using the app.

run 
docker run -p 8000:8000 jarvis42/song python manage.py runserver 0.0.0.0:8000
