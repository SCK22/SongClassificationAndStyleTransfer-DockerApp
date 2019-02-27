from django.db import models

# Create your models here.
class LoadSongs(models.Model):
    name_of_the_song = models.CharField(max_length=255, blank=True)
    document = models.FileField(upload_to='media/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    # for now we support only wav file formats
    somg_file_type = models.CharField(max_length=255,default = "wav")
    is_predicted = models.BooleanField(default = 0)
    def __str__(self):
        return self.name_of_the_song