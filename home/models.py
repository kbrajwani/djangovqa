import datetime

from django.db import models

# Create your models here.
class NSAI(models.Model):
    photo = models.FileField(upload_to='')
    question = models.CharField(max_length=128)

    def __str__(self):
        return "{}".format(datetime.datetime.now())