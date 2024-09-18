from django.db import models

# Create your models here.
class AgeEst(models.Model):
    image=models.ImageField(upload_to='img_upload/')
    result=models.CharField(max_length=405, null=True, blank=True)
    desc=models.CharField(max_length=200, null=True)

    def __str__(self):
        return (f'output{(self.desc).replace(" ", "_")}')