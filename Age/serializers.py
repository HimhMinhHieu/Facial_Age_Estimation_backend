from rest_framework import serializers
from .models import AgeEst

class AgeSer(serializers.ModelSerializer):
    class Meta:
        model=AgeEst
        fields='__all__'