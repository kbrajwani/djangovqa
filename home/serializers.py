from rest_framework import serializers
from .models import NSAI


class NSAISerializer(serializers.ModelSerializer):

    class Meta:

        model = NSAI
        fields = '__all__'