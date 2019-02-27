from rest_framework import serializers
from .models import *

class SongsInputSerializer(serializers.ModelSerializer):
    """Serializer to map the model instance into JSON format."""
    class Meta:
        """Meta class to map serializer's fields with the model fields. """
        model = LoadSongs
        fields = ('id','name_of_the_song','document','uploaded_at','is_predicted')
        read_only_fields = ('simulation_added_on','is_predicted','uploaded_at')
