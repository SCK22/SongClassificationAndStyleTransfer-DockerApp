from django.contrib import admin
from .models import *

# Register your models here.

class LoadSongsAdmin(admin.ModelAdmin):

    fieldsets = [
        ("Name", {"fields": ['name_of_the_song']}),
        ("upload", {"fields": ['document']}),
        ("Model Predicted", {"fields" : ['is_predicted']})]
    read_only_fields = ('simulation_added_on','is_predicted','uploaded_at')

admin.site.register(LoadSongs, LoadSongsAdmin)