from django.contrib import admin
from .models import NSAI

# Register your models here.

@admin.register(NSAI)
class NSAIAdmin(admin.ModelAdmin):
    pass

