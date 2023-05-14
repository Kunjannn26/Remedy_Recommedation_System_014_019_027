from django.contrib import admin
from .models import lab_tests,know,homeremedyinfo,ayurvedicinfo, patient , doctor , diseaseinfo , consultation,rating_review, allopathyinfo

# Register your models here.

admin.site.register(patient)
admin.site.register(doctor)
admin.site.register(diseaseinfo)
admin.site.register(allopathyinfo)
admin.site.register(ayurvedicinfo)
admin.site.register(homeremedyinfo)
admin.site.register(know)
admin.site.register(lab_tests)
admin.site.register(consultation)
admin.site.register(rating_review)