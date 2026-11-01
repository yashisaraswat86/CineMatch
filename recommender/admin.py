from django.contrib import admin
from .models import Profile, UserRating
@admin.register(Profile)
class ProfileAdmin(admin.ModelAdmin): list_display=('user','created_at','updated_at'); search_fields=('user__username','user__email')
@admin.register(UserRating)
class UserRatingAdmin(admin.ModelAdmin): list_display=('profile','movie_title','rating','created_at'); list_filter=('rating',); search_fields=('movie_title','profile__user__username')
