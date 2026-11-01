from django.conf import settings
from django.db import models

class Profile(models.Model):
    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="profile")
    favorite_genres = models.JSONField(default=list, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    def __str__(self): return self.user.username

class UserRating(models.Model):
    profile = models.ForeignKey(Profile, on_delete=models.CASCADE, related_name='ratings')
    movie_id = models.IntegerField()
    movie_title = models.CharField(max_length=255)
    rating = models.FloatField()
    created_at = models.DateTimeField(auto_now=True)
    class Meta:
        constraints=[models.UniqueConstraint(fields=['profile','movie_id'], name='unique_profile_movie')]
        ordering=['-created_at']
    def __str__(self): return f'{self.profile} — {self.movie_title} ({self.rating})'
