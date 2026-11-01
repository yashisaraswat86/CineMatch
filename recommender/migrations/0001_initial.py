from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion
class Migration(migrations.Migration):
 initial=True
 dependencies=[migrations.swappable_dependency(settings.AUTH_USER_MODEL)]
 operations=[
  migrations.CreateModel(name='Profile',fields=[('id',models.BigAutoField(auto_created=True,primary_key=True,serialize=False,verbose_name='ID')),('favorite_genres',models.JSONField(blank=True,default=list)),('created_at',models.DateTimeField(auto_now_add=True)),('updated_at',models.DateTimeField(auto_now=True)),('user',models.OneToOneField(on_delete=django.db.models.deletion.CASCADE,related_name='profile',to=settings.AUTH_USER_MODEL))]),
  migrations.CreateModel(name='UserRating',fields=[('id',models.BigAutoField(auto_created=True,primary_key=True,serialize=False,verbose_name='ID')),('movie_id',models.IntegerField()),('movie_title',models.CharField(max_length=255)),('rating',models.FloatField()),('created_at',models.DateTimeField(auto_now=True)),('profile',models.ForeignKey(on_delete=django.db.models.deletion.CASCADE,related_name='ratings',to='recommender.profile'))]),
  migrations.AddConstraint(model_name='userrating',constraint=models.UniqueConstraint(fields=('profile','movie_id'),name='unique_profile_movie')),
 ]
