from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
class SignupForm(UserCreationForm):
    email=forms.EmailField(required=False)
    class Meta: model=User; fields=('username','email','password1','password2')
class RatingForm(forms.Form):
    movie_id=forms.IntegerField(widget=forms.HiddenInput())
    movie_title=forms.CharField(widget=forms.HiddenInput())
    rating=forms.FloatField(min_value=.5,max_value=5,widget=forms.NumberInput(attrs={'step':'.5','min':'.5','max':'5'}))
