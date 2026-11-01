from django.urls import path
from . import views
urlpatterns=[
 path('',views.dashboard,name='dashboard'), path('signup/',views.signup,name='signup'), path('onboarding/',views.onboarding,name='onboarding'),
 path('rate/',views.rate,name='rate'), path('search/',views.search,name='search'),
 path('movie/<int:movie_id>/',views.movie_detail,name='movie_detail'),
 path('preferences/',views.preferences,name='preferences'), path('health/',views.health,name='health'),
]
