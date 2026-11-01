from django.contrib import messages
from django.contrib.auth import login
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.shortcuts import get_object_or_404, redirect, render
from .engine import get_engine
from .forms import RatingForm, SignupForm
from .models import Profile, UserRating

def ensure_profile(user):
    p,_=Profile.objects.get_or_create(user=user); return p

def signup(request):
    if request.user.is_authenticated:return redirect('dashboard')
    form=SignupForm(request.POST or None)
    if form.is_valid():
        user=form.save(); login(request,user); ensure_profile(user); return redirect('onboarding')
    return render(request,'registration/signup.html',{'form':form})

def dashboard(request):
    if not request.user.is_authenticated:return render(request,'recommender/home.html')
    p=ensure_profile(request.user); ratings=list(p.ratings.values_list('movie_id','rating'))
    try:
        e=get_engine(); recs=e.recommend(ratings).to_dict('records'); catalog=e.search('',30).to_dict('records'); meta=e.metadata
    except FileNotFoundError: recs=[]; catalog=[]; meta={}
    return render(request,'recommender/dashboard.html',{'profile':p,'recommendations':recs,'catalog':catalog,'ratings':p.ratings.all(),'meta':meta,'status':'personalized' if ratings else 'discover'})

def onboarding(request):
    if not request.user.is_authenticated:return redirect('login')
    p=ensure_profile(request.user)
    try: genres=get_engine().genres
    except FileNotFoundError: genres=[]
    if request.method=='POST':
        p.favorite_genres=request.POST.getlist('genres'); p.save(); return redirect('dashboard')
    return render(request,'recommender/onboarding.html',{'genres':genres,'profile':p})

@login_required
def rate(request):
    p=ensure_profile(request.user); form=RatingForm(request.POST or None)
    if request.method=='POST' and form.is_valid():
        d=form.cleaned_data; UserRating.objects.update_or_create(profile=p,movie_id=d['movie_id'],defaults={'rating':d['rating'],'movie_title':d['movie_title']}); messages.success(request,f"Saved your rating for {d['movie_title']}.")
    else:
        messages.error(request,'Please enter a valid rating between 0.5 and 5.')
    return redirect(request.POST.get('next') or 'dashboard')

@login_required
def search(request):
    q=request.GET.get('q','')
    try: results=get_engine().search(q).to_dict('records')
    except FileNotFoundError: results=[]
    return render(request,'recommender/search.html',{'results':results,'query':q})

@login_required
def movie_detail(request,movie_id):
    try: e=get_engine(); row=e.movies[e.movies.movieId.eq(movie_id)].iloc[0].to_dict()
    except (FileNotFoundError,IndexError): return redirect('dashboard')
    p=ensure_profile(request.user); previous=p.ratings.filter(movie_id=movie_id).first()
    return render(request,'recommender/movie.html',{'movie':row,'previous':previous})

@login_required
def preferences(request):
    return onboarding(request)

def health(request):
    try: e=get_engine(); return render(request,'recommender/health.html',{'ok':True,'meta':e.metadata})
    except Exception as exc: return render(request,'recommender/health.html',{'ok':False,'error':str(exc)})
