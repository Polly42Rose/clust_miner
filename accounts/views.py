from django.contrib.auth.forms import UserCreationForm
from django.urls import reverse_lazy
from django.views import generic
from django.contrib import messages
from django.shortcuts import redirect, HttpResponse, render
from django.contrib.auth.models import User


class SignUpView(generic.CreateView):
    form_class = UserCreationForm
    success_url = reverse_lazy('login')
    template_name = 'signup.html'


def handleSignUp(request):
    if request.method == "POST":
        # Get the post parameters
        username = request.POST['username']
        pass1 = request.POST['password1']
        pass2 = request.POST['password2']

        # check for errorneous input
        if len(username) > 10:
            messages.error(request, " Длина имени пользователя должна быть менее 10 символов!")
            return redirect('signup')

        if not username.isalnum():
            messages.error(request, " Имя пользователя должно состоять только из букв и цифр!")
            return redirect('signup')

        if pass1 != pass2:
             messages.error(request, " Пароли не совпадают!")
             return redirect('signup')

        if User.objects.filter(username=username).exists():
            messages.error(request, " Имя пользователя занято!")
            return redirect('signup')

        myuser = User.objects.create_user(username, password=pass1)
        myuser.save()
        messages.success(request, " Аккаунт успешно создан! Войдите в аккаунт.")
        return redirect('login')
    else:
        return render(request, 'signup.html')
