from .models import Log, User
from django.forms import ModelForm, FileInput, TextInput, PasswordInput


class LogForm(ModelForm):
    class Meta:
        model = Log
        fields = ["title", "xes_file", "n_traces", "attributes", "user_id"]
        widgets = {
            "title": TextInput(attrs={
                'id': 'log-title',
                'required': True,
                'class': 'form-control',
                'placeholder': 'Введите название лога'
            }),
            "xes_file": FileInput(attrs={
                'id': 'log-file',
                'required': True,
                'class': 'form-control',
                'type': 'file',
                'label': 'Выберете файл для загрузки'
            })
        }


class UserForm(ModelForm):
    class Meta:
        model = User
        fields = "__all__"
        widgets = {
            "login": TextInput(attrs={
                'id': 'login',
                'required': True,
                'class': 'form-control',
                'placeholder': 'Введите логин'
            }),
            "password": PasswordInput(attrs={
                'id': 'password',
                'required': True,
                'class': 'form-control',
                'placeholder': 'Введите пароль'
            }),
            "password2": PasswordInput(attrs={
                'id': 'password',
                'required': True,
                'class': 'form-control',
                'placeholder': 'Повторите пароль'
            })
        }

