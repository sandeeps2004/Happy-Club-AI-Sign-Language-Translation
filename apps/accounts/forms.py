from django import forms
from django.contrib.auth import get_user_model
from django.contrib.auth.forms import AuthenticationForm, UserCreationForm

from .constants import Roles

User = get_user_model()

# Matches both reference projects exactly
INPUT_CLASS = (
    "w-full border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-800 "
    "rounded-lg px-3 py-2.5 text-sm text-gray-900 dark:text-gray-100 "
    "placeholder-gray-400 dark:placeholder-gray-500 "
    "focus:ring-2 focus:ring-primary-500 focus:border-primary-500 focus:outline-none "
    "transition"
)

SELECT_CLASS = INPUT_CLASS

FILE_CLASS = (
    "w-full text-sm text-gray-500 dark:text-gray-400 "
    "file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 "
    "file:text-sm file:font-semibold "
    "file:bg-primary-50 file:text-primary-600 "
    "dark:file:bg-primary-900/20 dark:file:text-primary-400 "
    "hover:file:bg-primary-100 dark:hover:file:bg-primary-900/30 "
    "file:cursor-pointer file:transition"
)


class LoginForm(AuthenticationForm):
    username = forms.EmailField(
        label="Email",
        widget=forms.EmailInput(attrs={
            "class": INPUT_CLASS,
            "placeholder": "you@example.com",
            "autofocus": True,
        }),
    )
    password = forms.CharField(
        widget=forms.PasswordInput(attrs={
            "class": INPUT_CLASS,
            "placeholder": "Enter your password",
        }),
    )
    error_messages = {
        "invalid_login": "Invalid email or password. Please try again.",
        "inactive": "This account has been deactivated.",
    }


class RegisterForm(UserCreationForm):
    email = forms.EmailField(widget=forms.EmailInput(attrs={"class": INPUT_CLASS, "placeholder": "you@example.com"}))
    first_name = forms.CharField(max_length=150, widget=forms.TextInput(attrs={"class": INPUT_CLASS, "placeholder": "First name"}))
    last_name = forms.CharField(max_length=150, widget=forms.TextInput(attrs={"class": INPUT_CLASS, "placeholder": "Last name"}))
    role = forms.ChoiceField(
        choices=[c for c in Roles.CHOICES if c[0] != Roles.ADMIN],
        initial=Roles.USER,
        widget=forms.Select(attrs={"class": SELECT_CLASS}),
    )
    password1 = forms.CharField(label="Password", widget=forms.PasswordInput(attrs={"class": INPUT_CLASS, "placeholder": "Min 8 characters"}))
    password2 = forms.CharField(label="Confirm Password", widget=forms.PasswordInput(attrs={"class": INPUT_CLASS, "placeholder": "Confirm your password"}))

    class Meta:
        model = User
        fields = ("email", "first_name", "last_name", "role", "password1", "password2")

    def clean_email(self):
        email = self.cleaned_data.get("email", "").lower().strip()
        if User.objects.filter(email__iexact=email).exists():
            raise forms.ValidationError("An account with this email already exists.")
        return email


class ProfileForm(forms.ModelForm):
    first_name = forms.CharField(max_length=150, widget=forms.TextInput(attrs={"class": INPUT_CLASS}))
    last_name = forms.CharField(max_length=150, widget=forms.TextInput(attrs={"class": INPUT_CLASS}))
    phone = forms.CharField(max_length=20, required=False, widget=forms.TextInput(attrs={"class": INPUT_CLASS, "placeholder": "+1 (555) 123-4567"}))
    bio = forms.CharField(required=False, widget=forms.Textarea(attrs={"class": INPUT_CLASS, "rows": 3, "placeholder": "Tell us about yourself..."}))
    avatar = forms.ImageField(required=False, widget=forms.FileInput(attrs={"class": FILE_CLASS, "accept": "image/*"}))

    class Meta:
        model = User
        fields = ("first_name", "last_name", "phone", "bio", "avatar")
