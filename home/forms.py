from django import forms


class NSA(forms.Form):
    # imageupload = forms.FileField(label=" Input Image",required= True,widget=forms.FileField(attrs={'class': "form-control", 'placeholder': 'upload image','accept': 'image/*'}))
    photo = forms.FileField(widget=forms.FileInput(
        attrs={'class': "form-control", 'placeholder': 'Student Photo', 'accept': 'image/*'}))
    question = forms.CharField(label="Question",required=True,
                               widget=forms.TextInput(attrs={'class': "form-control", 'placeholder': 'Question'}))




