from django.http import HttpResponse
from django.shortcuts import render
import joblib

def home(request):
    return render(request, "home.html")

def result(request):
    # Load the trained model
    cls = joblib.load('final_model.sav')

    # List of field names
    field_names = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']

    # List to hold input values
    lis = []

    # Check each form field and append its value to the list
    for field in field_names:
        value = request.POST.get(field)  # Use POST instead of GET
        if not value:  # If the field is empty or missing
            return HttpResponse(f"Error: Missing or empty input for {field}. Please fill all fields.")

        try:
            lis.append(float(value))  # Convert to float
        except ValueError:
            return HttpResponse(f"Error: Invalid input format for {field}. Please enter a valid number.")

    # Predict using the model
    ans = cls.predict([lis])

    # Return the result
    return render(request, "result.html", {'ans': ans[0], 'lis': lis})


# from django.http import HttpResponse
# from django.shortcuts import render
# import joblib
#
# def home(request):
#     #return HttpResponse("Hello")
#     return render (request,"home.html")
#
# def result(request):
#     cls = joblib.load('final_model.sav')
#
#     lis = []
#
#     lis.append(request.GET['RI'])
#     lis.append(request.GET['Na'])
#     lis.append(request.GET['Mg'])
#     lis.append(request.GET['Al'])
#     lis.append(request.GET['Si'])
#     lis.append(request.GET['K'])
#     lis.append(request.GET['Ca'])
#     lis.append(request.GET['Ba'])
#     lis.append(request.GET['Fe'])
#
#     print(lis)
#     ans = cls.predict([lis])
#
#     return render(request,"result.html",{'ans':ans})
