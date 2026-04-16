"""Reconstructed Django view/controller module from the available project source snippets.

This file preserves the structure of the original academic work as closely as possible
while applying formatting and minor safety cleanups for readability.
"""

from pathlib import Path

import pandas as pd
from django.conf import settings
from django.contrib import messages
from django.shortcuts import render

from .algorithms.get_clinical_reports import GetClinicalReports
from .algorithms.get_current_status import MyCurrentStatus
from .algorithms.user_results import UserFinaleports
from .forms import UserRegistrationForm
from .models import UserRegistrationModel


def UserRegisterActions(request):
    if request.method == "POST":
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, "You have been successfully registered")
            form = UserRegistrationForm()
            return render(request, "UserRegistrations.html", {"form": form})

        messages.success(request, "Email or Mobile Already Existed")
    else:
        form = UserRegistrationForm()

    return render(request, "UserRegistrations.html", {"form": form})


def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get("loginname")
        pswd = request.POST.get("pswd")

        try:
            check = UserRegistrationModel.objects.get(loginid=loginid, password=pswd)
            status = check.status
            if status == "activated":
                request.session["id"] = check.id
                request.session["loggeduser"] = check.name
                request.session["loginid"] = loginid
                request.session["email"] = check.email
                return render(request, "users/UserHome.html", {})

            messages.success(request, "Your account is not yet activated")
            return render(request, "UserLogin.html")
        except Exception:
            messages.success(request, "Invalid login id and password")

    return render(request, "UserLogin.html", {})


def UserHome(request):
    return render(request, "users/UserHome.html", {})


def CovidCurrentStatus(request):
    output_path = Path(settings.MEDIA_ROOT) / "coviddata.csv"
    obj = MyCurrentStatus(output_path=output_path)
    df = obj.startCurrentStatus()

    columns = [
        "state",
        "positive",
        "negative",
        "pending",
        "totalTestResults",
        "hospitalizedCurrently",
        "recovered",
        "checkTimeEt",
        "death",
        "total",
    ]
    available_columns = [column for column in columns if column in df.columns]
    table_html = df[available_columns].to_html(index=False, classes="table table-striped")
    return render(request, "users/CovidCurrentData.html", {"data": table_html})


def UserClinicalDataReports(request):
    obj = GetClinicalReports()
    df = obj.startClinicalReports()
    if hasattr(df, "to_html"):
        data = df.to_html(index=False, classes="table table-striped")
    else:
        data = df
    return render(request, "users/UserClinicalData.html", {"data": data})


def UserChestXrayAnalysis(request):
    # Original snippet referenced a subprocess call similar to:
    # python keras-covid-19/train_covid19.py --dataset keras-covid-19/dataset
    return render(request, "users/UserCovidXreayimages.html", {})


def UserResults(request):
    obj = UserFinaleports()
    trainScore, testScore = obj.starProcess()
    return render(
        request,
        "users/UserLstmResults.html",
        {"trainScore": trainScore, "testScore": testScore},
    )
