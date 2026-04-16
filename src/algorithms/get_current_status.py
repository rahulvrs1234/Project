"""COVID-19 current-status analytics reconstructed from the project snippet."""

from __future__ import annotations

from pathlib import Path

import io
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests


class MyCurrentStatus:
    def __init__(self, output_path: Optional[Path] = None):
        self.output_path = Path(output_path) if output_path else None

    def startCurrentStatus(self, generate_plots: bool = False) -> pd.DataFrame:
        url = "http://covidtracking.com/api/states/daily.csv"
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        df = pd.read_csv(io.StringIO(response.content.decode("utf-8")))
        df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
        if "dateChecked" in df.columns:
            df.drop(["dateChecked"], axis=1, inplace=True)
        df["state"] = df["state"].astype(str)
        df.fillna(value=-1, inplace=True)

        if self.output_path:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(self.output_path, index=False)

        if generate_plots:
            self._generate_summary_plots(df)

        return df

    def _generate_summary_plots(self, df: pd.DataFrame) -> None:
        self._plot_var(df, var="positiveIncrease", state="NY")
        self._plot_var(df, var="hospitalizedIncrease", state="GA")
        self._plot_var(df, var="deathIncrease", state="MI")
        self._plot_var(df, var="totalTestResultsIncrease", state="MA")

        self._plot_xy(df, state="NY")
        self._plot_xy(df, "hospitalized", "death", "GA")
        self._plot_xy(df, "hospitalized", "death", "CA")
        self._plot_testing(df)
        self._plot_ratio_bars(df)
        self._plot_ratio_scatter(df)
        self._plot_case_count_trends(df)

    @staticmethod
    def _plot_var(df: pd.DataFrame, var: str = "positiveIncrease", state: str = "NY") -> None:
        y = df[df["state"] == state][var]
        x = df[df["state"] == state]["date"]
        plt.figure(figsize=(12, 4))
        plt.title(f'Plot of "{var}" for {state}', fontsize=18)
        plt.bar(x=x, height=y, edgecolor="k", color="orange")
        plt.grid(True)
        plt.xticks(fontsize=14, rotation=45)
        plt.yticks(fontsize=14)
        plt.show()

    @staticmethod
    def _plot_xy(
        df: pd.DataFrame,
        varx: str = "totalTestResultsIncrease",
        vary: str = "positiveIncrease",
        state: str = "NY",
    ) -> None:
        y = df[df["state"] == state][vary]
        x = df[df["state"] == state][varx]
        if (x.nunique() != 1) and (y.nunique() != 1):
            plt.figure(figsize=(12, 4))
            plt.title(f'Plot of "{varx}" vs. "{vary}" for {state}', fontsize=18)
            plt.scatter(x=x, y=y, edgecolor="k", color="lightgreen", s=100)
            plt.grid(True)
            plt.xticks(fontsize=14, rotation=45)
            plt.yticks(fontsize=14)
            plt.show()

    @staticmethod
    def _plot_testing(df: pd.DataFrame, lst_states=None) -> None:
        if lst_states is None:
            lst_states = ["NY", "CA", "MA", "TX", "PA"]

        legends = []
        plt.figure(figsize=(10, 5))
        plt.title("Total test results", fontsize=18)
        for state in lst_states:
            data = np.array(df[df["state"] == state]["totalTestResults"])[-1::-1]
            slope = int((data[-1] - data[0]) / len(data))
            plt.plot(data, linewidth=2)
            plt.text(x=len(data) - 2, y=data[-1] * 1.05, s=state, fontsize=14)
            legends.append(f"{slope} tests/day in {state}")
        plt.legend(legends, fontsize=14)
        plt.grid(True)
        plt.xlim(0, len(data) + 2)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.xlabel("Days", fontsize=16)
        plt.ylabel("Total test results", fontsize=16)
        plt.show()

    @staticmethod
    def _fatality_ratio(df: pd.DataFrame, state: str = "CA") -> float:
        date = df.iloc[0]["date"]
        try:
            death = float(df[(df["state"] == state) & (df["date"] == date)]["death"].iloc[0])
            positive = float(df[(df["state"] == state) & (df["date"] == date)]["positive"].iloc[0])
        except Exception:
            return -1
        if death != -1.0 and positive not in (-1.0, 0):
            return round(death / positive, 3)
        return -1

    @staticmethod
    def _hospitalization_ratio(df: pd.DataFrame, state: str = "NY") -> float:
        date = df.iloc[0]["date"]
        try:
            hospitalized = float(
                df[(df["state"] == state) & (df["date"] == date)]["hospitalized"].iloc[0]
            )
            positive = float(df[(df["state"] == state) & (df["date"] == date)]["positive"].iloc[0])
        except Exception:
            return -1
        if hospitalized != -1.0 and positive not in (-1.0, 0):
            return round(hospitalized / positive, 3)
        return -1

    @staticmethod
    def _positive_test_ratio(df: pd.DataFrame, state: str = "NY") -> float:
        date = df.iloc[0]["date"]
        try:
            positive = float(df[(df["state"] == state) & (df["date"] == date)]["positive"].iloc[0])
            total_tests = float(
                df[(df["state"] == state) & (df["date"] == date)]["totalTestResults"].iloc[0]
            )
        except Exception:
            return -1
        if positive != -1.0 and total_tests not in (-1.0, 0):
            return round(positive / total_tests, 3)
        return -1

    @staticmethod
    def _plot_ratio_bars(df: pd.DataFrame) -> None:
        states = ["CA", "NY", "MI", "MA", "PA", "IL"]

        fatality_values, fatality_states = [], []
        for state in states:
            value = MyCurrentStatus._fatality_ratio(df, state)
            if value != -1:
                fatality_values.append(value)
                fatality_states.append(state)
        plt.figure(figsize=(8, 4))
        plt.title("Fatality ratio chart", fontsize=18)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.bar(x=fatality_states, height=fatality_values, color="red", edgecolor="k", linewidth=2)
        plt.show()

        hospitalization_values, hospitalization_states = [], []
        for state in states:
            value = MyCurrentStatus._hospitalization_ratio(df, state)
            if value != -1:
                hospitalization_values.append(value)
                hospitalization_states.append(state)
        plt.figure(figsize=(8, 4))
        plt.title("Hospitalization ratio chart", fontsize=18)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.bar(
            x=hospitalization_states,
            height=hospitalization_values,
            color="brown",
            edgecolor="k",
            linewidth=2,
        )
        plt.show()

        positive_values, positive_states = [], []
        for state in states:
            value = MyCurrentStatus._positive_test_ratio(df, state)
            if value != -1:
                positive_values.append(value)
                positive_states.append(state)
        plt.figure(figsize=(8, 4))
        plt.title("Test-positive ratio chart", fontsize=18)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.bar(x=positive_states, height=positive_values, color="blue", edgecolor="k", linewidth=2)
        plt.show()

    @staticmethod
    def _plot_ratio_scatter(df: pd.DataFrame) -> None:
        states = list(df["state"].unique())
        for state in ["AS", "GU", "MP", "PU", "VI"]:
            if state in states:
                states.remove(state)

        fatality_values, fatality_states = [], []
        for state in states:
            value = MyCurrentStatus._fatality_ratio(df, state)
            if value != -1:
                fatality_values.append(value)
                fatality_states.append(state)

        fatality_values = np.array(fatality_values)
        plt.figure(figsize=(15, 7))
        plt.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
        plt.title("Fatality ratio across the states", fontsize=18)
        plt.scatter(
            x=fatality_states,
            y=fatality_values,
            s=4e5 * fatality_values**2,
            color="orange",
            edgecolor="red",
            alpha=0.75,
            linewidth=2.5,
        )
        for index, state in enumerate(fatality_states):
            plt.annotate(text=state, xy=(fatality_states[index], fatality_values[index]))
        plt.ylim(0, 0.12)
        plt.yticks(fontsize=16)
        plt.grid(True, axis="y")
        plt.show()

        positive_values, positive_states = [], []
        for state in states:
            value = MyCurrentStatus._positive_test_ratio(df, state)
            if value != -1:
                positive_values.append(value)
                positive_states.append(state)

        positive_values = np.array(positive_values)
        plt.figure(figsize=(15, 7))
        plt.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
        plt.title("Test-positive ratio across the states", fontsize=18)
        plt.scatter(
            x=positive_states,
            y=positive_values,
            s=2e4 * positive_values**2,
            color="blue",
            edgecolor="red",
            alpha=0.5,
            linewidth=2,
        )
        for index, state in enumerate(positive_states):
            plt.annotate(text=state, xy=(positive_states[index], positive_values[index]))
        plt.ylim(0, 0.6)
        plt.yticks(fontsize=16)
        plt.grid(True, axis="y")
        plt.show()

    @staticmethod
    def _case_counts_decrease(df: pd.DataFrame, days: int = 14, state: str = "NY"):
        positive_increase = np.array(df[df["state"] == state]["positiveIncrease"][: days + 1])[-1::-1]
        diff = np.diff(positive_increase)
        count_of_neg = np.sum(diff <= 0, axis=0)
        return count_of_neg, diff

    @staticmethod
    def _plot_case_count_trends(df: pd.DataFrame) -> None:
        states = ["CA", "MI", "GA", "LA"]
        x = np.arange(1, 15, 1)
        plt.figure(figsize=(10, 6))
        plt.title(
            "Last 14 days successive difference in new positive cases\n(more negative numbers is better)",
            fontsize=18,
        )
        for state in states:
            _, data = MyCurrentStatus._case_counts_decrease(df, days=14, state=state)
            plt.plot(x, data, linewidth=2)
        plt.legend(states, fontsize=16, ncol=2)
        plt.grid(True)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.xlabel("Last 14 days", fontsize=16)
        plt.ylabel("Successive difference in new cases", fontsize=16)
        plt.hlines(y=0, xmin=0, xmax=15, linestyles="--", lw=3)
        plt.show()
