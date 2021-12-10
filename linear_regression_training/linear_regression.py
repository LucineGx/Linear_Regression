from typing import Optional
from functools import partial
import warnings

import pandas as pd
import numpy as np
import shelve
import matplotlib.pyplot as plt


class LinearRegressionTraining:
    error_msg = None
    learning_rate = None
    training_set = None
    normalize = True
    input_col = None
    output_col = None
    float_precision = 0

    input_max = None
    theta0 = 0
    theta1 = 0

    def __init__(
            self,
            learning_rate: float,
            training_set_file: str,
            input_col: Optional[str] = None,
            output_col: Optional[str] = None,
            normalize: bool = True,
            debug: bool = False,
            float_precision: int = 0
    ):
        self.learning_rate = learning_rate
        self.normalize = normalize
        self.debug = debug
        self.float_precision = float_precision

        if self.validate_args(training_set_file, input_col, output_col):
            self.train_model()
        else:
            self.print_error()

    def validate_args(
            self, training_set_file: str, input_col: Optional[str], output_col: Optional[str]
    ) -> bool:
        if self.validate_training_set_file(training_set_file):
            if self.validate_input_output_col(input_col, output_col):
                return True
        return False

    def validate_training_set_file(self, training_set_file: str) -> bool:
        if not training_set_file:
            self.error_msg = "missing training set file name"
            return False
        try:
            self.training_set = pd.read_csv(training_set_file)
        except (FileNotFoundError, PermissionError) as e:
            self.error_msg = e
            return False
        return True

    def validate_input_output_col(
            self, input_col: Optional[str], output_col: Optional[str]) -> bool:
        columns = self.training_set.columns

        if (input_col and input_col not in columns) or (output_col and output_col not in columns):
            self.error_msg = "can't find matching columns for input-col and output-col"
            return False

        if input_col:
            self.input_col = input_col
        else:
            self.input_col = self.training_set.columns[0]
        if output_col:
            self.output_col = output_col
        else:
            self.output_col = self.training_set.columns[1]
        return True

    def train_model(self):
        self.format_data()
        self.gradient_descent()
        if self.error_msg:
            self.print_error()
        else:
            self.show_graph()
            self.save_parameters()

    def format_data(self):
        input_values = np.array(self.training_set[self.input_col])
        output_values = np.array(self.training_set[self.output_col])

        if self.normalize:
            self.input_max = np.max(input_values)
            input_values = input_values / self.input_max

        self.training_set = (
            pd.DataFrame({"x": input_values, "y": output_values}).sort_values(["x"])
        )

    def gradient_descent(self):
        m = self.training_set.shape[0]
        new_theta0 = None
        new_theta1 = None

        iterations = 0
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            while not self.has_converged(new_theta0, new_theta1, iterations):
                try:
                    new_theta0 = self.partially_derive_cost_with_respect_to_theta0(self, m)
                    new_theta1 = self.partially_derive_cost_with_respect_to_theta1(self, m)
                except RuntimeWarning:
                    new_theta0, new_theta1 = None, None

                iterations += 1
                if not iterations % 1000 and self.debug:
                    print(f"{iterations}: \ttheta0 = {new_theta0}\n\ttheta1 = {new_theta1}\n")

        if self.error_msg:
            print(f"\033[93mGradient Descent was interrupted after {iterations} iterations.\033[0m")
        else:
            print(f"\033[92mGradient Descent succeed after {iterations} iterations !\033[0m")

    def has_converged(
            self, new_theta0: Optional[float], new_theta1: Optional[float], iterations: int
    ) -> bool:
        if new_theta0 and new_theta1:
            try:
                if (
                    round(new_theta0, self.float_precision) == round(self.theta0, self.float_precision)
                    and
                    round(new_theta1, self.float_precision) == round(self.theta1, self.float_precision)
                ):
                    if self.debug:
                        print(f"{iterations}: Converge:\t{new_theta0} +-= {self.theta0}\n\t\t{new_theta1} +-= {self.theta1}")
                    return True
                else:
                    self.theta0, self.theta1 = new_theta0, new_theta1
                    return False
            except RuntimeWarning:
                if self.debug:
                    print(f"{iterations}: RunTime warning when comparing old and new thetas.")
                self.theta0, self.theta1 = None, None

        if not iterations:
            return False
        self.error_msg = "Learning rate is too big, the gradient descent is diverging."
        return True

    def derive_cost(self, m: int, variable_name: str, partially_derive: callable):
        partial_derivative_result = partially_derive(self.theta0, self.theta1, self.training_set, m)
        return self.__getattribute__(variable_name) - self.learning_rate * partial_derivative_result

    partially_derive_cost_with_respect_to_theta0 = partial(
        derive_cost,
        variable_name="theta0",
        partially_derive=lambda theta0, theta1, training_set, m: sum([
            theta0 + theta1 * row['x'] - row['y']
            for _, row
            in training_set.iterrows()
        ]) / m
    )
    partially_derive_cost_with_respect_to_theta1 = partial(
        derive_cost,
        variable_name="theta1",
        partially_derive=lambda theta0, theta1, training_set, m: sum([
            (theta0 + theta1 * row['x'] - row['y']) * row['x']
            for _, row
            in training_set.iterrows()
        ]) / m
    )

    def show_graph(self) -> None:
        if self.normalize:
            self.training_set['x'] = np.array(self.training_set['x']) * self.input_max
            self.theta1 = self.theta1 / self.input_max
        self.training_set['prediction'] = self.training_set['x'].map(lambda x: self.theta0 + self.theta1 * x)
        self.training_set.plot(x="x", y=["y", "prediction"], kind="line")
        plt.show()

    def save_parameters(self) -> None:
        if self.debug:
            print(f"theta0 = {self.theta0}\ntheta1 = {self.theta1}")
            print(f"prediction cost: {self.compute_cost()}")

        sh = shelve.open("linear_regression")
        sh["theta0"] = self.theta0
        sh["theta1"] = self.theta1
        sh["input_col"] = self.input_col
        sh["output_col"] = self.output_col
        sh.close()

    def compute_cost(self) -> float:
        m = self.training_set.shape[0]
        return (1 / (2 * m)) * sum([
            (self.theta0 + self.theta1 * row['x'] - row['y']) ** 2
            for _, row in self.training_set.iterrows()
        ])

    def print_error(self) -> None:
        print(f"\033[91mtraining.py: error:\033[0m {self.error_msg}")
