import argparse
import shelve


def predict(precision: int) -> None:
    sh = shelve.open("linear_regression")
    theta0, theta1 = sh.get("theta0", 0), sh.get("theta1", 0)
    input_col, output_col = sh.get("input_col", "explanatory variable"), sh.get("output_col", "result")
    sh.close()

    raw_x = input(f"Provide {input_col} value: ")
    try:
        x = float(raw_x)
    except ValueError:
        print("\033[91mpredict.py: error:\033[0m The input value should be a float or an integer.")
    else:
        prediction = theta0 + theta1 * x
        print(f"Predicted {output_col} for {input_col} = {x} is {round(prediction, precision)}.")


def handle_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="""
        This program is meant make a prediction with a regression model of one variable, using
        parameters saved by its training program.
        The input value is requested after the program launch.
    """)
    parser.add_argument(
        "-p",
        "--precision",
        type=int,
        help="Precision of the returned prediction.",
        default=0
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = handle_args()
    predict(args.precision)
