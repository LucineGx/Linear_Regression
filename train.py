import argparse

from linear_regression_training.linear_regression import LinearRegressionTraining

DEFAULT_FLOAT_PRECISION = 6
SCHOOL_PROJECT_LEARNING_RATE = 1.6


def handle_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="""
        This program is meant to train a machine learning linear regression model of one variable,
        and save the parameters of the resulting model.\n
		Default value of the learning rate is set to match the study case proposed by the school:
		price of a car depending of its kilometrage.
    """)
    parser.add_argument(
        "-r",
        "--learning-rate",
        type=float,
        help="Float value used as learning rate by the gradient descent algorithm.",
        default=SCHOOL_PROJECT_LEARNING_RATE
    )
    parser.add_argument(
        "-f",
        "--training-set-file",
        type=str,
        help="Name or path of the csv file containing the training data set.",
        default="data.csv"
    )
    parser.add_argument(
        "-i",
        "--input-col",
        type=str,
        help="Specify the input column name, when it's not the first column of the csv."
    )
    parser.add_argument(
        "-o",
        "--output-col",
        type=str,
        help="Specify the output column name, when it's not the second column of the csv."
    )
    parser.add_argument(
        "-p",
        "--float-precision",
        type=int,
        help="Specify the precision of the resulting model float parameters.",
        default=DEFAULT_FLOAT_PRECISION
    )
    parser.add_argument(
        "--normalize",
        dest="normalize",
        action="store_true",
        help="[DEFAULT] Normalize the data set."
    )
    parser.add_argument(
        "--no-normalize",
        dest="normalize",
        action="store_false",
        help="Do not normalize the data set."
    )
    parser.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        help="Print debug information about the current execution."
    )
    parser.add_argument(
        "--no-debug",
        dest="debug",
        action="store_false",
        help="[DEFAULT] Do not print debug information about the current execution."
    )
    parser.set_defaults(normalize=True, debug=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = handle_args()
    LinearRegressionTraining(
        learning_rate=args.learning_rate,
        training_set_file=args.training_set_file,
        input_col=args.input_col,
        output_col=args.output_col,
        normalize=args.normalize,
        debug=args.debug,
        float_precision=args.float_precision,
    )
