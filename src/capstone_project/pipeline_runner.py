from pathlib import Path
from joblib import dump

import click
from data_handler import get_dataset
from pipeline_creator import create_pipeline


@click.command()
@click.option(
    "-d",
    "--dataset-path",
    default="data/train.csv",
    # type=click.Path(exists=True, dir_okay=False, path_type=Path),
    show_default=True)

@click.option(
    "-s",
    "--save-model-path",
    default="data/model.joblib",
    # type=click.Path(dir_okay=False, writable=True, path_type=Path),
    show_default=True)

@click.option(
    "--random-state",
    default=42,
    type=int,
    show_default=True)

@click.option(
    "--test-split-ratio",
    default=0.2,
    # type=click.FloatRange(0, 1, min_open=True, max_open=True),
    show_default=True)

@click.option(
    "--use-scaler",
    default=True,
    type=bool,
    show_default=True)

@click.option(
    "--max-iter",
    default=1000,
    type=int,
    show_default=True)

@click.option(
    "--logreg-c",
    default=1.0,
    type=float,
    show_default=True)

def run_pipeline(dataset_path,save_model_path,random_state,test_split_ratio,use_scaler,max_iter,logreg_c):
    features_train, features_val, target_train, target_val=get_dataset(dataset_path,random_state,test_split_ratio)
    pipeline = create_pipeline(use_scaler, max_iter, logreg_c, random_state)
    pipeline.fit(features_train, target_train)
    dump(pipeline, save_model_path)



if __name__ == '__main__':
    run_pipeline()


