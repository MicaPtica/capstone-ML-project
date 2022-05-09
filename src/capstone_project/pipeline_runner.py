from pathlib import Path
from joblib import dump

import click
from data_handler import get_dataset,get_dataset_splitted
from pipeline_creator import create_pipeline

from sklearn.model_selection import cross_validate

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import make_scorer

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

@click.option(
    "--cv",
    default=True,
    type=bool,
    show_default=True)

@click.option(
    "--cv-k",
    default=5,
    type=int,
    show_default=True)



def run_pipeline(dataset_path,save_model_path,random_state,test_split_ratio,use_scaler,max_iter,logreg_c,cv,cv_k):
    if cv==True:
        print('CV on ')
        scoring = {'accuracy': 'accuracy',
            'precision': make_scorer(recall_score, average='macro'),
            'recall': make_scorer(recall_score, average='macro')
            }
        pipeline = create_pipeline(use_scaler, max_iter, logreg_c, random_state)
        features, target=get_dataset(dataset_path)
        scores = cross_validate(pipeline, features, target, scoring=scoring,cv=cv_k,return_train_score=True)
        
        click.echo(f"Training Accuracy scores: {scores['train_accuracy']}.")
        click.echo(f"Training Precision  scores: {scores['train_precision']}.")
        click.echo(f"Training Recall  scores: {scores['train_recall']}.")

    else:

        features_train, features_val, target_train, target_val=get_dataset_splitted(dataset_path,random_state,test_split_ratio)
        pipeline = create_pipeline(use_scaler, max_iter, logreg_c, random_state)
        pipeline.fit(features_train, target_train)
        accuracy = accuracy_score(target_val, pipeline.predict(features_val))
        click.echo(f"Accuracy: {accuracy}.")
        dump(pipeline, save_model_path)

if __name__ == '__main__':
    run_pipeline()


