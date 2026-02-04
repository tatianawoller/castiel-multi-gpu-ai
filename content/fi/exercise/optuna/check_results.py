import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend


if __name__ == "__main__":
    study = optuna.load_study("mnist", storage=JournalStorage(JournalFileBackend(file_path="./journal.log")))

    # Get all trials
    trials = study.trials

    # Print basic info
    print(f"Study name: {study.study_name}")
    print(f"Number of trials: {len(trials)}")
    print(f"Best trial value: {study.best_trial.value}")
    print(f"Best parameters: {study.best_trial.params}")
    print(f"Best trial index: {study.best_trial.number}")
