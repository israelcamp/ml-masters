from typing import Dict, List
import os

import tensorflow as tf
import tensorboard
from tensorboard.plugins.hparams import api_pb2
from tensorboard.plugins.hparams import summary as hparams_summary


def run_hparam_trial(run_dir: str, hparams: Dict[str, float], trainer: callable,
                     name: str = 'accuracy', description: str = 'The accuracy', step=0):

    writer = tf.summary.create_file_writer(run_dir)
    summary_start = hparams_summary.session_start_pb(hparams=hparams)

    with writer.as_default():
        metric = trainer(hparams)
        summary_end = hparams_summary.session_end_pb(api_pb2.STATUS_SUCCESS)

        tf.summary.scalar(name, metric, step=step,
                          description=description)
        tf.summary.import_event(tf.compat.v1.Event(
            summary=summary_start).SerializeToString())
        tf.summary.import_event(tf.compat.v1.Event(
            summary=summary_end).SerializeToString())
        return metric


def run_search_for_hparams(LOG_DIR: str, run_func: callable,
                           hparams_generator: callable,
                           hparams_generator_kwargs: dict,
                           verbose: bool = False) -> List[dict]:

    trials_hparams = []
    for (session_index, hparams) in enumerate(hparams_generator(**hparams_generator_kwargs)):
        run_dir = f'trial{session_index}'
        path = os.path.join(LOG_DIR, f'hparams_tuning/{run_dir}')

        metric = run_func(path, hparams)
        trials_hparams.append({
            'run_dir': path,
            'hparams': hparams,
            'metric': metric
        })

        if verbose:
            print(
                f'Run number {session_index+1} for {hparams} and {metric} saved at {path}')

    return trials_hparams
