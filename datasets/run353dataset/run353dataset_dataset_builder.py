"""run353dataset dataset."""

import tensorflow_datasets as tfds
import numpy as np
import csv


class Builder(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for run353dataset dataset."""

  VERSION = tfds.core.Version('1.1.2')
  RELEASE_NOTES = {
      '1.1.2': 'Finalized point selection; no more noisy data.',
  }
  MANUAL_DOWNLOAD_INSTRUCTIONS = """the raw data is not currently available for download"""

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(run353dataset): Specifies the tfds.core.DatasetInfo object
    return self.dataset_info_from_configs(
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            'eventNo': tfds.features.Scalar(dtype = np.int64, doc = 'event number (ID)'),
            'time_label': tfds.features.Scalar(dtype = np.float64, doc = 'PICOSEC-MM arrival time from full signal analysis'),
            'signal_data': tfds.features.Tensor(shape = (64,), dtype = np.float64, doc = '64 signal points for ANN input'),
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=('time_label', 'signal_data'),  # Set to `None` to disable
        homepage='https://dummy-page.org',
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # TODO(run353dataset): Downloads the data and defines the splits
    path = dl_manager.manual_dir / 'Run353.zip'
    extracted_path = dl_manager.extract(path)

    # TODO(run353dataset): Returns the Dict[split names, Iterator[Key, Example]]
    return {
        'train': self._generate_examples(extracted_path / 'Run353.csv'),
    }

  def _generate_examples(self, path):
    """Yields examples."""
    # TODO(run353dataset): Yields (key, example) tuples from the dataset
    with path.open() as f:
      for row in csv.DictReader(f):
        key = row['eventNo']
        yield key, {
          'eventNo': row['eventNo'],
          'time_label': row['time_label'],
          'signal_data': np.fromstring(row['signal_data'].replace("[", "").replace("]", "").replace("\n", ""), dtype = np.float64, sep = ' '),
      }
