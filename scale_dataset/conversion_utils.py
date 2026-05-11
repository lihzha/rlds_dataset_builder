import itertools
from functools import partial
from multiprocessing import Pool
from typing import Any, Callable, Dict, Iterable, Tuple, Union

import numpy as np
import tensorflow_datasets as tfds
from tensorflow_datasets.core import (
    dataset_builder,
    download,
    example_serializer,
    file_adapters,
    naming,
    utils,
)
from tensorflow_datasets.core import split_builder as split_builder_lib
from tensorflow_datasets.core import splits as splits_lib
from tensorflow_datasets.core import writer as writer_lib

Key = Union[str, int]
# The nested example dict passed to `features.encode_example`
Example = Dict[str, Any]
KeyExample = Tuple[Key, Example]


class MultiThreadedDatasetBuilder(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for example dataset."""

    N_WORKERS = 10  # number of parallel workers for data conversion
    MAX_PATHS_IN_MEMORY = (
        100  # number of paths converted & stored in memory before writing to disk
    )
    # -> the higher the faster / more parallel conversion, adjust based on avilable RAM
    # note that one path may yield multiple episodes and adjust accordingly
    PARSE_FCN = None  # needs to be filled with path-to-record-episode parse function

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        split_paths = self._split_paths()
        return {
            split: type(self).PARSE_FCN(paths=split_paths[split])
            for split in split_paths
        }

    def _generate_examples(self):
        pass  # this is implemented in global method to enable multiprocessing

    def _download_and_prepare(  # pytype: disable=signature-mismatch  # overriding-parameter-type-checks
        self,
        dl_manager: download.DownloadManager,
        download_config: download.DownloadConfig,
    ) -> None:
        """Generate all splits and returns the computed split infos."""
        assert self.PARSE_FCN is not None  # need to overwrite parse function
        split_builder = ParallelSplitBuilder(
            split_dict=self.info.splits,
            features=self.info.features,
            dataset_size=self.info.dataset_size,
            max_examples_per_split=download_config.max_examples_per_split,
            beam_options=download_config.beam_options,
            beam_runner=download_config.beam_runner,
            file_format=self.info.file_format,
            shard_config=download_config.get_shard_config(),
            split_paths=self._split_paths(),
            parse_function=type(self).PARSE_FCN,
            n_workers=self.N_WORKERS,
            max_paths_in_memory=self.MAX_PATHS_IN_MEMORY,
        )
        split_generators = self._split_generators(dl_manager)
        split_generators = split_builder.normalize_legacy_split_generators(
            split_generators=split_generators,
            generator_fn=self._generate_examples,
            is_beam=False,
        )
        dataset_builder._check_split_names(split_generators.keys())

        # Start generating data for all splits
        path_suffix = file_adapters.ADAPTER_FOR_FORMAT[
            self.info.file_format
        ].FILE_SUFFIX

        split_info_futures = []
        for split_name, generator in utils.tqdm(
            split_generators.items(),
            desc="Generating splits...",
            unit=" splits",
            leave=False,
        ):
            filename_template = naming.ShardedFileTemplate(
                split=split_name,
                dataset_name=self.name,
                data_dir=self.data_path,
                filetype_suffix=path_suffix,
            )
            future = split_builder.submit_split_generation(
                split_name=split_name,
                generator=generator,
                filename_template=filename_template,
                disable_shuffling=self.info.disable_shuffling,
                nondeterministic_order=True,  # Required by newer tfds API
            )
            split_info_futures.append(future)

        # Finalize the splits (after apache beam completed, if it was used)
        split_infos = [future.result() for future in split_info_futures]

        # Update the info object with the splits.
        split_dict = splits_lib.SplitDict(split_infos)
        self.info.set_splits(split_dict)


class _SplitInfoFuture:
    """Future containing the `tfds.core.SplitInfo` result."""

    def __init__(self, callback: Callable[[], splits_lib.SplitInfo]):
        self._callback = callback

    def result(self) -> splits_lib.SplitInfo:
        return self._callback()


def parse_examples_from_generator(
    paths, fcn, split_name, total_num_examples, features, serializer
):
    generator = fcn(paths)
    outputs = []
    for sample in utils.tqdm(
        generator,
        desc=f"Generating {split_name} examples...",
        unit=" examples",
        total=total_num_examples,
        leave=False,
        mininterval=1.0,
    ):
        if sample is None:
            continue
        key, example = sample
        try:
            example = features.encode_example(example)
        except Exception as e:  # pylint: disable=broad-except
            utils.reraise(e, prefix=f"Failed to encode example:\n{example}\n")

        # Convert string keys to numeric hashes for shuffler compatibility
        if isinstance(key, str):
            key = hash(key) & 0x7FFFFFFFFFFFFFFF  # Ensure positive integer

        outputs.append((key, serializer.serialize_example(example)))
    return outputs


class ParallelSplitBuilder(split_builder_lib.SplitBuilder):
    def __init__(
        self,
        *args,
        split_paths,
        parse_function,
        n_workers,
        max_paths_in_memory,
        **kwargs,
    ):
        # Store custom parameters before calling parent init
        self._split_paths = split_paths
        self._parse_function = parse_function
        self._n_workers = n_workers
        self._max_paths_in_memory = max_paths_in_memory

        # Extract file_format and shard_config - store for later use in _build_from_generator
        file_format = kwargs.pop('file_format', None)
        # shard_config might be passed to parent, so get it but don't pop
        self._shard_config = kwargs.get('shard_config', None)
        self._file_format = file_format

        # Create example_writer if file_format is provided and example_writer not already present
        if 'example_writer' not in kwargs and file_format is not None:
            from tensorflow_datasets.core import writer as writer_lib
            kwargs['example_writer'] = writer_lib.ExampleWriter(file_format=file_format)

        # CRITICAL FIX: Reduce MAX_MEM_BUFFER_SIZE to force disk-based bucketing
        # This prevents OOM during shuffling by ensuring examples are written to disk buckets
        # instead of being sorted entirely in memory
        from tensorflow_datasets.core import shuffle
        original_max_mem = shuffle.MAX_MEM_BUFFER_SIZE
        shuffle.MAX_MEM_BUFFER_SIZE = 10 << 20  # 10MB instead of 1GB (further reduced for large datasets)
        print(f"Reduced shuffle buffer from {original_max_mem >> 20}MB to {shuffle.MAX_MEM_BUFFER_SIZE >> 20}MB to prevent OOM")

        # Call parent init with all required kwargs
        super().__init__(*args, **kwargs)

    def _build_from_generator(
        self,
        split_name: str,
        generator: Iterable[KeyExample],
        filename_template: naming.ShardedFileTemplate,
        disable_shuffling: bool,
    ) -> _SplitInfoFuture:
        """Split generator for example generators.

        Args:
          split_name: str,
          generator: Iterable[KeyExample],
          filename_template: Template to format the filename for a shard.
          disable_shuffling: Specifies whether to shuffle the examples,

        Returns:
          future: The future containing the `tfds.core.SplitInfo`.
        """
        total_num_examples = None
        serialized_info = self._features.get_serialized_info()

        # Create example_writer from file_format for newer tfds API
        from tensorflow_datasets.core import writer as writer_lib_import
        example_writer = writer_lib_import.ExampleWriter(file_format=self._file_format)

        writer = writer_lib.Writer(
            serializer=example_serializer.ExampleSerializer(serialized_info),
            filename_template=filename_template,
            hash_salt=split_name,
            disable_shuffling=disable_shuffling,
            example_writer=example_writer,
            shard_config=self._shard_config,
        )

        del generator  # use parallel generators instead
        paths = self._split_paths[split_name]
        path_lists = chunk_max(
            paths, self._n_workers, self._max_paths_in_memory
        )  # generate N file lists
        print(f"Generating with {self._n_workers} workers!")
        pool = Pool(processes=self._n_workers)
        for i, paths in enumerate(path_lists):
            print(f"Processing chunk {i + 1} of {len(path_lists)}.")
            results = pool.map(
                partial(
                    parse_examples_from_generator,
                    fcn=self._parse_function,
                    split_name=split_name,
                    total_num_examples=total_num_examples,
                    serializer=writer._serializer,
                    features=self._features,
                ),
                paths,
            )
            # write results to shuffler --> this will automatically offload to disk if necessary
            print("Writing conversion results...")
            for result in itertools.chain(*results):
                key, serialized_example = result
                writer._shuffler.add(key, serialized_example)
        pool.close()

        print("Finishing split conversion...")
        shard_lengths, total_size = writer.finalize()

        split_info = splits_lib.SplitInfo(
            name=split_name,
            shard_lengths=shard_lengths,
            num_bytes=total_size,
            filename_template=filename_template,
        )
        return _SplitInfoFuture(lambda: split_info)


def dictlist2listdict(DL):
    "Converts a dict of lists to a list of dicts"
    return [dict(zip(DL, t)) for t in zip(*DL.values())]


def chunks(l, n):
    """Yield n number of sequential chunks from l."""
    d, r = divmod(len(l), n)
    for i in range(n):
        si = (d + 1) * (i if i < r else r) + d * (0 if i < r else i - r)
        yield l[si : si + (d + 1 if i < r else d)]


def chunk_max(l, n, max_chunk_sum):
    out = []
    for _ in range(int(np.ceil(len(l) / max_chunk_sum))):
        out.append(list(chunks(l[:max_chunk_sum], n)))
        l = l[max_chunk_sum:]
    return out
